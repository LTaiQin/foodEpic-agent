#!/usr/bin/env python3
"""Focused evaluation on a single question category.

Usage:
    python scripts/eval_category.py --category 3d_perception_object_location --num 20
    python scripts/eval_category.py --category 3d_perception_object_location --num 20 --resume
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_server import send_request, SOCKET_PATH

BENCHMARK_DIR = Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main/vqa-benchmark")


def load_category(category: str, num: int, seed: int = 42) -> list:
    """Load and sample questions from a category, ensuring video diversity."""
    f = BENCHMARK_DIR / f"{category}.json"
    if not f.exists():
        print(f"Category file not found: {f}")
        return []

    with open(f) as fh:
        data = json.load(fh)

    # Group by video
    by_video = defaultdict(list)
    for key, q in data.items():
        vid = q.get("inputs", {}).get("video 1", {}).get("id", "")
        by_video[vid].append({
            "id": key,
            "category": category,
            "question": q["question"],
            "choices": q.get("choices", []),
            "correct_idx": q.get("correct_idx", -1),
            "video_id": vid,
        })

    # Sample: pick from as many different videos as possible
    random.seed(seed)
    videos = list(by_video.keys())
    random.shuffle(videos)

    sampled = []
    # Round-robin across videos
    video_iters = {v: iter(qs) for v, qs in by_video.items()}
    while len(sampled) < num and video_iters:
        to_remove = []
        for v in list(video_iters.keys()):
            if len(sampled) >= num:
                break
            try:
                sampled.append(next(video_iters[v]))
            except StopIteration:
                to_remove.append(v)
        for v in to_remove:
            del video_iters[v]

    random.shuffle(sampled)
    return sampled[:num]


def run(category: str, num: int, output_file: str, resume: bool = False):
    """Run evaluation on a single category."""
    questions = load_category(category, num)
    if not questions:
        print("No questions found")
        return

    print(f"Category: {category}")
    print(f"Total questions: {len(questions)}")
    print(f"Unique videos: {len(set(q['video_id'] for q in questions))}")

    results = []
    correct = 0
    total = 0

    # Resume from existing file
    existing_ids = set()
    if resume and output_file and Path(output_file).exists():
        try:
            with open(output_file) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            correct = existing.get("correct", 0)
            total = existing.get("total", 0)
            existing_ids = {r["id"] for r in results}
            print(f"Resuming: {len(results)} already done, {correct}/{total} correct")
        except Exception:
            pass

    questions = [q for q in questions if q["id"] not in existing_ids]
    print(f"Remaining: {len(questions)}\n")

    print(f"{'#':>4} | {'Video':>25} | {'Pred':>5} | {'GT':>5} | {'OK':>3} | {'Conf':>5} | {'Tools':>5} | {'Time':>5}")
    print("-" * 85)

    for i, q in enumerate(questions):
        prompt = q["question"]
        if q["choices"]:
            choice_text = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(q["choices"]))
            prompt += (
                f"\n\n{choice_text}\n\n"
                "Select the best option. Reply with ONLY the letter (A, B, C, D, or E), nothing else."
            )

        start = time.time()
        try:
            result = send_request({
                "action": "answer",
                "question": prompt,
                "video_id": q["video_id"],
                "choices": q["choices"],
            })
        except Exception as e:
            result = {"answer": f"Error: {e}", "confidence": 0, "tool_calls": [], "iterations": 0}
        elapsed = time.time() - start

        pred_answer = result.get("answer", "")
        if isinstance(pred_answer, list):
            pred_answer = pred_answer[0] if pred_answer else ""
        pred_answer = str(pred_answer).strip()
        pred_idx = -1
        if q["choices"]:
            for j, choice in enumerate(q["choices"]):
                choice_str = str(choice).strip() if not isinstance(choice, str) else choice
                if pred_answer == choice_str or pred_answer.lower() == choice_str.lower():
                    pred_idx = j
                    break
            if pred_idx == -1:
                for j, choice in enumerate(q["choices"]):
                    choice_str = str(choice).strip() if not isinstance(choice, str) else choice
                    if choice_str[:30].lower() in pred_answer.lower() or pred_answer[:30].lower() in choice_str.lower():
                        pred_idx = j
                        break

        is_correct = pred_idx == q["correct_idx"]
        if is_correct:
            correct += 1
        total += 1

        tools = len(result.get("tool_calls", []))
        conf = result.get("confidence", 0)

        status = "OK" if is_correct else "MISS"
        vid_short = q["video_id"][-15:]
        print(f"{total:>4} | {vid_short:>25} | {pred_idx:>5} | {q['correct_idx']:>5} | {status:>3} | {conf:>5.2f} | {tools:>5} | {elapsed:>4.0f}s")

        # Show wrong answers for debugging
        if not is_correct and pred_answer:
            print(f"      Pred text: {pred_answer[:80]}")
            print(f"      Expected:  {q['choices'][q['correct_idx']] if q['choices'] else 'N/A'}")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"][:200],
            "video_id": q["video_id"],
            "prediction": pred_answer[:200],
            "pred_idx": pred_idx,
            "correct_idx": q["correct_idx"],
            "choices": q["choices"],
            "is_correct": is_correct,
            "confidence": conf,
            "tool_calls": tools,
            "latency": round(elapsed, 1),
        })

        # Save after each question
        if output_file:
            _save(output_file, results, correct, total)

    acc = correct / total if total > 0 else 0
    print(f"\n{'='*85}")
    print(f"Result: {correct}/{total} = {acc:.1%}")

    if output_file:
        _save(output_file, results, correct, total)
        print(f"Saved to {output_file}")


def _save(output_file, results, correct, total):
    acc = correct / total if total > 0 else 0
    report = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "results": results,
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.out:
        args.out = f"outputs/results/cat_{args.category}.json"

    if not Path(SOCKET_PATH).exists():
        print("ERROR: Model server not running.")
        sys.exit(1)

    run(args.category, args.num, args.out, args.resume)


if __name__ == "__main__":
    main()
