#!/usr/bin/env python3
"""Run HD-EPIC experiments via persistent model server (with parallel support).

Usage:
    python scripts/run_experiment_v2.py --limit 3 --out outputs/results/exp_v10.json --parallel 4
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_server import send_request, SOCKET_PATH

BENCHMARK_DIR = Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main/vqa-benchmark")


def load_benchmark(category: str = None) -> list:
    questions = []
    files = sorted(BENCHMARK_DIR.glob("*.json"))
    for f in files:
        cat_name = f.stem
        if category and cat_name != category:
            continue
        with open(f) as fh:
            data = json.load(fh)
        for key, q in data.items():
            video_info = q.get("inputs", {}).get("video 1", {})
            video_id = video_info.get("id", "")
            questions.append({
                "id": key, "category": cat_name, "question": q["question"],
                "choices": q.get("choices", []), "correct_idx": q.get("correct_idx", -1),
                "video_id": video_id,
            })
    return questions


def answer_question(q: dict) -> dict:
    prompt = q["question"]
    if q["choices"]:
        choice_text = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(q["choices"]))
        prompt += f"\n\n{choice_text}\n\nSelect the best option. Reply with ONLY the letter (A, B, C, D, or E), nothing else."

    start = time.time()
    try:
        result = send_request({
            "action": "answer", "question": prompt,
            "video_id": q["video_id"], "choices": q["choices"],
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
            if pred_answer == choice or pred_answer.lower() == str(choice).lower():
                pred_idx = j
                break
        if pred_idx == -1:
            for j, choice in enumerate(q["choices"]):
                if str(choice)[:30].lower() in pred_answer.lower() or pred_answer[:30].lower() in str(choice).lower():
                    pred_idx = j
                    break

    return {
        "id": q["id"], "category": q["category"], "question": q["question"][:200],
        "video_id": q["video_id"], "prediction": pred_answer[:200], "pred_idx": pred_idx,
        "correct_idx": q["correct_idx"], "is_correct": pred_idx == q["correct_idx"],
        "confidence": result.get("confidence", 0), "tool_calls": len(result.get("tool_calls", [])),
        "latency": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not Path(SOCKET_PATH).exists():
        print("ERROR: Model server not running.")
        sys.exit(1)

    questions = load_benchmark(args.category)
    print(f"Loaded {len(questions)} questions")

    if args.limit > 0:
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        sampled = []
        for cat, qs in by_cat.items():
            sampled.extend(qs[:args.limit])
        questions = sampled
        print(f"Sampled {len(questions)} ({args.limit} per category)")

    results = []
    correct = 0
    total = 0

    # Resume support
    existing_ids = set()
    if args.resume and args.out and Path(args.out).exists():
        try:
            with open(args.out) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            correct = existing.get("correct", 0)
            total = existing.get("total", 0)
            existing_ids = {r["id"] for r in results}
            print(f"Resuming: {len(results)} done")
        except Exception:
            pass

    questions = [q for q in questions if q["id"] not in existing_ids]
    print(f"Remaining: {len(questions)}\n")

    print(f"{'#':>4} | {'Category':>30} | {'Pred':>5} | {'GT':>5} | {'OK':>3} | {'Time':>5}")
    print("-" * 65)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_q = {executor.submit(answer_question, q): q for q in questions}
        for future in as_completed(future_to_q):
            result = future.result()
            results.append(result)
            total += 1
            if result["is_correct"]:
                correct += 1

            status = "OK" if result["is_correct"] else "MISS"
            print(f"{total:>4} | {result['category'][:30]:>30} | {result['pred_idx']:>5} | {result['correct_idx']:>5} | {status:>3} | {result['latency']:>4.0f}s")

            if args.out:
                _save(args.out, results, correct, total)

    elapsed = time.time() - start_time
    acc = correct / total if total > 0 else 0
    print(f"\n{'='*65}")
    print(f"Result: {correct}/{total} = {acc:.1%}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/total:.0f}s per question)")

    # Per-category
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_results[r["category"]]["total"] += 1
        if r["is_correct"]:
            cat_results[r["category"]]["correct"] += 1
    print(f"\nPer-category:")
    for cat in sorted(cat_results):
        c = cat_results[cat]
        print(f"  {cat}: {c['correct']}/{c['total']} ({c['correct']/c['total']*100:.0f}%)")

    if args.out:
        _save(args.out, results, correct, total)
        print(f"\nSaved to {args.out}")


def _save(output_file, results, correct, total):
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_results[r["category"]]["total"] += 1
        if r["is_correct"]:
            cat_results[r["category"]]["correct"] += 1

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "total": total, "correct": correct,
            "accuracy": correct / total if total else 0,
            "per_category": {k: {"correct": v["correct"], "total": v["total"]} for k, v in cat_results.items()},
            "results": results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
