#!/usr/bin/env python3
"""Run HD-EPIC experiments via persistent model server.

The model server loads SAM3/GroundingDINO once and keeps them in GPU memory.
This script sends questions to the server over a Unix socket.

Usage:
    # Start server first (in a separate tmux session):
    python scripts/model_server.py start

    # Run experiment:
    python scripts/run_experiment_v2.py --limit 2 --out outputs/results/exp_v5.json

    # Full evaluation:
    python scripts/run_experiment_v2.py --limit 0 --out outputs/results/full.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_server import send_request, server_status, SOCKET_PATH

BENCHMARK_DIR = Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main/vqa-benchmark")


def load_benchmark(category: str = None) -> list:
    """Load benchmark questions."""
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
                "id": key,
                "category": cat_name,
                "question": q["question"],
                "choices": q.get("choices", []),
                "correct_idx": q.get("correct_idx", -1),
                "video_id": video_id,
                "start_time": video_info.get("start_time", ""),
                "end_time": video_info.get("end_time", ""),
            })

    return questions


def run_experiment(
    category: str = None,
    limit: int = 2,
    output_file: str = None,
):
    """Run experiment via model server."""
    print("Loading benchmark...")
    questions = load_benchmark(category)
    print(f"Loaded {len(questions)} questions")

    if limit > 0:
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)

        sampled = []
        for cat, qs in by_cat.items():
            sampled.extend(qs[:limit])
        questions = sampled
        print(f"Sampled {len(questions)} questions ({limit} per category)")

    results = []
    correct = 0
    total = 0

    print(f"\nRunning {len(questions)} questions via model server...\n")
    print(f"{'#':>4} | {'Category':>30} | {'Pred':>5} | {'GT':>5} | {'OK':>3} | {'Conf':>5} | {'Tools':>5} | {'Time':>5}")
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
        print(f"{i+1:>4} | {q['category'][:30]:>30} | {pred_idx:>5} | {q['correct_idx']:>5} | {status:>3} | {conf:>5.2f} | {tools:>5} | {elapsed:>4.0f}s")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"][:100],
            "prediction": pred_answer[:100],
            "pred_idx": pred_idx,
            "correct_idx": q["correct_idx"],
            "is_correct": is_correct,
            "confidence": conf,
            "tool_calls": tools,
            "latency": round(elapsed, 1),
        })

        # Save after each question (for crash recovery)
        if output_file:
            _save_results(output_file, results, correct, total)

    acc = correct / total if total > 0 else 0
    print(f"\n{'='*85}")
    print(f"Results: {correct}/{total} = {acc:.1%}")

    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r["category"]
        cat_results[cat]["total"] += 1
        if r["is_correct"]:
            cat_results[cat]["correct"] += 1

    print(f"\nPer-category accuracy:")
    for cat in sorted(cat_results.keys()):
        c = cat_results[cat]
        cat_acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"  {cat}: {c['correct']}/{c['total']} = {cat_acc:.1%}")

    if output_file:
        _save_results(output_file, results, correct, total)
        print(f"\nSaved to {output_file}")

    return results


def _save_results(output_file: str, results: list, correct: int, total: int):
    """Save results to JSON."""
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r["category"]
        cat_results[cat]["total"] += 1
        if r["is_correct"]:
            cat_results[cat]["correct"] += 1

    report = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "per_category": {k: {"correct": v["correct"], "total": v["total"]} for k, v in cat_results.items()},
        "results": results,
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="HD-EPIC VQA Experiment (via model server)")
    parser.add_argument("--category", default=None, help="Specific category to evaluate")
    parser.add_argument("--limit", type=int, default=2, help="Questions per category (0=all)")
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    if not Path(SOCKET_PATH).exists():
        print("ERROR: Model server not running. Start it first:")
        print("  python scripts/model_server.py start")
        sys.exit(1)

    run_experiment(args.category, args.limit, args.out)


if __name__ == "__main__":
    main()
