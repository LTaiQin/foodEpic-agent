#!/usr/bin/env python3
"""Run experiments on HD-EPIC VQA benchmark.

Usage:
    # Quick test (5 questions per category)
    python scripts/run_experiment.py --limit 5

    # Single category
    python scripts/run_experiment.py --category fine_grained_action_recognition --limit 10

    # Full evaluation
    python scripts/run_experiment.py --limit 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from food_agent.agent_v2.pipeline import Pipeline

BENCHMARK_DIR = Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main/vqa-benchmark")

# Category to question type mapping
CATEGORY_MAP = {
    "3d_perception_fixture_interaction_counting": "3d_perception",
    "3d_perception_fixture_location": "3d_perception",
    "3d_perception_object_contents_retrieval": "3d_perception",
    "3d_perception_object_location": "3d_perception",
    "fine_grained_action_localization": "fine_grained_action",
    "fine_grained_action_recognition": "fine_grained_action",
    "fine_grained_how_recognition": "fine_grained_action",
    "fine_grained_why_recognition": "fine_grained_action",
    "gaze_gaze_estimation": "gaze",
    "gaze_interaction_anticipation": "gaze",
    "ingredient_food_detection": "ingredient",
    "ingredient_food_state_change": "ingredient",
    "nutrition_caloric_estimation": "nutrition",
    "nutrition_ingredient_quantity": "nutrition",
    "object_motion_absolute_state": "object_motion",
    "object_motion_object_interaction": "object_motion",
    "object_motion_relative_motion": "object_motion",
    "recipe_procedural_step": "recipe",
    "recipe_step_ordering": "recipe",
}


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


def parse_time(time_str: str) -> float:
    """Parse HH:MM:SS.mmm to seconds."""
    if not time_str:
        return 0
    parts = time_str.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0


def run_experiment(
    category: str = None,
    limit: int = 5,
    max_iterations: int = 3,
    output_file: str = None,
):
    """Run experiment on benchmark questions."""
    print(f"Loading benchmark...")
    questions = load_benchmark(category)
    print(f"Loaded {len(questions)} questions")

    if limit > 0:
        # Sample evenly across categories
        from collections import defaultdict
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)

        sampled = []
        for cat, qs in by_cat.items():
            sampled.extend(qs[:limit])
        questions = sampled
        print(f"Sampled {len(questions)} questions ({limit} per category)")

    print(f"Creating pipeline (light mode, no SAM3)...")
    pipeline = Pipeline(load_models=False)
    pipeline.agent.max_iterations = max_iterations
    pipeline.agent.timeout = 120

    results = []
    correct = 0
    total = 0

    print(f"\nRunning {len(questions)} questions...\n")
    print(f"{'#':>4} | {'Category':>30} | {'Pred':>5} | {'GT':>5} | {'OK':>3} | {'Conf':>5} | {'Tools':>5} | {'Time':>5}")
    print("-" * 85)

    for i, q in enumerate(questions):
        # Build prompt with choices - ask for letter only
        prompt = q["question"]
        if q["choices"]:
            choice_text = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(q["choices"]))
            prompt += (
                f"\n\n{choice_text}\n\n"
                "Select the best option. Reply with ONLY the letter (A, B, C, D, or E), nothing else."
            )

        start = time.time()
        try:
            result = pipeline.answer(
                question=prompt,
                video_id=q["video_id"],
                choices=q["choices"],
            )
        except Exception as e:
            result = {"answer": f"Error: {e}", "confidence": 0, "tool_calls": [], "iterations": 0}
        elapsed = time.time() - start

        # Parse prediction - the Generator now returns the full choice text
        pred_answer = result.get("answer", "")
        if isinstance(pred_answer, list):
            pred_answer = pred_answer[0] if pred_answer else ""
        pred_answer = str(pred_answer).strip()
        pred_idx = -1
        if q["choices"]:
            # The Generator.parse_answer should return the choice text directly
            # So we match it back to the index
            for j, choice in enumerate(q["choices"]):
                if pred_answer == choice or pred_answer.lower() == choice.lower():
                    pred_idx = j
                    break
            # Fallback: check if pred starts with the choice
            if pred_idx == -1:
                for j, choice in enumerate(q["choices"]):
                    if choice[:30].lower() in pred_answer.lower() or pred_answer[:30].lower() in choice.lower():
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

    # Summary
    acc = correct / total if total > 0 else 0
    print(f"\n{'='*85}")
    print(f"Results: {correct}/{total} = {acc:.1%}")

    # Per-category
    from collections import defaultdict
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

    # Save
    if output_file:
        report = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "per_category": {k: {"correct": v["correct"], "total": v["total"]} for k, v in cat_results.items()},
            "results": results,
        }
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HD-EPIC VQA Experiment")
    parser.add_argument("--category", default=None, help="Specific category to evaluate")
    parser.add_argument("--limit", type=int, default=5, help="Questions per category (0=all)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max agent iterations")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    run_experiment(args.category, args.limit, args.max_iterations, args.output)


if __name__ == "__main__":
    main()
