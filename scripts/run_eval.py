#!/usr/bin/env python3
"""Evaluation script for the HD-EPIC multimodal agent.

Usage:
    python scripts/run_eval.py --benchmark path/to/benchmark.json --limit 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from food_agent.evaluation import (
    MimoClient,
    BenchmarkLoader,
    accuracy,
    accuracy_per_category,
    average_confidence,
    average_tool_calls,
    average_latency,
)
from food_agent.agent_v2 import MultimodalAgent
from food_agent.reasoning.tool_registry import ToolRegistry


def build_agent(mimo_client: MimoClient) -> MultimodalAgent:
    """Build the multimodal agent with all tools registered."""
    registry = ToolRegistry()

    # Register knowledge tools
    from food_agent.knowledge import NutritionKB, CommonSenseKB
    nk = NutritionKB()
    cs = CommonSenseKB()

    registry.register("query_nutrition_kb", lambda ingredient: nk.lookup(ingredient))
    registry.register("query_commonsense", lambda concept, relation="UsedFor": cs.get_related_concepts(concept, relation))

    return MultimodalAgent(mimo_client=mimo_client, tool_registry=registry)


def run_evaluation(
    benchmark_path: str,
    limit: int = 10,
    output_file: str = "eval_results.json",
    max_iterations: int = 5,
):
    """Run evaluation on the benchmark."""
    print(f"Loading benchmark from {benchmark_path}...")
    loader = BenchmarkLoader(benchmark_path)
    print(f"Loaded {loader.total_questions} questions in {len(loader.get_categories())} categories")

    mimo = MimoClient()
    agent = build_agent(mimo)
    agent.max_iterations = max_iterations

    questions = loader.get_questions()[:limit]
    results = []

    print(f"\nEvaluating {len(questions)} questions...\n")

    for i, q in enumerate(questions):
        question_text = q.get("question", q.get("text", ""))
        choices = q.get("choices", q.get("options", None))
        ground_truth = q.get("answer", q.get("correct_answer", ""))
        category = q.get("category", "general")
        video_id = q.get("video_id", "")
        participant_id = q.get("participant_id", video_id.split("-")[0] if "-" in video_id else "")

        print(f"[{i+1}/{len(questions)}] {category}: {question_text[:80]}...")

        start = time.time()
        try:
            result = agent.run(
                question=question_text,
                video_id=video_id,
                participant_id=participant_id,
                choices=choices,
            )
        except Exception as e:
            result = {"answer": f"Error: {e}", "confidence": 0, "iterations": 0}
        latency = time.time() - start

        prediction = result.get("answer", "")
        is_correct = prediction == ground_truth

        results.append({
            "question_id": q.get("id", i),
            "question": question_text,
            "category": category,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "confidence": result.get("confidence", 0),
            "iterations": result.get("iterations", 0),
            "tool_calls": len(result.get("tool_calls", [])),
            "latency": round(latency, 2),
        })

        status = "OK" if is_correct else "MISS"
        print(f"  {status} | pred={prediction[:50]} | gt={ground_truth[:50]} | conf={result.get('confidence', 0):.2f}")

    # Compute metrics
    preds = [r["prediction"] for r in results]
    gts = [r["ground_truth"] for r in results]
    cats = [r["category"] for r in results]

    overall_acc = accuracy(preds, gts)
    per_cat_acc = accuracy_per_category(preds, gts, cats)
    avg_conf = average_confidence(results)
    avg_tc = average_tool_calls(results)
    avg_lat = average_latency(results)

    report = {
        "total_questions": len(results),
        "overall_accuracy": round(overall_acc, 4),
        "per_category_accuracy": {k: round(v, 4) for k, v in per_cat_acc.items()},
        "average_confidence": round(avg_conf, 4),
        "average_tool_calls": round(avg_tc, 2),
        "average_latency_seconds": round(avg_lat, 2),
        "results": results,
    }

    # Save
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total questions: {len(results)}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Average confidence: {avg_conf:.4f}")
    print(f"Average tool calls: {avg_tc:.2f}")
    print(f"Average latency: {avg_lat:.2f}s")
    print(f"\nPer-category accuracy:")
    for cat, acc in sorted(per_cat_acc.items()):
        print(f"  {cat}: {acc:.4f}")
    print(f"\nResults saved to {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="HD-EPIC Agent Evaluation")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark file or directory")
    parser.add_argument("--limit", type=int, default=10, help="Max questions to evaluate")
    parser.add_argument("--output", default="eval_results.json", help="Output file")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max agent iterations")
    args = parser.parse_args()

    run_evaluation(args.benchmark, args.limit, args.output, args.max_iterations)


if __name__ == "__main__":
    main()
