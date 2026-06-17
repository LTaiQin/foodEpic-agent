#!/usr/bin/env python3
"""End-to-end agent test: run the full pipeline on real HD-EPIC data.

Usage:
    python scripts/run_agent_e2e.py --video-id P01-20240202-110250 --question "What ingredients are visible?"
    python scripts/run_agent_e2e.py --video-id P01-20240202-110250 --limit 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from food_agent.agent_v2.pipeline import Pipeline


def run_single_question(pipeline: Pipeline, video_id: str, question: str, choices=None):
    """Run agent on a single question."""
    print(f"\n{'='*60}")
    print(f"Video: {video_id}")
    print(f"Question: {question}")
    if choices:
        print(f"Choices: {choices}")
    print(f"{'='*60}\n")

    start = time.time()
    result = pipeline.answer(
        question=question,
        video_id=video_id,
        choices=choices,
    )
    elapsed = time.time() - start

    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Category: {result['category']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {len(result['tool_calls'])}")
    print(f"Evidence items: {len(result['evidence_chain'])}")
    print(f"Time: {elapsed:.1f}s")

    if result['tool_calls']:
        print(f"\nTool call history:")
        for tc in result['tool_calls']:
            print(f"  [{tc['iteration']}] {tc['tool']}({json.dumps(tc['parameters'], default=str)[:80]})")

    return result


def run_demo_questions(pipeline: Pipeline, video_id: str):
    """Run a set of demo questions covering different categories."""
    questions = [
        {"question": "What food ingredients can you see in this scene?", "category": "ingredient"},
        {"question": "Where is the kitchen sink located relative to the wearer?", "category": "3d_perception"},
        {"question": "What sound events are happening in this time range?", "category": "audio"},
        {"question": "What is the person doing with their hands?", "category": "fine_grained_action"},
        {"question": "What is the person looking at?", "category": "gaze"},
    ]

    results = []
    for q in questions:
        result = run_single_question(pipeline, video_id, q["question"])
        results.append({**q, **result})

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for r in results:
        print(f"  [{r['category']}] conf={r['confidence']:.2f} tools={len(r['tool_calls'])} | {r['answer'][:60]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end agent test")
    parser.add_argument("--video-id", default="P01-20240202-110250", help="HD-EPIC video ID")
    parser.add_argument("--question", default=None, help="Single question to ask")
    parser.add_argument("--limit", type=int, default=5, help="Number of demo questions")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipeline = Pipeline()

    if args.question:
        result = run_single_question(pipeline, args.video_id, args.question)
        results = [result]
    else:
        results = run_demo_questions(pipeline, args.video_id)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
