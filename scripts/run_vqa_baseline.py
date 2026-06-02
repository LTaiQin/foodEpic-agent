#!/usr/bin/env python3
"""Run first-pass VQA baselines and write predictions/metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from food_agent.vqa import VQAPrediction, compute_metrics, load_vqa_samples


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "vqa_baseline")
    parser.add_argument("--baseline", default="choice0", choices=["choice0", "oracle"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--task-family", default=None)
    return parser.parse_args()


def choose_prediction(sample, baseline: str) -> tuple[int, str | None]:
    if baseline == "oracle":
        return sample.correct_idx, None
    return 0, "reasoning_error"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_vqa_samples(args.index_dir, limit=args.limit, task_family=args.task_family)
    predictions: list[VQAPrediction] = []
    for sample in samples:
        pred_idx, failure_type = choose_prediction(sample, args.baseline)
        predictions.append(
            VQAPrediction(
                sample_id=sample.vqa_id,
                baseline=args.baseline,
                task_family=sample.task_family,
                video_id=sample.primary_video_id,
                question=sample.question,
                choices=sample.choices,
                gold=sample.correct_idx,
                prediction=pred_idx,
                correct=pred_idx == sample.correct_idx,
                evidence_ids=[],
                tool_calls=[],
                failure_type=None if pred_idx == sample.correct_idx else failure_type,
            )
        )
    pred_path = args.out_dir / f"predictions_{args.baseline}.jsonl"
    pred_path.write_text("\n".join(pred.to_json() for pred in predictions) + "\n", encoding="utf-8")
    metrics = compute_metrics(predictions)
    metrics_path = args.out_dir / f"metrics_{args.baseline}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"predictions: {pred_path}")
    print(f"metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

