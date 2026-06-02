#!/usr/bin/env python3
"""Evaluate a prediction set with the FoodAgent advantage score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.advantage_metrics import food_agent_advantage_score, judge_advantage
from food_agent.vqa import VQAPrediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--food-metrics", type=Path, default=Path("outputs/results/food_state_metrics.json"))
    parser.add_argument("--spatial-metrics", type=Path, default=Path("outputs/results/spatial_context_metrics.json"))
    return parser.parse_args()


def load_predictions(path: Path) -> list[VQAPrediction]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(
            VQAPrediction(
                sample_id=obj["sample_id"],
                baseline=obj["baseline"],
                task_family=obj["task_family"],
                video_id=obj.get("video_id"),
                question=obj["question"],
                choices=obj["choices"],
                gold=int(obj["gold"]),
                prediction=int(obj["prediction"]),
                correct=bool(obj["correct"]),
                evidence_ids=list(obj.get("evidence_ids", [])),
                tool_calls=list(obj.get("tool_calls", [])),
                failure_type=obj.get("failure_type"),
            )
        )
    return rows


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    args = parse_args()
    preds = load_predictions(args.predictions)
    food_metrics = read_json(args.food_metrics)
    spatial_metrics = read_json(args.spatial_metrics)
    score = food_agent_advantage_score(preds, food_metrics, spatial_metrics)
    verdict = judge_advantage(score)
    output = {
        "score": score,
        "verdict": verdict,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

