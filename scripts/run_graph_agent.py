#!/usr/bin/env python3
"""Run the graph-based agent on a single VQA row or a sampled subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=(defaults.project_root / ".secrets" / "model.env").as_posix())
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--task-family", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    paths = ProjectPaths.from_env()
    df = pd.read_parquet(paths.output_root / "event_index" / "vqa_samples.parquet")
    video_df = df[df["primary_video_id"] == args.video_id].copy()
    if args.task_family:
        video_df = video_df[video_df["task_family"] == args.task_family].copy()
    rows = video_df.sort_values(["task_family", "vqa_id"]).head(args.limit).to_dict("records")
    agent = GraphAgent()
    outputs = []
    for row in rows:
        result = agent.answer_vqa_row(row, max_steps=args.max_steps)
        outputs.append(
            {
                "vqa_id": row["vqa_id"],
                "task_family": row["task_family"],
                "prediction": result.prediction,
                "gold": row["correct_idx"],
                "correct": result.prediction == int(row["correct_idx"]),
                "tool_trace": result.tool_trace,
                "evidence_bundle": result.evidence_bundle,
                "working_memory": result.working_memory,
                "retrieved_frames": result.retrieved_frames,
                "confidence": result.confidence,
                "raw_model_output": result.raw_model_output,
            }
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
