#!/usr/bin/env python3
"""Run the graph agent on one sample from multiple task families."""

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


DEFAULT_TASKS = [
    "ingredient_ingredient_retrieval",
    "recipe_multi_step_localization",
    "object_motion_object_movement_itinerary",
    "object_motion_stationary_object_localization",
    "ingredient_ingredient_weight",
    "3d_perception_fixture_interaction_counting",
]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=(defaults.project_root / ".secrets" / "model.env").as_posix())
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--out-name", default="graph_agent_multitask_smoke.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    paths = ProjectPaths.from_env()
    df = pd.read_parquet(paths.output_root / "event_index" / "vqa_samples.parquet")
    rows: list[dict] = []
    for task_family in args.tasks:
        sub = df[df["task_family"] == task_family].head(1)
        if len(sub):
            rows.extend(sub.to_dict("records"))
    agent = GraphAgent(paths=paths)
    outputs = []
    for index, row in enumerate(rows, start=1):
        result = agent.answer_vqa_row(row, max_steps=args.max_steps)
        payload = {
            "vqa_id": row["vqa_id"],
            "task_family": row["task_family"],
            "video_id": row["primary_video_id"],
            "prediction": result.prediction,
            "gold": int(row["correct_idx"]),
            "correct": result.prediction == int(row["correct_idx"]),
            "confidence": result.confidence,
            "tool_trace": result.tool_trace,
            "working_memory": result.working_memory,
            "evidence_bundle": result.evidence_bundle,
            "retrieved_frames": result.retrieved_frames,
            "raw_model_output": result.raw_model_output,
        }
        outputs.append(payload)
        print(
            f"[{index}/{len(rows)}] task={row['task_family']} pred={result.prediction} gold={int(row['correct_idx'])} "
            f"correct={result.prediction == int(row['correct_idx'])} tools={[entry['tool'] for entry in result.tool_trace]}",
            flush=True,
        )
    out_path = paths.output_root / args.out_name
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
