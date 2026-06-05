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
    parser.add_argument("--append-jsonl", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    out_path = paths.output_root / args.out_name
    completed_ids: set[str] = set()
    outputs: list[dict[str, object]] = []
    if args.resume and out_path.exists():
        completed_ids = load_completed_ids(out_path, jsonl=args.append_jsonl)
        if not args.append_jsonl:
            outputs = load_existing_records(out_path)
    for index, row in enumerate(rows, start=1):
        if row["vqa_id"] in completed_ids:
            print(f"[{index}/{len(rows)}] task={row['task_family']} skip_resume sample={row['vqa_id']}", flush=True)
            continue
        result = agent.answer_vqa_row(row, max_steps=args.max_steps)
        payload = result.to_dict(gold=int(row["correct_idx"]), include_row=row)
        outputs.append(payload)
        write_result(out_path, payload, jsonl=args.append_jsonl)
        print(
            f"[{index}/{len(rows)}] task={row['task_family']} pred={result.prediction} gold={int(row['correct_idx'])} "
            f"correct={result.prediction == int(row['correct_idx'])} tools={[entry['tool'] for entry in result.tool_trace]}",
            flush=True,
        )
    if not args.append_jsonl:
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    return 0


def load_completed_ids(path: Path, *, jsonl: bool) -> set[str]:
    if not path.exists():
        return set()
    if jsonl:
        completed: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("vqa_id"):
                completed.add(str(payload["vqa_id"]))
        return completed
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item["vqa_id"]) for item in payload if isinstance(item, dict) and item.get("vqa_id")}


def load_existing_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def write_result(path: Path, payload: dict[str, object], *, jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return


if __name__ == "__main__":
    raise SystemExit(main())
