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
    parser.add_argument("--out-file", default="")
    parser.add_argument("--append-jsonl", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    out_path = Path(args.out_file) if args.out_file else None
    completed_ids: set[str] = set()
    if out_path and args.resume and out_path.exists():
        completed_ids = load_completed_ids(out_path, jsonl=args.append_jsonl)
    outputs = []
    for row in rows:
        if row["vqa_id"] in completed_ids:
            continue
        result = agent.answer_vqa_row(row, max_steps=args.max_steps)
        payload = {
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
        outputs.append(payload)
        if out_path:
            write_result(out_path, payload, jsonl=args.append_jsonl)
        print(
            f"{row['vqa_id']} pred={result.prediction} gold={row['correct_idx']} "
            f"correct={result.prediction == int(row['correct_idx'])}",
            flush=True,
        )
    if not out_path:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
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


def write_result(path: Path, payload: dict[str, object], *, jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return
    existing: list[dict[str, object]] = []
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(current, list):
                existing = [item for item in current if isinstance(item, dict)]
        except json.JSONDecodeError:
            existing = []
    existing.append(payload)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
