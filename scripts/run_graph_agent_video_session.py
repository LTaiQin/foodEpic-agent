#!/usr/bin/env python3
"""Run the graph agent over multiple questions from one video with one persistent session."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "graph_agent_video_session")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    df = pd.read_parquet(paths.output_root / "event_index" / "vqa_samples.parquet")
    subset = df[df["primary_video_id"] == args.video_id].copy()
    if args.task_family:
        subset = subset[subset["task_family"] == args.task_family].copy()
    rows = subset.sort_values(["task_family", "vqa_id"]).head(args.limit).to_dict("records")

    run_name = build_run_name(args.video_id, args.task_family, args.limit)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions_video_session.jsonl"
    summary_path = run_dir / "session_summary.json"

    completed = load_completed_ids(pred_path) if args.resume else set()
    agent = GraphAgent(paths=paths)
    session = agent.begin_video_session(args.video_id)
    outputs: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        if row["vqa_id"] in completed:
            print(f"[{index}/{len(rows)}] skip_resume sample={row['vqa_id']}", flush=True)
            continue
        try:
            result = session.answer_vqa_row(row, max_steps=args.max_steps)
            payload = result.to_dict(gold=int(row["correct_idx"]), include_row=row)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "vqa_id": row["vqa_id"],
                "video_id": row["primary_video_id"],
                "task_family": row["task_family"],
                "prediction": None,
                "gold": int(row["correct_idx"]),
                "correct": False,
                "answer_text": "",
                "confidence": 0.0,
                "elapsed_seconds": None,
                "tool_trace": [],
                "evidence_bundle": [],
                "working_memory": [],
                "retrieved_frames": [],
                "raw_model_output": "",
                "question": row.get("question"),
                "choices_json": row.get("choices_json"),
                "inputs_json": row.get("inputs_json"),
                "failure_type": f"agent_error:{type(exc).__name__}",
                "failure_message": str(exc),
            }
        outputs.append(payload)
        append_jsonl(pred_path, payload)
        print(
            f"[{index}/{len(rows)}] sample={row['vqa_id']} pred={payload.get('prediction')} gold={int(row['correct_idx'])} "
            f"correct={payload.get('correct')} reuse={count_reuse_memory(payload)} tools={extract_tool_calls(payload)}",
            flush=True,
        )

    all_records = load_jsonl_records(pred_path)
    summary = build_session_summary(args.video_id, all_records, session.trace_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_run_name(video_id: str, task_family: str | None, limit: int) -> str:
    suffix = task_family or "all"
    return f"{video_id}_{suffix}_limit{limit}"


def extract_tool_calls(payload: dict[str, Any]) -> list[str]:
    trace = payload.get("tool_trace") or []
    if not isinstance(trace, list):
        return []
    return [str(item.get("tool")) for item in trace if isinstance(item, dict) and item.get("tool")]


def count_reuse_memory(payload: dict[str, Any]) -> int:
    working_memory = payload.get("working_memory") or []
    if not isinstance(working_memory, list):
        return 0
    return sum(1 for item in working_memory if isinstance(item, str) and item.startswith("reuse:"))


def load_completed_ids(path: Path) -> set[str]:
    return {str(item.get("vqa_id")) for item in load_jsonl_records(path) if item.get("vqa_id")}


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_session_summary(video_id: str, records: list[dict[str, Any]], session_trace_path: Path) -> dict[str, Any]:
    correct = sum(1 for item in records if item.get("correct") is True)
    task_counts = Counter(str(item.get("task_family")) for item in records if item.get("task_family"))
    tool_counts = Counter()
    reuse_total = 0
    for item in records:
        tool_counts.update(extract_tool_calls(item))
        reuse_total += count_reuse_memory(item)
    return {
        "video_id": video_id,
        "count": len(records),
        "correct": correct,
        "accuracy": (correct / len(records)) if records else None,
        "avg_reuse_memory_items": (reuse_total / len(records)) if records else 0.0,
        "task_family_counts": dict(task_counts),
        "tool_counts": dict(tool_counts),
        "session_trace_path": session_trace_path.as_posix(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
