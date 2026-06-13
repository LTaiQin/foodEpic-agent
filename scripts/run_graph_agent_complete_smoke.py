#!/usr/bin/env python3
"""Run a small end-to-end smoke suite for the complete graph agent on one video."""

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


DEFAULT_OPEN_QUERIES = [
    {
        "task_family": "open_query_temporal_summary",
        "question": "What happened after the ingredient was added?",
        "inputs_json": "{}",
    },
    {
        "task_family": "open_query_location",
        "question": "Where is the bowl at this moment?",
        "inputs_json": '{"anchor_time":"00:00:10.000"}',
    },
]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--task-families", nargs="*", default=["recipe_step_recognition", "ingredient_ingredient_weight"])
    parser.add_argument("--vqa-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "graph_agent_complete_smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-session", action="store_true")
    parser.add_argument("--rebuild-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)
    if args.rebuild_graph:
        agent.rebuild_video_graph(args.video_id)
    if args.reset_session:
        agent.reset_video_session(args.video_id)
    session = agent.begin_video_session(args.video_id)

    run_name = build_run_name(args.video_id, args.task_families, args.vqa_limit)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "complete_smoke_records.jsonl"
    summary_path = run_dir / "complete_smoke_summary.json"

    completed = load_completed_ids(pred_path) if args.resume else set()
    records: list[dict[str, Any]] = []

    for item in load_vqa_rows(paths=paths, video_id=args.video_id, task_families=args.task_families, limit=args.vqa_limit):
        record_id = str(item["vqa_id"])
        if record_id in completed:
            print(f"[VQA] skip_resume {record_id}", flush=True)
            continue
        try:
            result = session.answer_vqa_row(item, max_steps=args.max_steps)
            payload = result.to_dict(gold=int(item["correct_idx"]), include_row=item)
            payload["mode"] = "vqa"
        except Exception as exc:  # noqa: BLE001
            payload = build_failure_payload(item, mode="vqa", exc=exc)
        append_jsonl(pred_path, payload)
        records.append(payload)
        print(
            f"[VQA] {record_id} pred={payload.get('prediction')} correct={payload.get('correct')} "
            f"tools={extract_tool_calls(payload)}",
            flush=True,
        )

    for index, query in enumerate(DEFAULT_OPEN_QUERIES, start=1):
        query_id = f"open_query_smoke:{args.video_id}:{index}"
        if query_id in completed:
            print(f"[OPEN] skip_resume {query_id}", flush=True)
            continue
        try:
            result = session.answer_open_query(
                question=query["question"],
                inputs_json=query["inputs_json"],
                task_family=query["task_family"],
                max_steps=args.max_steps,
                query_id=query_id,
            )
            payload = result.to_dict(
                include_row={
                    "question": query["question"],
                    "choices_json": json.dumps(["OPEN_ENDED_RESPONSE"], ensure_ascii=False),
                    "inputs_json": query["inputs_json"],
                }
            )
            payload["mode"] = "open_query"
        except Exception as exc:  # noqa: BLE001
            payload = build_failure_payload(
                {
                    "vqa_id": query_id,
                    "primary_video_id": args.video_id,
                    "task_family": query["task_family"],
                    "question": query["question"],
                    "inputs_json": query["inputs_json"],
                },
                mode="open_query",
                exc=exc,
            )
        append_jsonl(pred_path, payload)
        records.append(payload)
        print(
            f"[OPEN] {query_id} answer={str(payload.get('answer_text') or '')[:80]} "
            f"tools={extract_tool_calls(payload)}",
            flush=True,
        )

    all_records = load_jsonl_records(pred_path)
    summary = build_summary(args.video_id, all_records, session.trace_path, session.state_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_run_name(video_id: str, task_families: list[str], vqa_limit: int) -> str:
    suffix = "-".join(task_families) if task_families else "default"
    return f"{video_id}_{suffix}_vqa{vqa_limit}"


def load_vqa_rows(*, paths: ProjectPaths, video_id: str, task_families: list[str], limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(paths.output_root / "event_index" / "vqa_samples.parquet")
    subset = df[df["primary_video_id"] == video_id].copy()
    rows: list[dict[str, Any]] = []
    for task_family in task_families:
        task_rows = subset[subset["task_family"] == task_family].sort_values("vqa_id").head(limit)
        rows.extend(task_rows.to_dict("records"))
    return rows


def build_failure_payload(item: dict[str, Any], *, mode: str, exc: Exception) -> dict[str, Any]:
    return {
        "vqa_id": item.get("vqa_id"),
        "video_id": item.get("primary_video_id"),
        "task_family": item.get("task_family"),
        "prediction": None,
        "gold": item.get("correct_idx"),
        "correct": False,
        "answer_text": "",
        "confidence": 0.0,
        "elapsed_seconds": None,
        "tool_trace": [],
        "evidence_bundle": [],
        "working_memory": [],
        "retrieved_frames": [],
        "visited_times": [],
        "artifacts": [],
        "raw_model_output": "",
        "question": item.get("question"),
        "choices_json": item.get("choices_json"),
        "inputs_json": item.get("inputs_json"),
        "failure_type": f"agent_error:{type(exc).__name__}",
        "failure_message": str(exc),
        "mode": mode,
    }


def extract_tool_calls(payload: dict[str, Any]) -> list[str]:
    trace = payload.get("tool_trace") or []
    if not isinstance(trace, list):
        return []
    return [str(item.get("tool")) for item in trace if isinstance(item, dict) and item.get("tool")]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def load_completed_ids(path: Path) -> set[str]:
    return {str(item.get("vqa_id")) for item in load_jsonl_records(path) if item.get("vqa_id")}


def build_summary(video_id: str, records: list[dict[str, Any]], trace_path: Path, state_path: Path) -> dict[str, Any]:
    mode_counts = Counter(str(item.get("mode")) for item in records if item.get("mode"))
    task_counts = Counter(str(item.get("task_family")) for item in records if item.get("task_family"))
    tool_counts = Counter()
    planner_relation_expansion_count = 0
    seed_relation_expansion_count = 0
    verifier_count = 0
    reuse_relation_count = 0
    reuse_relation_edge_count = 0
    planner_override_count = 0
    verifier_blocked_finish_count = 0
    freeform_structured_summary_count = 0
    open_query_fallback_count = 0
    reuse_time_anchor_count = 0
    artifact_reuse_count = 0
    evidence_report_count = 0
    tool_failure_count = 0
    ineffective_tool_count = 0
    failed_tool_recovery_count = 0
    ineffective_tool_avoidance_count = 0
    for item in records:
        tool_counts.update(extract_tool_calls(item))
        working_memory = item.get("working_memory") or []
        if isinstance(working_memory, list):
            reuse_relation_count += sum(
                1 for entry in working_memory if isinstance(entry, str) and entry.startswith("reuse_relation:")
            )
            reuse_relation_edge_count += sum(
                1 for entry in working_memory if isinstance(entry, str) and entry.startswith("reuse_relation_edge:")
            )
            planner_override_count += sum(
                1 for entry in working_memory if isinstance(entry, str) and entry.startswith("planner_override ")
            )
            verifier_blocked_finish_count += sum(
                1
                for entry in working_memory
                if isinstance(entry, str) and entry.startswith("planner_override verifier_blocked_finish=")
            )
            freeform_structured_summary_count += sum(
                1
                for entry in working_memory
                if isinstance(entry, str)
                and entry in {"freeform_answer_mode=structured_summary", "freeform_answer_mode=grounded_structured_answer"}
            )
            open_query_fallback_count += sum(
                1
                for entry in working_memory
                if isinstance(entry, str) and entry == "freeform_answer_mode=fallback_summary"
            )
            reuse_time_anchor_count += sum(
                1
                for entry in working_memory
                if isinstance(entry, str) and entry.startswith("planner_override verifier_blocked_finish=finish -> extract_frame_at_time")
            )
            for entry in working_memory:
                if not isinstance(entry, str):
                    continue
                if entry.startswith("relation_seed_expanded_node_count="):
                    seed_relation_expansion_count += 1
        artifacts = item.get("artifacts") or []
        if isinstance(artifacts, list):
            artifact_reuse_count += len([entry for entry in artifacts if isinstance(entry, str) and entry])
        if item.get("evidence_report_path"):
            evidence_report_count += 1
        trace = item.get("tool_trace") or []
        if isinstance(trace, list):
            planner_relation_expansion_count += sum(
                1 for entry in trace if isinstance(entry, dict) and entry.get("tool") == "expand_graph_context"
            )
            failed_tool_recovery_count += count_recovery_events(trace, failure_key="tool_failed")
            ineffective_tool_avoidance_count += count_recovery_events(trace, failure_key="tool_ineffective")
        verifier_count += int(item.get("verification_count") or 0)
        tool_failure_count += int(item.get("failure_count") or len(item.get("tool_failures") or []))
        ineffective_tool_count += int(item.get("ineffective_tool_count") or len(item.get("ineffective_tools") or []))
    vqa_records = [item for item in records if item.get("mode") == "vqa"]
    vqa_correct = sum(1 for item in vqa_records if item.get("correct") is True)
    return {
        "video_id": video_id,
        "count": len(records),
        "mode_counts": dict(mode_counts),
        "task_family_counts": dict(task_counts),
        "vqa_accuracy": (vqa_correct / len(vqa_records)) if vqa_records else None,
        "tool_counts": dict(tool_counts),
        "planner_relation_expansion_count": planner_relation_expansion_count,
        "seed_relation_expansion_count": seed_relation_expansion_count,
        "relation_expansion_count": planner_relation_expansion_count,
        "reuse_relation_count": reuse_relation_count,
        "reuse_relation_edge_count": reuse_relation_edge_count,
        "verifier_count": verifier_count,
        "planner_override_count": planner_override_count,
        "verifier_blocked_finish_count": verifier_blocked_finish_count,
        "freeform_structured_summary_count": freeform_structured_summary_count,
        "open_query_fallback_count": open_query_fallback_count,
        "reuse_time_anchor_count": reuse_time_anchor_count,
        "artifact_reuse_count": artifact_reuse_count,
        "evidence_report_count": evidence_report_count,
        "tool_failure_count": tool_failure_count,
        "ineffective_tool_count": ineffective_tool_count,
        "failed_tool_recovery_count": failed_tool_recovery_count,
        "ineffective_tool_avoidance_count": ineffective_tool_avoidance_count,
        "session_trace_path": trace_path.as_posix(),
        "session_state_path": state_path.as_posix(),
    }


def count_recovery_events(trace: list[dict[str, Any]], *, failure_key: str) -> int:
    recoveries = 0
    pending_failed_tool = ""
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        tool_name = str(entry.get("tool") or "")
        raw_result = entry.get("raw_result")
        if isinstance(raw_result, dict) and raw_result.get(failure_key):
            if tool_name:
                pending_failed_tool = tool_name
            continue
        if pending_failed_tool:
            if not tool_name or tool_name == pending_failed_tool:
                continue
            recoveries += 1
            pending_failed_tool = ""
    return recoveries


if __name__ == "__main__":
    raise SystemExit(main())
