#!/usr/bin/env python3
"""Run a small real VQA subset and produce trace-rich artifacts for acceptance audit."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths
from scripts.audit_graph_agent_mechanisms import build_audit_report
from scripts.run_graph_agent_batch import (
    RAW_REVISIT_TOOLS,
    STRUCTURED_QUERY_TOOLS,
    count_failed_tool_recoveries,
    count_ineffective_tool_avoidances,
    count_tools,
)


DEFAULT_TASK_FAMILIES = [
    "ingredient_ingredient_weight",
    "ingredient_ingredient_retrieval",
    "recipe_step_recognition",
]

DEFAULT_OPEN_QUERY_PROBES = [
    {
        "task_family": "open_query_location",
        "question": "Where is the main bowl at this moment?",
        "inputs_json": "{}",
    },
]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_real_subset_acceptance",
    )
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--task-family", action="append", default=None)
    parser.add_argument("--limit-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--sample-timeout-seconds", type=int, default=180)
    parser.add_argument("--include-open-query-probes", action="store_true")
    parser.add_argument("--open-query-max-steps", type=int, default=3)
    parser.add_argument("--open-query-timeout-seconds", type=int, default=90)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    report = run_real_subset_acceptance(
        index_file=args.index_file,
        out_dir=args.out_dir,
        video_id=args.video_id,
        task_families=args.task_family or DEFAULT_TASK_FAMILIES,
        limit_per_task=args.limit_per_task,
        max_steps=args.max_steps,
        sample_timeout_seconds=args.sample_timeout_seconds,
        include_open_query_probes=args.include_open_query_probes,
        open_query_max_steps=args.open_query_max_steps,
        open_query_timeout_seconds=args.open_query_timeout_seconds,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_real_subset_acceptance(
    *,
    index_file: Path,
    out_dir: Path,
    video_id: str | None,
    task_families: list[str],
    limit_per_task: int,
    max_steps: int,
    sample_timeout_seconds: int = 180,
    include_open_query_probes: bool = False,
    open_query_max_steps: int = 3,
    open_query_timeout_seconds: int = 90,
    resume: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions_graph_agent.jsonl"
    session_trace_path = out_dir / "session_trace.jsonl"
    session_state_path = out_dir / "session_state.json"
    summary_path = out_dir / "real_subset_summary.json"
    audit_path = out_dir / "real_subset_audit.json"
    progress_path = out_dir / "progress.json"
    current_sample_path = out_dir / "current_sample.json"
    heartbeat_path = out_dir / "current_sample_heartbeat.jsonl"

    completed = load_existing(predictions_path) if resume else {}
    df = pd.read_parquet(index_file)
    selected_rows = select_rows(df=df, video_id=video_id, task_families=task_families, limit_per_task=limit_per_task)
    if not selected_rows:
        raise RuntimeError("no VQA rows selected for real subset acceptance")
    probe_video_id = video_id or str(selected_rows[0].get("primary_video_id") or "")
    open_query_probes = build_open_query_probes(video_id=probe_video_id, anchor_row=selected_rows[0]) if include_open_query_probes else []

    records: list[dict[str, Any]] = []
    session_trace_rows: list[dict[str, Any]] = []
    session_state_payload: dict[str, Any] = {}
    update_progress(
        progress_path=progress_path,
        completed_records=list(completed.values()),
        total=len(selected_rows) + len(open_query_probes),
        current_sample_id=None,
        current_status="idle",
    )

    for index, row in enumerate(selected_rows, start=1):
        sample_id = str(row["vqa_id"])
        if sample_id in completed:
            payload = completed[sample_id]
            print(f"[{index}/{len(selected_rows)}] skip_resume sample={sample_id}", flush=True)
        else:
            update_progress(
                progress_path=progress_path,
                completed_records=list(completed.values()),
                total=len(selected_rows) + len(open_query_probes),
                current_sample_id=sample_id,
                current_status="running",
            )
            print(
                f"[{index}/{len(selected_rows)}] start sample={sample_id} video={row.get('primary_video_id')} "
                f"task={row.get('task_family')}",
                flush=True,
            )
            started_at = time.time()
            write_current_sample(
                current_sample_path=current_sample_path,
                payload={
                    "kind": "vqa",
                    "status": "running",
                    "index": index,
                    "total": len(selected_rows),
                    "vqa_id": sample_id,
                    "video_id": str(row.get("primary_video_id") or ""),
                    "task_family": str(row.get("task_family") or ""),
                    "started_at": started_at,
                },
            )
            try:
                payload = execute_sample_with_timeout(
                    row=row,
                    max_steps=max_steps,
                    timeout_seconds=sample_timeout_seconds,
                    heartbeat_path=heartbeat_path,
                )
            except SampleTimeoutError as exc:
                payload = build_failure_payload(
                    row=row,
                    failure_type="agent_timeout",
                    failure_message=str(exc),
                    heartbeat_path=heartbeat_path,
                )
            except Exception as exc:  # noqa: BLE001
                payload = build_failure_payload(
                    row=row,
                    failure_type=f"agent_error:{type(exc).__name__}",
                    failure_message=str(exc),
                    heartbeat_path=heartbeat_path,
                )
            append_jsonl(predictions_path, payload)
            completed[sample_id] = payload
            append_jsonl(session_trace_path, trace_row_from_payload(payload))
            update_session_state_snapshot(
                session_state_path=session_state_path,
                project_paths=ProjectPaths.from_env(),
                video_id=str(row.get("primary_video_id") or ""),
            )
            update_progress(
                progress_path=progress_path,
                completed_records=list(completed.values()),
                total=len(selected_rows) + len(open_query_probes),
                current_sample_id=sample_id,
                current_status="completed",
            )
            write_current_sample(
                current_sample_path=current_sample_path,
                payload={
                    "kind": "vqa",
                    "status": "finished",
                    "index": index,
                    "total": len(selected_rows),
                    "vqa_id": sample_id,
                    "video_id": str(row.get("primary_video_id") or ""),
                    "task_family": str(row.get("task_family") or ""),
                    "started_at": started_at,
                    "elapsed_seconds": round(time.time() - started_at, 2),
                    "failure_type": payload.get("failure_type"),
                    "prediction": payload.get("prediction"),
                    "correct": payload.get("correct"),
                    "tool_calls": payload.get("tool_calls") or [],
                },
            )
            persist_partial_artifacts(
                out_dir=out_dir,
                predictions_path=predictions_path,
                session_trace_path=session_trace_path,
                session_state_path=session_state_path,
                summary_path=summary_path,
                audit_path=audit_path,
                records=list(completed.values()),
                task_families=task_families,
                limit_per_task=limit_per_task,
                video_id=video_id,
            )
            print(
                f"[{index}/{len(selected_rows)}] sample={sample_id} pred={payload.get('prediction')} "
                f"gold={int(row['correct_idx'])} correct={payload.get('correct')} tools={payload.get('tool_calls')} "
                f"failure={payload.get('failure_type')} elapsed={round(time.time() - started_at, 2)}s",
                flush=True,
            )
        records.append(payload)
        if sample_id in completed and sample_id in load_existing(predictions_path):
            session_trace_rows.append(trace_row_from_payload(payload))
        if row.get("primary_video_id"):
            session_dir = ProjectPaths.from_env().graph_agent_sessions_root / str(row["primary_video_id"])
            maybe_state = session_dir / "session_state.json"
            if maybe_state.exists():
                try:
                    session_state_payload = json.loads(maybe_state.read_text(encoding="utf-8"))
                except Exception:
                    session_state_payload = {}

    base_total = len(selected_rows) + len(open_query_probes)
    for index, query in enumerate(open_query_probes, start=1):
        query_id = str(query["query_id"])
        if query_id in completed:
            payload = completed[query_id]
            print(f"[OPEN {index}/{len(open_query_probes)}] skip_resume query={query_id}", flush=True)
        else:
            update_progress(
                progress_path=progress_path,
                completed_records=list(completed.values()),
                total=base_total,
                current_sample_id=query_id,
                current_status="running",
            )
            print(
                f"[OPEN {index}/{len(open_query_probes)}] start query={query_id} video={query['video_id']} "
                f"task={query['task_family']}",
                flush=True,
            )
            started_at = time.time()
            write_current_sample(
                current_sample_path=current_sample_path,
                payload={
                    "kind": "open_query",
                    "status": "running",
                    "index": index,
                    "total": len(open_query_probes),
                    "vqa_id": query_id,
                    "video_id": str(query["video_id"]),
                    "task_family": str(query["task_family"]),
                    "started_at": started_at,
                },
            )
            try:
                payload = execute_open_query_with_timeout(
                    query=query,
                    max_steps=open_query_max_steps,
                    timeout_seconds=open_query_timeout_seconds,
                    heartbeat_path=heartbeat_path,
                )
            except SampleTimeoutError as exc:
                payload = build_open_query_failure_payload(
                    query=query,
                    failure_type="agent_timeout",
                    failure_message=str(exc),
                    heartbeat_path=heartbeat_path,
                )
            except Exception as exc:  # noqa: BLE001
                payload = build_open_query_failure_payload(
                    query=query,
                    failure_type=f"agent_error:{type(exc).__name__}",
                    failure_message=str(exc),
                    heartbeat_path=heartbeat_path,
                )
            append_jsonl(predictions_path, payload)
            completed[query_id] = payload
            append_jsonl(session_trace_path, trace_row_from_payload(payload))
            update_session_state_snapshot(
                session_state_path=session_state_path,
                project_paths=ProjectPaths.from_env(),
                video_id=str(query["video_id"]),
            )
            update_progress(
                progress_path=progress_path,
                completed_records=list(completed.values()),
                total=base_total,
                current_sample_id=query_id,
                current_status="completed",
            )
            write_current_sample(
                current_sample_path=current_sample_path,
                payload={
                    "kind": "open_query",
                    "status": "finished",
                    "index": index,
                    "total": len(open_query_probes),
                    "vqa_id": query_id,
                    "video_id": str(query["video_id"]),
                    "task_family": str(query["task_family"]),
                    "started_at": started_at,
                    "elapsed_seconds": round(time.time() - started_at, 2),
                    "failure_type": payload.get("failure_type"),
                    "tool_calls": payload.get("tool_calls") or [],
                },
            )
            persist_partial_artifacts(
                out_dir=out_dir,
                predictions_path=predictions_path,
                session_trace_path=session_trace_path,
                session_state_path=session_state_path,
                summary_path=summary_path,
                audit_path=audit_path,
                records=list(completed.values()),
                task_families=task_families,
                limit_per_task=limit_per_task,
                video_id=video_id,
            )
            print(
                f"[OPEN {index}/{len(open_query_probes)}] query={query_id} failure={payload.get('failure_type')} "
                f"elapsed={round(time.time() - started_at, 2)}s",
                flush=True,
            )
        records.append(payload)

    if not session_trace_path.exists():
        write_jsonl(session_trace_path, session_trace_rows)
    if session_state_payload:
        session_state_path.write_text(json.dumps(session_state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif session_state_path.exists():
        session_state_path.unlink()

    summary, audit = persist_partial_artifacts(
        out_dir=out_dir,
        predictions_path=predictions_path,
        session_trace_path=session_trace_path,
        session_state_path=session_state_path,
        summary_path=summary_path,
        audit_path=audit_path,
        records=records,
        task_families=task_families,
        limit_per_task=limit_per_task,
        video_id=video_id,
    )
    update_progress(
        progress_path=progress_path,
        completed_records=records,
        total=len(selected_rows) + len(open_query_probes),
        current_sample_id=None,
        current_status="finished",
    )
    return {
        "selection_summary": summary,
        "audit": audit,
        "paths": {
            "predictions": predictions_path.as_posix(),
            "session_trace": session_trace_path.as_posix(),
            "session_state": session_state_path.as_posix() if session_state_path.exists() else None,
            "summary": summary_path.as_posix(),
            "audit": audit_path.as_posix(),
            "progress": progress_path.as_posix(),
            "current_sample": current_sample_path.as_posix(),
            "heartbeat": heartbeat_path.as_posix(),
        },
    }


class SampleTimeoutError(RuntimeError):
    pass


def execute_sample_with_timeout(
    *,
    row: dict[str, Any],
    max_steps: int,
    timeout_seconds: int,
    heartbeat_path: Path | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return _run_sample_once(row=row, max_steps=max_steps, heartbeat_path=heartbeat_path)
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_sample_worker, args=(row, max_steps, queue, str(heartbeat_path) if heartbeat_path else ""))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise SampleTimeoutError(f"sample exceeded timeout_seconds={timeout_seconds}")
    if process.exitcode not in {0, None} and queue.empty():
        raise RuntimeError(f"sample worker exited with code={process.exitcode}")
    if queue.empty():
        raise RuntimeError("sample worker finished without payload")
    payload = queue.get()
    if not isinstance(payload, dict):
        raise RuntimeError(f"sample worker returned invalid payload type={type(payload).__name__}")
    if payload.get("_worker_error"):
        raise RuntimeError(str(payload.get("failure_message") or payload["_worker_error"]))
    return payload


def execute_open_query_with_timeout(
    *,
    query: dict[str, Any],
    max_steps: int,
    timeout_seconds: int,
    heartbeat_path: Path | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return _run_open_query_once(query=query, max_steps=max_steps, heartbeat_path=heartbeat_path)
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_open_query_worker, args=(query, max_steps, queue, str(heartbeat_path) if heartbeat_path else ""))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise SampleTimeoutError(f"open_query exceeded timeout_seconds={timeout_seconds}")
    if process.exitcode not in {0, None} and queue.empty():
        raise RuntimeError(f"open_query worker exited with code={process.exitcode}")
    if queue.empty():
        raise RuntimeError("open_query worker finished without payload")
    payload = queue.get()
    if not isinstance(payload, dict):
        raise RuntimeError(f"open_query worker returned invalid payload type={type(payload).__name__}")
    if payload.get("_worker_error"):
        raise RuntimeError(str(payload.get("failure_message") or payload["_worker_error"]))
    return payload


def _sample_worker(row: dict[str, Any], max_steps: int, queue: mp.Queue, heartbeat_path: str) -> None:
    try:
        queue.put(_run_sample_once(row=row, max_steps=max_steps, heartbeat_path=Path(heartbeat_path) if heartbeat_path else None))
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "_worker_error": f"{type(exc).__name__}",
                "failure_message": str(exc),
            }
        )


def _open_query_worker(query: dict[str, Any], max_steps: int, queue: mp.Queue, heartbeat_path: str) -> None:
    try:
        queue.put(_run_open_query_once(query=query, max_steps=max_steps, heartbeat_path=Path(heartbeat_path) if heartbeat_path else None))
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "_worker_error": f"{type(exc).__name__}",
                "failure_message": str(exc),
            }
        )


def _run_sample_once(*, row: dict[str, Any], max_steps: int, heartbeat_path: Path | None = None) -> dict[str, Any]:
    if heartbeat_path:
        os.environ["FOOD_AGENT_HEARTBEAT_PATH"] = heartbeat_path.as_posix()
    agent = GraphAgent(paths=ProjectPaths.from_env())
    result = agent.answer_vqa_row(row, max_steps=max_steps)
    return result.to_dict(gold=int(row["correct_idx"]), include_row=row)


def _run_open_query_once(*, query: dict[str, Any], max_steps: int, heartbeat_path: Path | None = None) -> dict[str, Any]:
    if heartbeat_path:
        os.environ["FOOD_AGENT_HEARTBEAT_PATH"] = heartbeat_path.as_posix()
    agent = GraphAgent(paths=ProjectPaths.from_env())
    result = agent.answer_open_query(
        video_id=str(query["video_id"]),
        question=str(query["question"]),
        inputs_json=str(query.get("inputs_json") or "{}"),
        task_family=str(query["task_family"]),
        max_steps=max_steps,
        query_id=str(query["query_id"]),
    )
    return result.to_dict(
        include_row={
            "question": query["question"],
            "choices_json": json.dumps(["OPEN_ENDED_RESPONSE"], ensure_ascii=False),
            "inputs_json": query.get("inputs_json") or "{}",
        }
    )


@contextmanager
def sample_timeout(timeout_seconds: int):
    if timeout_seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise SampleTimeoutError(f"sample exceeded timeout_seconds={timeout_seconds}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def select_rows(*, df: pd.DataFrame, video_id: str | None, task_families: list[str], limit_per_task: int) -> list[dict[str, Any]]:
    subset = df.copy()
    if video_id:
        subset = subset[subset["primary_video_id"] == video_id].copy()
    rows: list[dict[str, Any]] = []
    for family in task_families:
        family_rows = subset[subset["task_family"] == family].sort_values("vqa_id").head(limit_per_task)
        rows.extend(family_rows.to_dict("records"))
    return rows


def build_summary(*, records: list[dict[str, Any]], task_families: list[str], limit_per_task: int, video_id: str | None) -> dict[str, Any]:
    vqa_records = [item for item in records if item.get("gold") is not None]
    correct = sum(1 for item in vqa_records if item.get("correct") is True)
    failure_counts: dict[str, int] = {}
    for item in records:
        key = str(item.get("failure_type") or "none")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    completed_count = len(records)
    metric_rows = [extract_record_metrics(item) for item in records]
    latency_values = [row["elapsed_seconds"] for row in metric_rows if row["elapsed_seconds"] is not None]
    prompt_tokens = [row["prompt_tokens"] for row in metric_rows]
    completion_tokens = [row["completion_tokens"] for row in metric_rows]
    total_tokens = [row["total_tokens"] for row in metric_rows]
    estimated_costs = [row["estimated_cost"] for row in metric_rows]
    tool_failure_records = [row for row in metric_rows if row["tool_failure_count"] > 0]
    ineffective_tool_records = [row for row in metric_rows if row["ineffective_tool_count"] > 0]
    cached_artifact_hits = sum(1 for row in metric_rows if row["cached_artifact_hit"])
    memory_reuse_hits = sum(1 for row in metric_rows if row["reuse_memory_count"] > 0)
    relation_reuse_hits = sum(1 for row in metric_rows if row["relation_reuse_count"] > 0)
    raw_revisit_hits = sum(1 for row in metric_rows if row["raw_revisit_count"] > 0)
    planner_override_total = sum(row["planner_override_count"] for row in metric_rows)
    verifier_block_total = sum(row["verifier_blocked_finish_count"] for row in metric_rows)
    failed_tool_recovery_total = sum(row["failed_tool_recovery_count"] for row in metric_rows)
    ineffective_tool_avoidance_total = sum(row["ineffective_tool_avoidance_count"] for row in metric_rows)
    reuse_summary = build_reuse_benefit_summary_from_records(records)
    return {
        "count": len(records),
        "correct": correct,
        "accuracy": (correct / len(vqa_records)) if vqa_records else None,
        "vqa_count": len(vqa_records),
        "open_query_count": len(records) - len(vqa_records),
        "task_families": task_families,
        "limit_per_task": limit_per_task,
        "video_id": video_id,
        "failure_counts": failure_counts,
        "memory_reuse_rate": (memory_reuse_hits / completed_count) if completed_count else 0.0,
        "relation_reuse_rate": (relation_reuse_hits / completed_count) if completed_count else 0.0,
        "raw_revisit_rate": (raw_revisit_hits / completed_count) if completed_count else 0.0,
        "planner_override_count": planner_override_total,
        "verifier_blocked_finish_count": verifier_block_total,
        "tool_failure_recovery_rate": (
            sum(1 for row in tool_failure_records if row["failed_tool_recovery_count"] > 0) / len(tool_failure_records)
        )
        if tool_failure_records
        else 0.0,
        "ineffective_tool_avoidance_rate": (
            sum(1 for row in ineffective_tool_records if row["ineffective_tool_avoidance_count"] > 0)
            / len(ineffective_tool_records)
        )
        if ineffective_tool_records
        else 0.0,
        "cached_artifact_reuse_hits": cached_artifact_hits,
        "cached_artifact_reuse_gain": reuse_summary["cached_artifact_reuse_gain"],
        "same_video_follow_up_cost_reduction": reuse_summary["same_video_follow_up_cost_reduction"],
        "avg_tool_calls": (sum(row["tool_call_count"] for row in metric_rows) / completed_count) if completed_count else 0.0,
        "avg_reasoning_steps": (sum(row["reasoning_step_count"] for row in metric_rows) / completed_count) if completed_count else 0.0,
        "avg_elapsed_seconds": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "avg_prompt_tokens": (sum(prompt_tokens) / completed_count) if completed_count else 0.0,
        "avg_completion_tokens": (sum(completion_tokens) / completed_count) if completed_count else 0.0,
        "avg_total_tokens": (sum(total_tokens) / completed_count) if completed_count else 0.0,
        "avg_estimated_cost": (sum(estimated_costs) / completed_count) if completed_count else 0.0,
        "video_session_summary": reuse_summary["video_session_summary"],
        "attempt_summary": {
            "max_attempt_count": 1,
            "avg_attempt_count": 1.0 if completed_count else 0.0,
            "retried_sample_count": 0,
        },
    }


def extract_record_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    tool_calls = payload.get("tool_calls") or [
        entry.get("tool") for entry in payload.get("tool_trace", []) if isinstance(entry, dict) and entry.get("tool")
    ]
    working_memory = payload.get("working_memory") or []
    usage = payload.get("usage") or {}
    return {
        "tool_call_count": len(tool_calls),
        "reasoning_step_count": len(payload.get("tool_trace") or []),
        "elapsed_seconds": float(payload["elapsed_seconds"]) if payload.get("elapsed_seconds") is not None else None,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "total_tokens": float(usage.get("total_tokens") or 0.0),
        "estimated_cost": float(usage.get("estimated_cost") or 0.0),
        "reuse_memory_count": sum(
            1 for item in working_memory if isinstance(item, str) and item.startswith("reuse:")
        ),
        "relation_reuse_count": sum(
            1 for item in working_memory if isinstance(item, str) and item.startswith("reuse_relation:")
        ),
        "planner_override_count": sum(
            1 for item in working_memory if isinstance(item, str) and item.startswith("planner_override ")
        ),
        "verifier_blocked_finish_count": sum(
            1
            for item in working_memory
            if isinstance(item, str) and item.startswith("planner_override verifier_blocked_finish=")
        ),
        "tool_failure_count": len(payload.get("tool_failures") or []),
        "ineffective_tool_count": len(payload.get("ineffective_tools") or []),
        "failed_tool_recovery_count": count_failed_tool_recoveries(payload.get("tool_trace") or []),
        "ineffective_tool_avoidance_count": count_ineffective_tool_avoidances(payload.get("tool_trace") or []),
        "raw_revisit_count": count_tools(tool_calls, RAW_REVISIT_TOOLS),
        "structured_query_count": count_tools(tool_calls, STRUCTURED_QUERY_TOOLS),
        "cached_artifact_hit": any(tool == "retrieve_cached_artifacts" for tool in tool_calls),
        "tool_calls": tool_calls,
    }


def build_reuse_benefit_summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for payload in records:
        video_key = str(payload.get("video_id") or "")
        by_video.setdefault(video_key, []).append(payload)

    comparable_video_count = 0
    first_cost_total = 0.0
    follow_cost_total = 0.0
    first_token_total = 0.0
    follow_token_total = 0.0
    first_tool_total = 0.0
    follow_tool_total = 0.0
    first_raw_total = 0.0
    follow_raw_total = 0.0
    video_session_rows: list[dict[str, Any]] = []

    for video_id, payloads in sorted(by_video.items()):
        annotated: list[dict[str, Any]] = []
        for position, payload in enumerate(payloads, start=1):
            metrics = extract_record_metrics(payload)
            annotated.append({"payload": payload, "metrics": metrics, "position": position})
        count = len(annotated)
        if count:
            video_session_rows.append(
                {
                    "video_id": video_id,
                    "count": count,
                    "avg_tool_calls": sum(item["metrics"]["tool_call_count"] for item in annotated) / count,
                    "avg_reasoning_steps": sum(item["metrics"]["reasoning_step_count"] for item in annotated) / count,
                    "avg_estimated_cost": sum(item["metrics"]["estimated_cost"] for item in annotated) / count,
                    "avg_total_tokens": sum(item["metrics"]["total_tokens"] for item in annotated) / count,
                    "avg_raw_revisit_count": sum(item["metrics"]["raw_revisit_count"] for item in annotated) / count,
                }
            )
        first = [item for item in annotated if item["position"] == 1]
        follow = [item for item in annotated if item["position"] >= 2]
        if not first or not follow:
            continue
        comparable_video_count += 1
        first_item = first[0]["metrics"]
        first_cost_total += first_item["estimated_cost"]
        first_token_total += first_item["total_tokens"]
        first_tool_total += first_item["tool_call_count"]
        first_raw_total += first_item["raw_revisit_count"]
        follow_cost_total += sum(item["metrics"]["estimated_cost"] for item in follow) / len(follow)
        follow_token_total += sum(item["metrics"]["total_tokens"] for item in follow) / len(follow)
        follow_tool_total += sum(item["metrics"]["tool_call_count"] for item in follow) / len(follow)
        follow_raw_total += sum(item["metrics"]["raw_revisit_count"] for item in follow) / len(follow)

    first_cost_avg = (first_cost_total / comparable_video_count) if comparable_video_count else 0.0
    follow_cost_avg = (follow_cost_total / comparable_video_count) if comparable_video_count else 0.0
    first_token_avg = (first_token_total / comparable_video_count) if comparable_video_count else 0.0
    follow_token_avg = (follow_token_total / comparable_video_count) if comparable_video_count else 0.0
    first_tool_avg = (first_tool_total / comparable_video_count) if comparable_video_count else 0.0
    follow_tool_avg = (follow_tool_total / comparable_video_count) if comparable_video_count else 0.0
    first_raw_avg = (first_raw_total / comparable_video_count) if comparable_video_count else 0.0
    follow_raw_avg = (follow_raw_total / comparable_video_count) if comparable_video_count else 0.0

    return {
        "video_session_summary": {
            "video_count": len(video_session_rows),
            "videos": video_session_rows,
        },
        "cached_artifact_reuse_gain": {
            "cached_artifact_hit_count": sum(
                1
                for payload in records
                if any(tool == "retrieve_cached_artifacts" for tool in (extract_record_metrics(payload)["tool_calls"]))
            ),
            "comparable_video_count": comparable_video_count,
            "first_question_avg_raw_revisit_count": first_raw_avg,
            "follow_up_avg_raw_revisit_count": follow_raw_avg,
            "follow_up_minus_first_raw_revisit_count": follow_raw_avg - first_raw_avg,
        },
        "same_video_follow_up_cost_reduction": {
            "comparable_video_count": comparable_video_count,
            "first_question_avg_estimated_cost": first_cost_avg,
            "follow_up_avg_estimated_cost": follow_cost_avg,
            "follow_up_minus_first_estimated_cost": follow_cost_avg - first_cost_avg,
            "first_question_avg_total_tokens": first_token_avg,
            "follow_up_avg_total_tokens": follow_token_avg,
            "follow_up_minus_first_total_tokens": follow_token_avg - first_token_avg,
            "first_question_avg_tool_calls": first_tool_avg,
            "follow_up_avg_tool_calls": follow_tool_avg,
            "follow_up_minus_first_tool_calls": follow_tool_avg - first_tool_avg,
        },
    }


def persist_partial_artifacts(
    *,
    out_dir: Path,
    predictions_path: Path,
    session_trace_path: Path,
    session_state_path: Path,
    summary_path: Path,
    audit_path: Path,
    records: list[dict[str, Any]],
    task_families: list[str],
    limit_per_task: int,
    video_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = build_summary(
        records=records,
        task_families=task_families,
        limit_per_task=limit_per_task,
        video_id=video_id,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = build_audit_report(
        run_dir=out_dir,
        predictions_file=predictions_path if predictions_path.exists() else None,
        result_files=[],
        session_trace=session_trace_path if session_trace_path.exists() else None,
        session_state=session_state_path if session_state_path.exists() else None,
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, audit


def trace_row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "vqa_id": payload.get("vqa_id"),
        "tool_calls": payload.get("tool_calls") or [entry.get("tool") for entry in payload.get("tool_trace", []) if isinstance(entry, dict)],
        "usage": payload.get("usage") or {},
        "tool_failures": payload.get("tool_failures") or [],
        "ineffective_tools": payload.get("ineffective_tools") or [],
        "latest_verification": payload.get("latest_verification") or (payload.get("verification_history") or [{}])[-1],
        "action_intent_trace_tail": (payload.get("action_intent_trace") or [])[-5:],
        "open_questions_tail": (payload.get("open_questions") or [])[-8:],
        "working_memory_tail": (payload.get("working_memory") or [])[-12:],
        "evidence_tail": (payload.get("evidence_bundle") or [])[-12:],
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("vqa_id"):
            existing[str(payload["vqa_id"])] = payload
    return existing


def build_failure_payload(
    *,
    row: dict[str, Any],
    failure_type: str,
    failure_message: str,
    heartbeat_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "vqa_id": str(row.get("vqa_id") or ""),
        "video_id": row.get("primary_video_id"),
        "task_family": row.get("task_family"),
        "prediction": None,
        "gold": int(row["correct_idx"]),
        "correct": False,
        "answer_text": "",
        "confidence": 0.0,
        "elapsed_seconds": None,
        "usage": {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "estimated_cost": 0.0},
        "tool_trace": [],
        "evidence_bundle": [],
        "working_memory": [],
        "retrieved_frames": [],
        "verification_history": [],
        "tool_failures": [],
        "ineffective_tools": [],
        "open_questions": [],
        "raw_model_output": "",
        "question": row.get("question"),
        "choices_json": row.get("choices_json"),
        "inputs_json": row.get("inputs_json"),
        "failure_type": failure_type,
        "failure_message": failure_message,
        "tool_calls": [],
        "heartbeat_tail": load_heartbeat_tail(heartbeat_path),
    }


def build_open_query_failure_payload(
    *,
    query: dict[str, Any],
    failure_type: str,
    failure_message: str,
    heartbeat_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "vqa_id": str(query["query_id"]),
        "video_id": query["video_id"],
        "task_family": query["task_family"],
        "prediction": None,
        "gold": None,
        "correct": None,
        "answer_text": "",
        "confidence": 0.0,
        "elapsed_seconds": None,
        "usage": {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "estimated_cost": 0.0},
        "tool_trace": [],
        "evidence_bundle": [],
        "working_memory": [],
        "retrieved_frames": [],
        "verification_history": [],
        "tool_failures": [],
        "ineffective_tools": [],
        "open_questions": [],
        "raw_model_output": "",
        "question": query["question"],
        "choices_json": json.dumps(["OPEN_ENDED_RESPONSE"], ensure_ascii=False),
        "inputs_json": query.get("inputs_json") or "{}",
        "failure_type": failure_type,
        "failure_message": failure_message,
        "tool_calls": [],
        "heartbeat_tail": load_heartbeat_tail(heartbeat_path),
    }


def update_session_state_snapshot(*, session_state_path: Path, project_paths: ProjectPaths, video_id: str) -> None:
    if not video_id:
        return
    maybe_state = project_paths.graph_agent_sessions_root / video_id / "session_state.json"
    if not maybe_state.exists():
        return
    try:
        payload = json.loads(maybe_state.read_text(encoding="utf-8"))
    except Exception:
        return
    session_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_progress(
    *,
    progress_path: Path,
    completed_records: list[dict[str, Any]],
    total: int,
    current_sample_id: str | None,
    current_status: str,
) -> None:
    vqa_records = [item for item in completed_records if item.get("gold") is not None]
    correct = sum(1 for item in vqa_records if item.get("correct") is True)
    payload = {
        "total": total,
        "completed": len(completed_records),
        "remaining": max(total - len(completed_records), 0),
        "current_sample_id": current_sample_id,
        "current_status": current_status,
        "correct": correct,
        "accuracy": (correct / len(vqa_records)) if vqa_records else None,
        "failure_counts": build_summary(
            records=completed_records,
            task_families=[],
            limit_per_task=0,
            video_id=None,
        )["failure_counts"],
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_current_sample(*, current_sample_path: Path, payload: dict[str, Any]) -> None:
    current_sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_heartbeat_tail(path: Path | None, *, limit: int = 20) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
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


def build_open_query_probes(*, video_id: str, anchor_row: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not video_id:
        return []
    anchor_time = infer_anchor_time(anchor_row)
    probes: list[dict[str, Any]] = []
    for index, item in enumerate(DEFAULT_OPEN_QUERY_PROBES, start=1):
        inputs_json = item["inputs_json"]
        if item["task_family"] == "open_query_location" and anchor_time:
            inputs_json = json.dumps({"anchor_time": anchor_time}, ensure_ascii=False)
        probes.append(
            {
                "query_id": f"open_query_probe:{video_id}:{index}",
                "video_id": video_id,
                "task_family": item["task_family"],
                "question": item["question"],
                "inputs_json": inputs_json,
            }
        )
    return probes


def infer_anchor_time(row: dict[str, Any] | None) -> str | None:
    if not isinstance(row, dict):
        return None
    question = str(row.get("question") or "")
    match = None
    import re

    match = re.search(r"<TIME\s+([0-9:.]+)\s+video\s+1>", question)
    if match:
        return match.group(1)
    inputs_json = row.get("inputs_json")
    if isinstance(inputs_json, str) and inputs_json.strip():
        try:
            payload = json.loads(inputs_json)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, dict):
                    for key in ("time", "start_time"):
                        if value.get(key):
                            return str(value[key])
    return None


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
