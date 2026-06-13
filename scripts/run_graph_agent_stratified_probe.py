#!/usr/bin/env python3
"""Run a random stratified probe over many VQA task families with resume support."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths
from food_agent.vqa import VQAPrediction, compute_metrics
from scripts.run_graph_agent_batch import (
    RAW_REVISIT_TOOLS,
    STRUCTURED_QUERY_TOOLS,
    append_prediction_jsonl,
    build_failure_cases,
    build_progress_payload,
    build_video_positions,
    build_video_session_summary,
    count_failed_tool_recoveries,
    count_ineffective_tool_avoidances,
    count_planner_overrides_from_result,
    count_relation_reuse_from_result,
    count_reuse_memory_from_result,
    count_tools,
    count_verifier_blocked_finish_from_result,
    load_predictions_by_id,
    summarize_failures,
    write_jsonl_records,
    write_status,
)


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "graph_agent_stratified_probe")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--task-family-count", type=int, default=20)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries-per-sample", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)
    run_name = build_run_name(
        task_family_count=args.task_family_count,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        suffix=args.run_suffix,
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions_graph_agent.jsonl"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics_graph_agent.json"
    failures_path = run_dir / "failure_summary.json"
    failure_cases_path = run_dir / "failure_cases.jsonl"
    progress_path = run_dir / "progress.json"
    selection_path = run_dir / "selection.json"
    status_path = run_dir / "run_status.json"
    retry_state_path = run_dir / "retry_state.json"

    rows = load_or_select_rows(
        index_file=args.index_file,
        selection_path=selection_path,
        resume=args.resume,
        seed=args.seed,
        task_family_count=args.task_family_count,
        samples_per_task=args.samples_per_task,
    )
    if not rows:
        raise RuntimeError("no rows selected for stratified probe")

    completed = load_predictions_by_id(pred_path) if args.resume else {}
    retry_state = load_retry_state(retry_state_path) if args.resume else {}
    total = len(rows)
    write_status(
        status_path,
        status="running",
        run_name=run_name,
        total=total,
        completed=len(completed),
        pid=0,
        task_family=None,
        task_family_group="stratified_probe",
    )
    if not selection_path.exists():
        selection_path.write_text(
            json.dumps({"rows": [normalize_row(row) for row in rows]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    video_positions = build_video_positions([SimpleSample.from_row(row) for row in rows])
    running: dict[str, VQAPrediction] = dict(completed)

    for index, row in enumerate(rows, start=1):
        sample_id = str(row["vqa_id"])
        if should_skip_completed_sample(
            sample_id=sample_id,
            completed=completed,
            retry_state=retry_state,
            max_retries_per_sample=args.max_retries_per_sample,
        ):
            pred = completed[sample_id]
            print(
                f"[{index}/{total}] skip sample={sample_id} task={row['task_family']} pred={pred.prediction} "
                f"gold={pred.gold} correct={pred.correct} failure={pred.failure_type} attempts={pred.attempt_count}",
                flush=True,
            )
            continue
        pred = run_probe_sample(
            agent=agent,
            row=row,
            sample_id=sample_id,
            session_video_position=video_positions.get(sample_id, 0),
            max_steps=args.max_steps,
            retry_state=retry_state,
            max_retries_per_sample=args.max_retries_per_sample,
        )
        retry_state[sample_id] = {
            "attempt_count": pred.attempt_count,
            "failure_type": pred.failure_type,
            "updated_at": time.time(),
        }
        retry_state_path.write_text(json.dumps(retry_state, ensure_ascii=False, indent=2), encoding="utf-8")
        append_prediction_jsonl(pred_path, pred)
        running[pred.sample_id] = pred
        summary = update_probe_artifacts(
            rows=rows,
            running=running,
            run_name=run_name,
            summary_path=summary_path,
            metrics_path=metrics_path,
            failures_path=failures_path,
            failure_cases_path=failure_cases_path,
            progress_path=progress_path,
        )
        write_status(
            status_path,
            status="running",
            run_name=run_name,
            total=total,
            completed=len(running),
            pid=0,
            task_family=None,
            task_family_group="stratified_probe",
            last_sample=pred.sample_id,
        )
        metrics = compute_metrics(list(running.values()))
        print(
            f"[{index}/{total}] sample={pred.sample_id} task={pred.task_family} pred={pred.prediction} gold={pred.gold} "
            f"correct={pred.correct} raw={pred.raw_revisit_count} structured={pred.structured_query_count} "
            f"attempts={pred.attempt_count} failure={pred.failure_type} "
            f"running_acc={metrics.get('correct', 0)}/{len(running)}={format_ratio(metrics.get('accuracy'))}",
            flush=True,
        )

    final_summary = update_probe_artifacts(
        rows=rows,
        running=running,
        run_name=run_name,
        summary_path=summary_path,
        metrics_path=metrics_path,
        failures_path=failures_path,
        failure_cases_path=failure_cases_path,
        progress_path=progress_path,
    )
    write_status(
        status_path,
        status="completed",
        run_name=run_name,
        total=total,
        completed=len(running),
        pid=0,
        task_family=None,
        task_family_group="stratified_probe",
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


def select_stratified_rows(
    *,
    df: pd.DataFrame,
    seed: int,
    task_family_count: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = sorted(str(item) for item in df["task_family"].dropna().unique().tolist())
    if not families:
        return []
    chosen_families = families if task_family_count >= len(families) else rng.sample(families, task_family_count)
    rows: list[dict[str, Any]] = []
    for family in sorted(chosen_families):
        subset = df[df["task_family"] == family].copy()
        if subset.empty:
            continue
        candidate_rows = subset.to_dict("records")
        rng.shuffle(candidate_rows)
        rows.extend(candidate_rows[:samples_per_task])
    rows.sort(key=lambda item: (str(item.get("primary_video_id") or ""), str(item.get("task_family") or ""), str(item.get("vqa_id") or "")))
    return rows


def load_or_select_rows(
    *,
    index_file: Path,
    selection_path: Path,
    resume: bool,
    seed: int,
    task_family_count: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    if resume and selection_path.exists():
        try:
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if isinstance(rows, list) and rows:
            return [normalize_row(dict(row)) for row in rows if isinstance(row, dict)]
    df = pd.read_parquet(index_file)
    return select_stratified_rows(
        df=df,
        seed=seed,
        task_family_count=task_family_count,
        samples_per_task=samples_per_task,
    )


def run_probe_sample(
    *,
    agent: GraphAgent,
    row: dict[str, Any],
    sample_id: str,
    session_video_position: int,
    max_steps: int,
    retry_state: dict[str, dict[str, Any]],
    max_retries_per_sample: int,
) -> VQAPrediction:
    prior_attempts = int((retry_state.get(sample_id) or {}).get("attempt_count") or 0)
    attempts = max(prior_attempts, 0)
    last_error: Exception | None = None
    while attempts < max_retries_per_sample:
        attempts += 1
        try:
            result = agent.answer_vqa_row(row, max_steps=max_steps)
            tool_calls = [entry.get("tool", "") for entry in result.tool_trace]
            return VQAPrediction(
                sample_id=sample_id,
                baseline="graph-agent-stratified-probe",
                task_family=str(row["task_family"]),
                video_id=str(row["primary_video_id"]),
                question=str(row["question"]),
                choices=json.loads(row["choices_json"]),
                gold=int(row["correct_idx"]),
                prediction=int(result.prediction if result.prediction is not None else 0),
                correct=result.prediction == int(row["correct_idx"]),
                evidence_ids=[],
                tool_calls=tool_calls,
                failure_type=None if result.prediction == int(row["correct_idx"]) else "reasoning_error",
                attempt_count=attempts,
                reuse_memory_count=count_reuse_memory_from_result(result),
                raw_revisit_count=count_tools(tool_calls, RAW_REVISIT_TOOLS),
                structured_query_count=count_tools(tool_calls, STRUCTURED_QUERY_TOOLS),
                session_video_position=session_video_position,
                relation_reuse_count=count_relation_reuse_from_result(result),
                planner_override_count=count_planner_overrides_from_result(result),
                verifier_blocked_finish_count=count_verifier_blocked_finish_from_result(result),
                tool_failure_count=len(result.tool_failures),
                ineffective_tool_count=len(result.ineffective_tools),
                failed_tool_recovery_count=count_failed_tool_recoveries(result.tool_trace),
                ineffective_tool_avoidance_count=count_ineffective_tool_avoidances(result.tool_trace),
                tool_call_count=len(tool_calls),
                reasoning_step_count=len(result.tool_trace),
                elapsed_seconds=float(result.elapsed_seconds),
                prompt_tokens=float((result.usage or {}).get("prompt_tokens") or 0.0),
                completion_tokens=float((result.usage or {}).get("completion_tokens") or 0.0),
                total_tokens=float((result.usage or {}).get("total_tokens") or 0.0),
                estimated_cost=float((result.usage or {}).get("estimated_cost") or 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    error_type = type(last_error).__name__ if last_error is not None else "RuntimeError"
    return VQAPrediction(
        sample_id=sample_id,
        baseline="graph-agent-stratified-probe",
        task_family=str(row["task_family"]),
        video_id=str(row["primary_video_id"]),
        question=str(row["question"]),
        choices=json.loads(row["choices_json"]),
        gold=int(row["correct_idx"]),
        prediction=0,
        correct=False,
        evidence_ids=[],
        tool_calls=[],
        failure_type=f"agent_error:{error_type}",
        attempt_count=max(attempts, 1),
        reuse_memory_count=0,
        raw_revisit_count=0,
        structured_query_count=0,
        session_video_position=session_video_position,
        relation_reuse_count=0,
        planner_override_count=0,
        verifier_blocked_finish_count=0,
        tool_failure_count=0,
        ineffective_tool_count=0,
        failed_tool_recovery_count=0,
        ineffective_tool_avoidance_count=0,
        tool_call_count=0,
        reasoning_step_count=0,
        elapsed_seconds=None,
        prompt_tokens=0.0,
        completion_tokens=0.0,
        total_tokens=0.0,
        estimated_cost=0.0,
    )


def update_probe_artifacts(
    *,
    rows: list[dict[str, Any]],
    running: dict[str, VQAPrediction],
    run_name: str,
    summary_path: Path,
    metrics_path: Path,
    failures_path: Path,
    failure_cases_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    ordered = [running[str(row["vqa_id"])] for row in rows if str(row["vqa_id"]) in running]
    metrics = compute_metrics(ordered)
    failure_summary = summarize_failures(ordered)
    progress = build_progress_payload(total=len(rows), predictions=ordered)
    failure_cases = build_failure_cases(ordered)
    video_session_summary = build_video_session_summary(ordered)
    task_cover = {}
    for row in rows:
        family = str(row["task_family"])
        task_cover[family] = task_cover.get(family, 0) + 1
    summary = {
        "run_name": run_name,
        "sample_count": len(rows),
        "completed_count": len(ordered),
        "task_family_count": len(task_cover),
        "task_family_cover": task_cover,
        "metrics": metrics,
        "failure_summary": failure_summary,
        "progress": progress,
        "video_session_summary": video_session_summary,
        "attempt_summary": build_attempt_summary(ordered),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path.write_text(json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl_records(failure_cases_path, failure_cases)
    return summary


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            normalized[key] = int(value)
        elif isinstance(value, (np.floating,)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def load_retry_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def should_skip_completed_sample(
    *,
    sample_id: str,
    completed: dict[str, VQAPrediction],
    retry_state: dict[str, dict[str, Any]],
    max_retries_per_sample: int,
) -> bool:
    pred = completed.get(sample_id)
    if pred is None:
        return False
    if pred.correct:
        return True
    attempts = int((retry_state.get(sample_id) or {}).get("attempt_count") or pred.attempt_count or 0)
    return attempts >= max_retries_per_sample


def build_attempt_summary(predictions: list[VQAPrediction]) -> dict[str, Any]:
    if not predictions:
        return {"max_attempt_count": 0, "avg_attempt_count": 0.0, "retried_sample_count": 0}
    attempt_counts = [int(pred.attempt_count or 0) for pred in predictions]
    retried = sum(1 for count in attempt_counts if count > 1)
    return {
        "max_attempt_count": max(attempt_counts),
        "avg_attempt_count": sum(attempt_counts) / len(attempt_counts),
        "retried_sample_count": retried,
    }


def build_run_name(*, task_family_count: int, samples_per_task: int, seed: int, suffix: str | None) -> str:
    name = f"stratified_f{task_family_count}_k{samples_per_task}_seed{seed}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def format_ratio(value: float | None) -> str:
    if value is None:
        return "0.000"
    return f"{value:.3f}"


class SimpleSample:
    def __init__(self, *, vqa_id: str, primary_video_id: str):
        self.vqa_id = vqa_id
        self.primary_video_id = primary_video_id

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SimpleSample":
        return cls(vqa_id=str(row["vqa_id"]), primary_video_id=str(row["primary_video_id"]))


if __name__ == "__main__":
    raise SystemExit(main())
