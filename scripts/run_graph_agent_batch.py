#!/usr/bin/env python3
"""Run the graph agent over a VQA subset with resume and summary outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths
from food_agent.vqa import VQAPrediction, compute_metrics, load_vqa_samples


TASK_FAMILY_GROUPS = {
    "food-core": [
        "ingredient_ingredient_retrieval",
        "ingredient_exact_ingredient_recognition",
        "ingredient_ingredient_recognition",
        "recipe_step_recognition",
        "recipe_recipe_recognition",
        "recipe_following_activity_recognition",
        "nutrition_nutrition_change",
    ],
    "multimodal-core": [
        "gaze_gaze_estimation",
        "gaze_interaction_anticipation",
        "3d_perception_fixture_location",
        "3d_perception_fixture_interaction_counting",
        "object_motion_object_movement_counting",
        "object_motion_stationary_object_localization",
    ],
}
STRUCTURED_QUERY_TOOLS = {
    "query_time",
    "query_object",
    "query_event",
    "query_state",
    "query_location",
    "query_region",
    "query_ocr",
    "get_neighbors",
    "query_ingredient_measurement",
    "compute_nutrition_change",
    "compare_choice_nutrition",
}
RAW_REVISIT_TOOLS = {
    "extract_frame_at_time",
    "extract_frames_for_range",
    "sample_sparse_frames",
    "sample_frames_around_peaks",
    "extract_input_reference_frames",
    "render_bbox_overlay",
    "extract_region_with_context",
    "run_ocr_on_image",
    "run_ocr_on_region",
    "detect_audio_peaks",
}


def count_relation_reuse_from_result(result) -> int:
    return sum(1 for item in result.working_memory if isinstance(item, str) and item.startswith("reuse_relation:"))


def count_planner_overrides_from_result(result) -> int:
    return sum(1 for item in result.working_memory if isinstance(item, str) and item.startswith("planner_override "))


def count_verifier_blocked_finish_from_result(result) -> int:
    return sum(
        1
        for item in result.working_memory
        if isinstance(item, str) and item.startswith("planner_override verifier_blocked_finish=")
    )


def count_failed_tool_recoveries(tool_trace: list[dict[str, Any]]) -> int:
    return count_recovery_events(tool_trace, failure_key="tool_failed")


def count_ineffective_tool_avoidances(tool_trace: list[dict[str, Any]]) -> int:
    return count_recovery_events(tool_trace, failure_key="tool_ineffective")


def count_recovery_events(tool_trace: list[dict[str, Any]], *, failure_key: str) -> int:
    recoveries = 0
    pending_failed_tool = ""
    for entry in tool_trace:
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


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "graph_agent_batch")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--task-family-group", choices=sorted(TASK_FAMILY_GROUPS), default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--no-video-session-mode", action="store_false", dest="video_session_mode")
    parser.set_defaults(video_session_mode=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)
    samples = load_selected_samples(
        args.index_dir,
        args.limit,
        args.task_family,
        args.task_family_group,
        random_seed=args.random_seed,
        video_session_mode=args.video_session_mode,
    )
    run_name = build_run_name(args.task_family, args.task_family_group, args.limit, args.run_suffix)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions_graph_agent.jsonl"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics_graph_agent.json"
    failures_path = run_dir / "failure_summary.json"
    failure_cases_path = run_dir / "failure_cases.jsonl"
    progress_path = run_dir / "progress.json"
    status_path = run_dir / "run_status.json"

    completed = load_predictions_by_id(pred_path) if args.resume else {}
    if completed:
        print(f"[resume] loaded_completed={len(completed)} from {pred_path}", flush=True)

    total = len(samples)
    running: dict[str, VQAPrediction] = dict(completed)
    write_status(
        status_path,
        status="running",
        run_name=run_name,
        total=total,
        completed=len(running),
        pid=os.getpid(),
        task_family=args.task_family,
        task_family_group=args.task_family_group,
    )
    if completed:
        update_run_artifacts(
            samples=samples,
            running=running,
            run_name=run_name,
            task_family=args.task_family,
            task_family_group=args.task_family_group,
            pred_path=pred_path,
            metrics_path=metrics_path,
            failures_path=failures_path,
            failure_cases_path=failure_cases_path,
            summary_path=summary_path,
            progress_path=progress_path,
        )
    video_positions: dict[str, int] = build_video_positions(samples)

    for index, sample in enumerate(samples, start=1):
        if sample.vqa_id in completed:
            pred = completed[sample.vqa_id]
            print(
                f"[{index}/{total}] skip sample={sample.vqa_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct} "
                f"reuse={pred.reuse_memory_count} raw={pred.raw_revisit_count} structured={pred.structured_query_count} "
                f"session_pos={pred.session_video_position} failure={pred.failure_type}",
                flush=True,
            )
            continue
        row = {
            "vqa_id": sample.vqa_id,
            "task_family": sample.task_family,
            "primary_video_id": sample.primary_video_id,
            "question": sample.question,
            "choices_json": json.dumps(sample.choices, ensure_ascii=False),
            "correct_idx": sample.correct_idx,
            "inputs_json": json.dumps(sample.inputs, ensure_ascii=False),
        }
        try:
            result = agent.answer_vqa_row(row, max_steps=args.max_steps)
            tool_calls = [entry.get("tool", "") for entry in result.tool_trace]
            pred = VQAPrediction(
                sample_id=sample.vqa_id,
                baseline="graph-agent",
                task_family=sample.task_family,
                video_id=sample.primary_video_id,
                question=sample.question,
                choices=sample.choices,
                gold=sample.correct_idx,
                prediction=int(result.prediction if result.prediction is not None else 0),
                correct=result.prediction == sample.correct_idx,
                evidence_ids=[],
                tool_calls=tool_calls,
                failure_type=None if result.prediction == sample.correct_idx else "reasoning_error",
                attempt_count=1,
                reuse_memory_count=count_reuse_memory_from_result(result),
                raw_revisit_count=count_tools(tool_calls, RAW_REVISIT_TOOLS),
                structured_query_count=count_tools(tool_calls, STRUCTURED_QUERY_TOOLS),
                session_video_position=video_positions.get(sample.vqa_id, 0),
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
            pred = VQAPrediction(
                sample_id=sample.vqa_id,
                baseline="graph-agent",
                task_family=sample.task_family,
                video_id=sample.primary_video_id,
                question=sample.question,
                choices=sample.choices,
                gold=sample.correct_idx,
                prediction=0,
                correct=False,
                evidence_ids=[],
                tool_calls=[],
                failure_type=f"agent_error:{type(exc).__name__}",
                attempt_count=1,
                reuse_memory_count=0,
                raw_revisit_count=0,
                structured_query_count=0,
                session_video_position=video_positions.get(sample.vqa_id, 0),
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
        append_prediction_jsonl(pred_path, pred)
        running[pred.sample_id] = pred
        update_run_artifacts(
            samples=samples,
            running=running,
            run_name=run_name,
            task_family=args.task_family,
            task_family_group=args.task_family_group,
            pred_path=pred_path,
            metrics_path=metrics_path,
            failures_path=failures_path,
            failure_cases_path=failure_cases_path,
            summary_path=summary_path,
            progress_path=progress_path,
        )
        write_status(
            status_path,
            status="running",
            run_name=run_name,
            total=total,
            completed=len(running),
            pid=os.getpid(),
            task_family=args.task_family,
            task_family_group=args.task_family_group,
            last_sample=pred.sample_id,
        )
        metrics = compute_metrics(list(running.values()))
        running_correct = metrics.get("correct", 0)
        running_accuracy = metrics.get("accuracy")
        print(
            f"[{index}/{total}] sample={pred.sample_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct} "
            f"reuse={pred.reuse_memory_count} raw={pred.raw_revisit_count} structured={pred.structured_query_count} "
            f"session_pos={pred.session_video_position} failure={pred.failure_type} "
            f"running_acc={running_correct}/{len(running)}={_format_ratio(running_accuracy)}",
            flush=True,
        )

    predictions = [running[sample.vqa_id] for sample in samples if sample.vqa_id in running]
    summary = update_run_artifacts(
        samples=samples,
        running=running,
        run_name=run_name,
        task_family=args.task_family,
        task_family_group=args.task_family_group,
        pred_path=pred_path,
        metrics_path=metrics_path,
        failures_path=failures_path,
        failure_cases_path=failure_cases_path,
        summary_path=summary_path,
        progress_path=progress_path,
    )
    write_status(
        status_path,
        status="completed",
        run_name=run_name,
        total=total,
        completed=len(predictions),
        pid=os.getpid(),
        task_family=args.task_family,
        task_family_group=args.task_family_group,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_run_name(task_family: str | None, task_family_group: str | None, limit: int, suffix: str | None) -> str:
    base = task_family_group or task_family or "mixed"
    name = f"{base}_limit{limit}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def load_selected_samples(
    index_dir: Path,
    limit: int,
    task_family: str | None,
    task_family_group: str | None,
    *,
    random_seed: int | None = None,
    video_session_mode: bool = True,
):
    if task_family_group:
        samples = []
        for family in TASK_FAMILY_GROUPS[task_family_group]:
            samples.extend(load_vqa_samples(index_dir, limit=limit, task_family=family))
        if random_seed is not None:
            import random

            rng = random.Random(random_seed)
            rng.shuffle(samples)
            samples = samples[:limit]
        if video_session_mode:
            return sorted(samples, key=lambda sample: (str(sample.primary_video_id or ""), sample.task_family, sample.vqa_id))
        return sorted(samples, key=lambda sample: (sample.task_family, sample.vqa_id))
    if random_seed is None:
        samples = load_vqa_samples(index_dir, limit=limit, task_family=task_family)
    else:
        import random
        import pandas as pd

        df = pd.read_parquet(index_dir / "vqa_samples.parquet")
        if task_family:
            df = df[df["task_family"] == task_family]
        rows = df.to_dict("records")
        rng = random.Random(random_seed)
        rng.shuffle(rows)
        rows = rows[:limit]
        from food_agent.vqa import VQASample

        samples = [VQASample.from_row(pd.Series(row)) for row in rows]
    if video_session_mode:
        return sorted(samples, key=lambda sample: (str(sample.primary_video_id or ""), sample.task_family, sample.vqa_id))
    return sorted(samples, key=lambda sample: (sample.task_family, sample.vqa_id))


def append_prediction_jsonl(path: Path, prediction: VQAPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prediction.to_json() + "\n")


def load_predictions_by_id(path: Path) -> dict[str, VQAPrediction]:
    latest: dict[str, VQAPrediction] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        payload.setdefault("attempt_count", 1)
        payload.setdefault("reuse_memory_count", 0)
        payload.setdefault("raw_revisit_count", 0)
        payload.setdefault("structured_query_count", 0)
        payload.setdefault("session_video_position", 0)
        payload.setdefault("relation_reuse_count", 0)
        payload.setdefault("planner_override_count", 0)
        payload.setdefault("verifier_blocked_finish_count", 0)
        payload.setdefault("tool_failure_count", 0)
        payload.setdefault("ineffective_tool_count", 0)
        payload.setdefault("failed_tool_recovery_count", 0)
        payload.setdefault("ineffective_tool_avoidance_count", 0)
        payload.setdefault("tool_call_count", len(payload.get("tool_calls") or []))
        payload.setdefault("reasoning_step_count", len(payload.get("tool_calls") or []))
        payload.setdefault("elapsed_seconds", None)
        payload.setdefault("prompt_tokens", 0.0)
        payload.setdefault("completion_tokens", 0.0)
        payload.setdefault("total_tokens", 0.0)
        payload.setdefault("estimated_cost", 0.0)
        pred = VQAPrediction(**payload)
        latest[pred.sample_id] = pred
    return latest


def summarize_failures(predictions: list[VQAPrediction]) -> dict[str, Any]:
    by_failure = Counter(pred.failure_type or "none" for pred in predictions)
    by_task = {}
    for pred in predictions:
        bucket = by_task.setdefault(pred.task_family, Counter())
        bucket[pred.failure_type or "none"] += 1
    return {
        "overall": dict(by_failure),
        "by_task_family": {key: dict(counter) for key, counter in by_task.items()},
    }


def build_progress_payload(*, total: int, predictions: list[VQAPrediction]) -> dict[str, Any]:
    metrics = compute_metrics(predictions)
    completed = len(predictions)
    correct = metrics.get("correct", 0) or 0
    reuse_total = sum(pred.reuse_memory_count for pred in predictions)
    raw_total = sum(pred.raw_revisit_count for pred in predictions)
    structured_total = sum(pred.structured_query_count for pred in predictions)
    relation_reuse_total = sum(pred.relation_reuse_count for pred in predictions)
    planner_override_total = sum(pred.planner_override_count for pred in predictions)
    verifier_block_total = sum(pred.verifier_blocked_finish_count for pred in predictions)
    tool_failure_total = sum(pred.tool_failure_count for pred in predictions)
    ineffective_tool_total = sum(pred.ineffective_tool_count for pred in predictions)
    failed_tool_recovery_total = sum(pred.failed_tool_recovery_count for pred in predictions)
    ineffective_tool_avoidance_total = sum(pred.ineffective_tool_avoidance_count for pred in predictions)
    tool_call_total = sum(pred.tool_call_count for pred in predictions)
    reasoning_step_total = sum(pred.reasoning_step_count for pred in predictions)
    latency_values = [float(pred.elapsed_seconds) for pred in predictions if pred.elapsed_seconds is not None]
    prompt_token_total = sum(float(pred.prompt_tokens) for pred in predictions)
    completion_token_total = sum(float(pred.completion_tokens) for pred in predictions)
    total_token_total = sum(float(pred.total_tokens) for pred in predictions)
    estimated_cost_total = sum(float(pred.estimated_cost) for pred in predictions)
    return {
        "total": total,
        "completed": completed,
        "remaining": max(total - completed, 0),
        "correct": correct,
        "accuracy": metrics.get("accuracy"),
        "avg_reuse_memory_count": (reuse_total / completed) if completed else 0.0,
        "avg_raw_revisit_count": (raw_total / completed) if completed else 0.0,
        "avg_structured_query_count": (structured_total / completed) if completed else 0.0,
        "avg_relation_reuse_count": (relation_reuse_total / completed) if completed else 0.0,
        "avg_planner_override_count": (planner_override_total / completed) if completed else 0.0,
        "avg_verifier_blocked_finish_count": (verifier_block_total / completed) if completed else 0.0,
        "avg_tool_failure_count": (tool_failure_total / completed) if completed else 0.0,
        "avg_ineffective_tool_count": (ineffective_tool_total / completed) if completed else 0.0,
        "avg_failed_tool_recovery_count": (failed_tool_recovery_total / completed) if completed else 0.0,
        "avg_ineffective_tool_avoidance_count": (ineffective_tool_avoidance_total / completed) if completed else 0.0,
        "avg_tool_call_count": (tool_call_total / completed) if completed else 0.0,
        "avg_reasoning_step_count": (reasoning_step_total / completed) if completed else 0.0,
        "avg_elapsed_seconds": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "avg_prompt_tokens": (prompt_token_total / completed) if completed else 0.0,
        "avg_completion_tokens": (completion_token_total / completed) if completed else 0.0,
        "avg_total_tokens": (total_token_total / completed) if completed else 0.0,
        "avg_estimated_cost": (estimated_cost_total / completed) if completed else 0.0,
    }


def build_failure_cases(predictions: list[VQAPrediction]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for pred in predictions:
        if pred.correct and not pred.failure_type:
            continue
        cases.append(
            {
                "sample_id": pred.sample_id,
                "task_family": pred.task_family,
                "prediction": pred.prediction,
                "gold": pred.gold,
                "correct": pred.correct,
                "failure_type": pred.failure_type,
                "tool_calls": pred.tool_calls,
                "reuse_memory_count": pred.reuse_memory_count,
                "raw_revisit_count": pred.raw_revisit_count,
                "structured_query_count": pred.structured_query_count,
                "session_video_position": pred.session_video_position,
                "relation_reuse_count": pred.relation_reuse_count,
                "planner_override_count": pred.planner_override_count,
                "verifier_blocked_finish_count": pred.verifier_blocked_finish_count,
                "tool_failure_count": pred.tool_failure_count,
                "ineffective_tool_count": pred.ineffective_tool_count,
                "failed_tool_recovery_count": pred.failed_tool_recovery_count,
                "ineffective_tool_avoidance_count": pred.ineffective_tool_avoidance_count,
                "tool_call_count": pred.tool_call_count,
                "reasoning_step_count": pred.reasoning_step_count,
                "elapsed_seconds": pred.elapsed_seconds,
                "prompt_tokens": pred.prompt_tokens,
                "completion_tokens": pred.completion_tokens,
                "total_tokens": pred.total_tokens,
                "estimated_cost": pred.estimated_cost,
            }
        )
    return cases


def build_video_positions(samples) -> dict[str, int]:
    positions: dict[str, int] = {}
    per_video_counter: Counter[str] = Counter()
    for sample in samples:
        video_id = str(sample.primary_video_id or "")
        per_video_counter[video_id] += 1
        positions[sample.vqa_id] = per_video_counter[video_id]
    return positions


def count_reuse_memory_from_result(result) -> int:
    return sum(1 for item in result.working_memory if isinstance(item, str) and item.startswith("reuse:"))


def count_tools(tool_calls: list[str], allowed: set[str]) -> int:
    return sum(1 for tool in tool_calls if tool in allowed)


def build_video_session_summary(predictions: list[VQAPrediction]) -> dict[str, Any]:
    by_video: dict[str, list[VQAPrediction]] = {}
    for pred in predictions:
        by_video.setdefault(str(pred.video_id or ""), []).append(pred)
    video_summaries: list[dict[str, Any]] = []
    for video_id, preds in sorted(by_video.items()):
        correct = sum(1 for pred in preds if pred.correct)
        tool_counts = Counter()
        task_counts = Counter()
        for pred in preds:
            tool_counts.update(pred.tool_calls)
            task_counts[pred.task_family] += 1
        count = len(preds)
        video_summaries.append(
            {
                "video_id": video_id,
                "count": count,
                "correct": correct,
                "accuracy": (correct / count) if count else None,
                "avg_reuse_memory_count": sum(pred.reuse_memory_count for pred in preds) / count if count else 0.0,
                "avg_raw_revisit_count": sum(pred.raw_revisit_count for pred in preds) / count if count else 0.0,
                "avg_structured_query_count": sum(pred.structured_query_count for pred in preds) / count if count else 0.0,
                "avg_relation_reuse_count": sum(pred.relation_reuse_count for pred in preds) / count if count else 0.0,
                "avg_planner_override_count": sum(pred.planner_override_count for pred in preds) / count if count else 0.0,
                "avg_verifier_blocked_finish_count": sum(pred.verifier_blocked_finish_count for pred in preds) / count if count else 0.0,
                "avg_tool_failure_count": sum(pred.tool_failure_count for pred in preds) / count if count else 0.0,
                "avg_ineffective_tool_count": sum(pred.ineffective_tool_count for pred in preds) / count if count else 0.0,
                "avg_failed_tool_recovery_count": sum(pred.failed_tool_recovery_count for pred in preds) / count if count else 0.0,
                "avg_ineffective_tool_avoidance_count": sum(pred.ineffective_tool_avoidance_count for pred in preds) / count if count else 0.0,
                "avg_tool_call_count": sum(pred.tool_call_count for pred in preds) / count if count else 0.0,
                "avg_reasoning_step_count": sum(pred.reasoning_step_count for pred in preds) / count if count else 0.0,
                "avg_elapsed_seconds": (
                    sum(float(pred.elapsed_seconds) for pred in preds if pred.elapsed_seconds is not None)
                    / max(sum(1 for pred in preds if pred.elapsed_seconds is not None), 1)
                ),
                "avg_prompt_tokens": sum(float(pred.prompt_tokens) for pred in preds) / count if count else 0.0,
                "avg_completion_tokens": sum(float(pred.completion_tokens) for pred in preds) / count if count else 0.0,
                "avg_total_tokens": sum(float(pred.total_tokens) for pred in preds) / count if count else 0.0,
                "avg_estimated_cost": sum(float(pred.estimated_cost) for pred in preds) / count if count else 0.0,
                "task_family_counts": dict(task_counts),
                "tool_counts": dict(tool_counts),
            }
        )
    return {
        "video_count": len(video_summaries),
        "videos": video_summaries,
    }


def build_reuse_benefit_summary(predictions: list[VQAPrediction]) -> dict[str, Any]:
    by_video: dict[str, list[VQAPrediction]] = {}
    reuse_examples: list[dict[str, Any]] = []
    cached_artifact_hits = 0
    graph_seed_hits = 0
    session_follow_up_hits = 0

    for pred in predictions:
        by_video.setdefault(str(pred.video_id or ""), []).append(pred)
        if pred.reuse_memory_count > 0 or pred.session_video_position >= 2:
            session_follow_up_hits += 1
            if len(reuse_examples) < 10:
                reuse_examples.append(
                    {
                        "sample_id": pred.sample_id,
                        "video_id": pred.video_id,
                        "task_family": pred.task_family,
                        "session_video_position": pred.session_video_position,
                        "reuse_memory_count": pred.reuse_memory_count,
                        "raw_revisit_count": pred.raw_revisit_count,
                        "structured_query_count": pred.structured_query_count,
                        "tool_call_count": pred.tool_call_count,
                        "reasoning_step_count": pred.reasoning_step_count,
                        "elapsed_seconds": pred.elapsed_seconds,
                    }
                )
        if pred.relation_reuse_count > 0:
            graph_seed_hits += 1
        if any(tool == "retrieve_cached_artifacts" for tool in pred.tool_calls):
            cached_artifact_hits += 1

    comparable_videos = 0
    first_raw_total = 0.0
    follow_raw_total = 0.0
    first_structured_total = 0.0
    follow_structured_total = 0.0
    first_tool_total = 0.0
    follow_tool_total = 0.0

    for preds in by_video.values():
        ordered = sorted(preds, key=lambda item: (item.session_video_position, item.sample_id))
        first = [item for item in ordered if item.session_video_position == 1]
        follow = [item for item in ordered if item.session_video_position >= 2]
        if not first or not follow:
            continue
        comparable_videos += 1
        first_row = first[0]
        first_raw_total += first_row.raw_revisit_count
        first_structured_total += first_row.structured_query_count
        first_tool_total += len(first_row.tool_calls)
        follow_raw_total += sum(item.raw_revisit_count for item in follow) / len(follow)
        follow_structured_total += sum(item.structured_query_count for item in follow) / len(follow)
        follow_tool_total += sum(item.tool_call_count for item in follow) / len(follow)

    first_raw_avg = (first_raw_total / comparable_videos) if comparable_videos else 0.0
    follow_raw_avg = (follow_raw_total / comparable_videos) if comparable_videos else 0.0
    first_structured_avg = (first_structured_total / comparable_videos) if comparable_videos else 0.0
    follow_structured_avg = (follow_structured_total / comparable_videos) if comparable_videos else 0.0
    first_tool_avg = (first_tool_total / comparable_videos) if comparable_videos else 0.0
    follow_tool_avg = (follow_tool_total / comparable_videos) if comparable_videos else 0.0

    return {
        "session_follow_up_hits": session_follow_up_hits,
        "graph_seed_hits": graph_seed_hits,
        "cached_artifact_hits": cached_artifact_hits,
        "comparable_video_count": comparable_videos,
        "first_question_avg_raw_revisit_count": first_raw_avg,
        "follow_up_avg_raw_revisit_count": follow_raw_avg,
        "first_question_avg_structured_query_count": first_structured_avg,
        "follow_up_avg_structured_query_count": follow_structured_avg,
        "first_question_avg_tool_calls": first_tool_avg,
        "follow_up_avg_tool_calls": follow_tool_avg,
        "follow_up_minus_first_raw_revisit_count": follow_raw_avg - first_raw_avg,
        "follow_up_minus_first_structured_query_count": follow_structured_avg - first_structured_avg,
        "follow_up_minus_first_tool_calls": follow_tool_avg - first_tool_avg,
        "reuse_examples": reuse_examples,
    }


def update_run_artifacts(
    *,
    samples,
    running: dict[str, VQAPrediction],
    run_name: str,
    task_family: str | None,
    task_family_group: str | None,
    pred_path: Path,
    metrics_path: Path,
    failures_path: Path,
    failure_cases_path: Path,
    summary_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    predictions = [running[sample.vqa_id] for sample in samples if sample.vqa_id in running]
    metrics = compute_metrics(predictions)
    failure_summary = summarize_failures(predictions)
    progress = build_progress_payload(total=len(samples), predictions=predictions)
    failure_cases = build_failure_cases(predictions)
    video_session_summary = build_video_session_summary(predictions)
    reuse_benefit_summary = build_reuse_benefit_summary(predictions)
    summary = {
        "run_name": run_name,
        "sample_count": len(samples),
        "completed_count": len(predictions),
        "task_family": task_family,
        "task_family_group": task_family_group,
        "prediction_path": pred_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "failure_summary_path": failures_path.as_posix(),
        "failure_cases_path": failure_cases_path.as_posix(),
        "progress_path": progress_path.as_posix(),
        "metrics": metrics,
        "failure_summary": failure_summary,
        "progress": progress,
        "video_session_summary": video_session_summary,
        "reuse_benefit_summary": reuse_benefit_summary,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path.write_text(json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl_records(failure_cases_path, failure_cases)
    return summary


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_status(
    path: Path,
    *,
    status: str,
    run_name: str,
    total: int,
    completed: int,
    pid: int,
    task_family: str | None,
    task_family_group: str | None,
    last_sample: str | None = None,
) -> None:
    payload = {
        "status": status,
        "run_name": run_name,
        "total": total,
        "completed": completed,
        "remaining": max(total - completed, 0),
        "pid": pid,
        "task_family": task_family,
        "task_family_group": task_family_group,
        "last_sample": last_sample,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "0.000"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
