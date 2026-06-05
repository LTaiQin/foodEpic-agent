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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)
    samples = load_selected_samples(args.index_dir, args.limit, args.task_family, args.task_family_group)
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
    for index, sample in enumerate(samples, start=1):
        if sample.vqa_id in completed:
            pred = completed[sample.vqa_id]
            print(
                f"[{index}/{total}] skip sample={sample.vqa_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct} failure={pred.failure_type}",
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
                tool_calls=[entry.get("tool", "") for entry in result.tool_trace],
                failure_type=None if result.prediction == sample.correct_idx else "reasoning_error",
                attempt_count=1,
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
            f"failure={pred.failure_type} running_acc={running_correct}/{len(running)}={_format_ratio(running_accuracy)}",
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


def load_selected_samples(index_dir: Path, limit: int, task_family: str | None, task_family_group: str | None):
    if task_family_group:
        samples = []
        for family in TASK_FAMILY_GROUPS[task_family_group]:
            samples.extend(load_vqa_samples(index_dir, limit=limit, task_family=family))
        return samples
    return load_vqa_samples(index_dir, limit=limit, task_family=task_family)


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
    return {
        "total": total,
        "completed": completed,
        "remaining": max(total - completed, 0),
        "correct": correct,
        "accuracy": metrics.get("accuracy"),
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
            }
        )
    return cases


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
