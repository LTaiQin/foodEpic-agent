#!/usr/bin/env python3
"""Aggregate small-run graph-agent artifacts into a single report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-summary", action="append", default=[])
    parser.add_argument("--real-summary", action="append", default=[])
    parser.add_argument("--probe-predictions", action="append", default=[])
    parser.add_argument("--real-predictions", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_small_run_report(
        probe_summaries=[Path(item) for item in args.probe_summary],
        real_summaries=[Path(item) for item in args.real_summary],
        probe_predictions=[Path(item) for item in args.probe_predictions],
        real_predictions=[Path(item) for item in args.real_predictions],
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_small_run_report(
    *,
    probe_summaries: list[Path],
    real_summaries: list[Path],
    probe_predictions: list[Path],
    real_predictions: list[Path],
) -> dict[str, Any]:
    probe_summary_rows = [load_json(path) for path in probe_summaries if path.exists()]
    real_summary_rows = [load_json(path) for path in real_summaries if path.exists()]
    probe_prediction_rows = [row for path in probe_predictions if path.exists() for row in load_jsonl(path)]
    real_prediction_rows = [row for path in real_predictions if path.exists() for row in load_jsonl(path)]

    all_prediction_rows = probe_prediction_rows + real_prediction_rows
    task_families = sorted(
        {
            str(row.get("task_family") or family)
            for row in all_prediction_rows
            for family in [row.get("task_family")]
            if family
        }
    )

    return {
        "artifact_counts": {
            "probe_summary_count": len(probe_summary_rows),
            "real_summary_count": len(real_summary_rows),
            "probe_prediction_count": len(probe_prediction_rows),
            "real_prediction_count": len(real_prediction_rows),
        },
        "coverage": {
            "task_family_count": len(task_families),
            "task_families": task_families,
            "sample_count": len(all_prediction_rows),
            "vqa_sample_count": sum(1 for row in all_prediction_rows if row.get("gold") is not None),
            "open_query_sample_count": sum(1 for row in all_prediction_rows if row.get("gold") is None),
            "video_count": len({str(row.get("video_id") or "") for row in all_prediction_rows if row.get("video_id")}),
        },
        "accuracy_summary": build_accuracy_summary(all_prediction_rows),
        "agent_metrics": build_agent_metric_summary(probe_summary_rows, real_summary_rows, all_prediction_rows),
        "attempt_summary": build_attempt_summary_from_rows(all_prediction_rows, probe_summary_rows, real_summary_rows),
        "failure_summary": build_failure_summary(all_prediction_rows, probe_summary_rows, real_summary_rows),
        "success_examples": build_example_rows(all_prediction_rows, want_correct=True),
        "failure_examples": build_example_rows(all_prediction_rows, want_correct=False),
        "sources": {
            "probe_summaries": [path.as_posix() for path in probe_summaries],
            "real_summaries": [path.as_posix() for path in real_summaries],
            "probe_predictions": [path.as_posix() for path in probe_predictions],
            "real_predictions": [path.as_posix() for path in real_predictions],
        },
    }


def build_accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vqa_rows = [row for row in rows if row.get("gold") is not None]
    correct = sum(1 for row in vqa_rows if row.get("correct") is True)
    by_family: dict[str, dict[str, Any]] = {}
    for row in vqa_rows:
        family = str(row.get("task_family") or "unknown")
        bucket = by_family.setdefault(family, {"count": 0, "correct": 0})
        bucket["count"] += 1
        bucket["correct"] += int(row.get("correct") is True)
    for bucket in by_family.values():
        bucket["accuracy"] = (bucket["correct"] / bucket["count"]) if bucket["count"] else None
    return {
        "count": len(vqa_rows),
        "correct": correct,
        "accuracy": (correct / len(vqa_rows)) if vqa_rows else None,
        "by_task_family": by_family,
    }


def build_agent_metric_summary(
    probe_summary_rows: list[dict[str, Any]],
    real_summary_rows: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_rows = [extract_prediction_metrics(row) for row in all_prediction_rows]
    completed = len(metric_rows)
    probe_progress_rows = [row.get("progress") or {} for row in probe_summary_rows]
    real_progress_rows = real_summary_rows

    def avg(metric: str) -> float:
        return (sum(float(row.get(metric) or 0.0) for row in metric_rows) / completed) if completed else 0.0

    def max_from_nested(rows: list[dict[str, Any]], key: str) -> float:
        values = [float(row.get(key) or 0.0) for row in rows]
        return max(values) if values else 0.0

    return {
        "memory_reuse_rate": rate(metric_rows, "reuse_memory_hit"),
        "relation_reuse_rate": rate(metric_rows, "relation_reuse_hit"),
        "raw_revisit_rate": rate(metric_rows, "raw_revisit_hit"),
        "planner_override_count": sum(int(row["planner_override_count"]) for row in metric_rows),
        "verifier_blocked_finish_count": sum(int(row["verifier_blocked_finish_count"]) for row in metric_rows),
        "tool_failure_recovery_rate": recovery_rate(metric_rows, "tool_failure_count", "failed_tool_recovery_count"),
        "ineffective_tool_avoidance_rate": recovery_rate(
            metric_rows, "ineffective_tool_count", "ineffective_tool_avoidance_count"
        ),
        "cached_artifact_reuse_hits": sum(int(row["cached_artifact_hit"]) for row in metric_rows),
        "avg_tool_calls_per_question": avg("tool_call_count"),
        "avg_reasoning_steps_per_question": avg("reasoning_step_count"),
        "avg_wall_clock_latency_per_question": avg("elapsed_seconds"),
        "avg_prompt_tokens_per_question": avg("prompt_tokens"),
        "avg_completion_tokens_per_question": avg("completion_tokens"),
        "avg_total_tokens_per_question": avg("total_tokens"),
        "avg_api_cost_per_question": avg("estimated_cost"),
        "same_video_follow_up_cost_reduction": max_from_nested(
            [row.get("same_video_follow_up_cost_reduction") or {} for row in real_progress_rows],
            "follow_up_minus_first_estimated_cost",
        ),
        "cached_artifact_reuse_gain": [
            row.get("cached_artifact_reuse_gain") or {} for row in real_progress_rows if row.get("cached_artifact_reuse_gain")
        ],
        "video_session_summaries": [
            row.get("video_session_summary") or {}
            for row in probe_summary_rows + real_summary_rows
            if row.get("video_session_summary")
        ],
        "probe_progress_snapshots": probe_progress_rows,
    }


def build_attempt_summary_from_rows(
    all_prediction_rows: list[dict[str, Any]],
    probe_summary_rows: list[dict[str, Any]],
    real_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt_counts = [int(row.get("attempt_count") or 1) for row in all_prediction_rows]
    nested = [row.get("attempt_summary") or {} for row in probe_summary_rows + real_summary_rows]
    return {
        "max_attempt_count": max(attempt_counts) if attempt_counts else 0,
        "avg_attempt_count": (sum(attempt_counts) / len(attempt_counts)) if attempt_counts else 0.0,
        "retried_sample_count": sum(1 for count in attempt_counts if count > 1),
        "nested_attempt_summaries": nested,
    }


def build_failure_summary(
    all_prediction_rows: list[dict[str, Any]],
    probe_summary_rows: list[dict[str, Any]],
    real_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    overall: dict[str, int] = {}
    for row in all_prediction_rows:
        key = str(row.get("failure_type") or "none")
        overall[key] = overall.get(key, 0) + 1
    return {
        "overall": overall,
        "probe_failure_summaries": [row.get("failure_summary") or {} for row in probe_summary_rows],
        "real_failure_counts": [row.get("failure_counts") or {} for row in real_summary_rows],
    }


def build_example_rows(rows: list[dict[str, Any]], *, want_correct: bool) -> list[dict[str, Any]]:
    filtered = [row for row in rows if (row.get("correct") is True) == want_correct and row.get("gold") is not None]
    filtered.sort(
        key=lambda row: (
            -int(row.get("correct") is True),
            float(row.get("estimated_cost") or ((row.get("usage") or {}).get("estimated_cost") or 0.0)),
            str(row.get("sample_id") or row.get("vqa_id") or ""),
        )
    )
    examples: list[dict[str, Any]] = []
    for row in filtered[:10]:
        metrics = extract_prediction_metrics(row)
        examples.append(
            {
                "sample_id": row.get("sample_id") or row.get("vqa_id"),
                "task_family": row.get("task_family"),
                "video_id": row.get("video_id"),
                "prediction": row.get("prediction"),
                "gold": row.get("gold"),
                "correct": row.get("correct"),
                "failure_type": row.get("failure_type"),
                "tool_calls": row.get("tool_calls") or [],
                "tool_call_count": metrics["tool_call_count"],
                "reasoning_step_count": metrics["reasoning_step_count"],
                "estimated_cost": metrics["estimated_cost"],
                "total_tokens": metrics["total_tokens"],
            }
        )
    return examples


def extract_prediction_metrics(row: dict[str, Any]) -> dict[str, Any]:
    usage = row.get("usage") or {}
    tool_calls = row.get("tool_calls") or []
    working_memory = row.get("working_memory") or []
    tool_trace = row.get("tool_trace") or []
    prompt_tokens = float(row.get("prompt_tokens") or usage.get("prompt_tokens") or 0.0)
    completion_tokens = float(row.get("completion_tokens") or usage.get("completion_tokens") or 0.0)
    total_tokens = float(row.get("total_tokens") or usage.get("total_tokens") or 0.0)
    estimated_cost = float(row.get("estimated_cost") or usage.get("estimated_cost") or 0.0)
    elapsed_seconds = row.get("elapsed_seconds")
    return {
        "tool_call_count": len(tool_calls),
        "reasoning_step_count": len(tool_trace) if tool_trace else len(tool_calls),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "reuse_memory_hit": any(isinstance(item, str) and item.startswith("reuse:") for item in working_memory),
        "relation_reuse_hit": any(
            isinstance(item, str) and item.startswith("reuse_relation:") for item in working_memory
        ),
        "raw_revisit_hit": any(
            tool in {
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
            for tool in tool_calls
        ),
        "planner_override_count": sum(
            1 for item in working_memory if isinstance(item, str) and item.startswith("planner_override ")
        ),
        "verifier_blocked_finish_count": sum(
            1
            for item in working_memory
            if isinstance(item, str) and item.startswith("planner_override verifier_blocked_finish=")
        ),
        "tool_failure_count": len(row.get("tool_failures") or []),
        "failed_tool_recovery_count": int(bool(row.get("tool_failures")) and len(tool_calls) > len(row.get("tool_failures") or [])),
        "ineffective_tool_count": len(row.get("ineffective_tools") or []),
        "ineffective_tool_avoidance_count": int(
            bool(row.get("ineffective_tools")) and len(tool_calls) > len(row.get("ineffective_tools") or [])
        ),
        "cached_artifact_hit": any(tool == "retrieve_cached_artifacts" for tool in tool_calls),
    }


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return (sum(1 for row in rows if row.get(key)) / len(rows)) if rows else 0.0


def recovery_rate(rows: list[dict[str, Any]], failure_key: str, recovery_key: str) -> float:
    relevant = [row for row in rows if int(row.get(failure_key) or 0) > 0]
    if not relevant:
        return 0.0
    recovered = sum(1 for row in relevant if int(row.get(recovery_key) or 0) > 0)
    return recovered / len(relevant)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def render_markdown_report(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    accuracy = report["accuracy_summary"]
    metrics = report["agent_metrics"]
    lines = [
        "# Graph Agent Small Run Report",
        "",
        "## Coverage",
        f"- task families: {coverage['task_family_count']}",
        f"- videos: {coverage['video_count']}",
        f"- samples: {coverage['sample_count']}",
        f"- VQA samples: {coverage['vqa_sample_count']}",
        f"- open-query samples: {coverage['open_query_sample_count']}",
        "",
        "## Accuracy",
        f"- overall accuracy: {format_float(accuracy.get('accuracy'))}",
        f"- correct / total: {accuracy.get('correct', 0)} / {accuracy.get('count', 0)}",
        "",
        "## Agent Metrics",
        f"- memory reuse rate: {format_float(metrics.get('memory_reuse_rate'))}",
        f"- relation reuse rate: {format_float(metrics.get('relation_reuse_rate'))}",
        f"- raw revisit rate: {format_float(metrics.get('raw_revisit_rate'))}",
        f"- planner override count: {metrics.get('planner_override_count', 0)}",
        f"- verifier blocked finish count: {metrics.get('verifier_blocked_finish_count', 0)}",
        f"- tool failure recovery rate: {format_float(metrics.get('tool_failure_recovery_rate'))}",
        f"- ineffective tool avoidance rate: {format_float(metrics.get('ineffective_tool_avoidance_rate'))}",
        f"- avg tool calls / question: {format_float(metrics.get('avg_tool_calls_per_question'))}",
        f"- avg reasoning steps / question: {format_float(metrics.get('avg_reasoning_steps_per_question'))}",
        f"- avg latency / question: {format_float(metrics.get('avg_wall_clock_latency_per_question'))}",
        f"- avg total tokens / question: {format_float(metrics.get('avg_total_tokens_per_question'))}",
        f"- avg API cost / question: {format_float(metrics.get('avg_api_cost_per_question'))}",
        "",
        "## Failure Summary",
    ]
    for key, value in sorted((report["failure_summary"].get("overall") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Success Examples"])
    for row in report["success_examples"]:
        lines.append(
            f"- {row['sample_id']} | {row['task_family']} | pred={row['prediction']} gold={row['gold']} "
            f"| tools={row['tool_call_count']} | cost={format_float(row['estimated_cost'])}"
        )
    lines.extend(["", "## Failure Examples"])
    for row in report["failure_examples"]:
        lines.append(
            f"- {row['sample_id']} | {row['task_family']} | pred={row['prediction']} gold={row['gold']} "
            f"| failure={row['failure_type']} | tools={row['tool_call_count']}"
        )
    return "\n".join(lines) + "\n"


def format_float(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
