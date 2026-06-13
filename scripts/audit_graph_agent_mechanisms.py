#!/usr/bin/env python3
"""Audit whether key graph-agent mechanisms actually appeared in run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


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
    "inspect_visual_evidence",
}


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing result artifacts to audit.")
    parser.add_argument("--predictions-file", type=Path, default=None, help="Explicit predictions JSONL file.")
    parser.add_argument("--result-file", type=Path, action="append", default=None, help="Explicit single result JSON file.")
    parser.add_argument("--session-trace", type=Path, default=None, help="Explicit session_trace.jsonl path.")
    parser.add_argument("--session-state", type=Path, default=None, help="Explicit session_state.json path.")
    parser.add_argument(
        "--out-file",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_mechanism_audit.json",
        help="Where to write the audit JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit_report(
        run_dir=args.run_dir,
        predictions_file=args.predictions_file,
        result_files=args.result_file or [],
        session_trace=args.session_trace,
        session_state=args.session_state,
    )
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_audit_report(
    *,
    run_dir: Path | None,
    predictions_file: Path | None,
    result_files: list[Path],
    session_trace: Path | None,
    session_state: Path | None,
) -> dict[str, Any]:
    discovered = discover_inputs(
        run_dir=run_dir,
        predictions_file=predictions_file,
        result_files=result_files,
        session_trace=session_trace,
        session_state=session_state,
    )
    records = load_records(discovered["prediction_files"], discovered["result_files"])
    session_trace_rows = load_jsonl_records(discovered["session_trace"]) if discovered["session_trace"] else []
    session_state_payload = load_json(discovered["session_state"]) if discovered["session_state"] else {}

    record_metrics = summarize_records(records)
    trace_metrics = summarize_session_trace(session_trace_rows)
    session_metrics = summarize_session_state(session_state_payload)

    requirements = build_requirement_report(record_metrics, trace_metrics, session_metrics)
    satisfied = sum(1 for item in requirements if item["satisfied"])
    total = len(requirements)

    return {
        "audit_version": 1,
        "inputs": {
            "run_dir": discovered["run_dir"].as_posix() if discovered["run_dir"] else None,
            "prediction_files": [path.as_posix() for path in discovered["prediction_files"]],
            "result_files": [path.as_posix() for path in discovered["result_files"]],
            "session_trace": discovered["session_trace"].as_posix() if discovered["session_trace"] else None,
            "session_state": discovered["session_state"].as_posix() if discovered["session_state"] else None,
        },
        "summary": {
            "record_count": len(records),
            "session_trace_count": len(session_trace_rows),
            "requirements_satisfied": satisfied,
            "requirements_total": total,
            "coverage_ratio": (satisfied / total) if total else 0.0,
        },
        "mechanism_metrics": {
            "records": record_metrics,
            "session_trace": trace_metrics,
            "session_state": session_metrics,
        },
        "requirements": requirements,
    }


def discover_inputs(
    *,
    run_dir: Path | None,
    predictions_file: Path | None,
    result_files: list[Path],
    session_trace: Path | None,
    session_state: Path | None,
) -> dict[str, Any]:
    discovered_run_dir = run_dir
    prediction_files: list[Path] = []
    explicit_result_files = [path for path in result_files if path.exists()]
    if predictions_file is not None and predictions_file.exists():
        prediction_files.append(predictions_file)
    if run_dir is not None and run_dir.exists():
        prediction_files.extend(sorted(run_dir.glob("*predictions*.jsonl")))
        prediction_files.extend(sorted(run_dir.glob("*records*.jsonl")))
        if session_trace is None:
            maybe_trace = run_dir / "session_trace.jsonl"
            if maybe_trace.exists():
                session_trace = maybe_trace
        if session_state is None:
            maybe_state = run_dir / "session_state.json"
            if maybe_state.exists():
                session_state = maybe_state
    if not prediction_files and explicit_result_files:
        parent = explicit_result_files[0].parent
        discovered_run_dir = parent if parent.exists() else discovered_run_dir
    return {
        "run_dir": discovered_run_dir,
        "prediction_files": dedupe_paths(prediction_files),
        "result_files": dedupe_paths(explicit_result_files),
        "session_trace": session_trace if session_trace and session_trace.exists() else None,
        "session_state": session_state if session_state and session_state.exists() else None,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts = Counter()
    working_memory_entries = 0
    open_question_entries = 0
    verification_entries = 0
    metrics = Counter()
    videos = set()
    task_families = Counter()

    for record in records:
        videos.add(str(record.get("video_id") or ""))
        if record.get("task_family"):
            task_families[str(record["task_family"])] += 1
        trace = record.get("tool_trace") or []
        if isinstance(trace, list):
            for entry in trace:
                if not isinstance(entry, dict):
                    continue
                tool = str(entry.get("tool") or "")
                if tool:
                    tool_counts[tool] += 1
                    if tool in RAW_REVISIT_TOOLS:
                        metrics["raw_revisit_tool_calls"] += 1
                    if tool == "expand_graph_context":
                        metrics["relation_expansion_calls"] += 1
                raw_result = entry.get("raw_result")
                if isinstance(raw_result, dict):
                    if raw_result.get("tool_failed"):
                        metrics["tool_failures_in_trace"] += 1
                    if raw_result.get("tool_ineffective"):
                        metrics["tool_ineffective_in_trace"] += 1
            metrics["failed_tool_recovery_count"] += count_recovery_events(trace, failure_key="tool_failed")
            metrics["ineffective_tool_avoidance_count"] += count_recovery_events(trace, failure_key="tool_ineffective")

        working_memory = record.get("working_memory") or []
        if isinstance(working_memory, list):
            working_memory_entries += len(working_memory)
            for item in working_memory:
                if not isinstance(item, str):
                    continue
                if item.startswith("reuse:"):
                    metrics["memory_reuse_items"] += 1
                if item.startswith("reuse_relation:"):
                    metrics["relation_reuse_items"] += 1
                if item.startswith("reuse_relation_edge:"):
                    metrics["relation_reuse_edges"] += 1
                if item.startswith("planner_override "):
                    metrics["planner_override_items"] += 1
                if item.startswith("planner_override verifier_blocked_finish="):
                    metrics["verifier_blocked_finish_items"] += 1
                if item.startswith("conflict_detected="):
                    metrics["conflict_detected_items"] += 1
                if item.startswith("conflict_hint type="):
                    metrics["conflict_hint_items"] += 1
                if item.startswith("conflict_resolved="):
                    metrics["conflict_resolved_items"] += 1
                if item.startswith("writeback_skipped "):
                    metrics["writeback_skipped_items"] += 1
                if item == "session_writeback_skipped reason=conflict":
                    metrics["session_writeback_skipped_items"] += 1
                if item == "freeform_answer_mode=structured_summary":
                    metrics["freeform_structured_summary_items"] += 1
                if item == "freeform_answer_mode=grounded_structured_answer":
                    metrics["freeform_structured_summary_items"] += 1
                if item == "freeform_answer_mode=fallback_summary":
                    metrics["freeform_fallback_items"] += 1
                if item.startswith("deterministic_finalize prediction="):
                    metrics["deterministic_finalize_items"] += 1
                if item.startswith("cached_artifact ") or item.startswith("reuse_cached_artifact="):
                    metrics["cached_artifact_reuse_items"] += 1

        open_questions = record.get("open_questions") or []
        if isinstance(open_questions, list):
            open_question_entries += len(open_questions)
            for item in open_questions:
                if isinstance(item, str) and item.startswith("conflict:"):
                    metrics["open_conflict_items"] += 1

        verification_history = record.get("verification_history") or []
        if isinstance(verification_history, list):
            verification_entries += len(verification_history)
            for item in verification_history:
                if not isinstance(item, dict):
                    continue
                if item.get("sufficient") is False:
                    metrics["verifier_insufficient_events"] += 1
                conflicts = item.get("conflicts") or []
                missing = item.get("missing_evidence_types") or []
                if conflicts:
                    metrics["verifier_conflict_events"] += 1
                if missing:
                    metrics["verifier_missing_events"] += 1

        tool_failures = record.get("tool_failures") or []
        if isinstance(tool_failures, list):
            metrics["tool_failure_list_items"] += len(tool_failures)
        ineffective_tools = record.get("ineffective_tools") or []
        if isinstance(ineffective_tools, list):
            metrics["ineffective_tool_list_items"] += len(ineffective_tools)

    metrics["record_count"] = len(records)
    metrics["working_memory_entries"] = working_memory_entries
    metrics["open_question_entries"] = open_question_entries
    metrics["verification_entries"] = verification_entries
    return {
        "counts": dict(metrics),
        "tool_counts": dict(tool_counts),
        "task_family_counts": dict(task_families),
        "video_ids": sorted(video for video in videos if video),
    }


def summarize_session_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts = Counter()
    metrics = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        tool_calls = row.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tool in tool_calls:
                if not tool:
                    continue
                tool_name = str(tool)
                tool_counts[tool_name] += 1
                if tool_name in RAW_REVISIT_TOOLS:
                    metrics["raw_revisit_tool_calls"] += 1
                if tool_name == "expand_graph_context":
                    metrics["relation_expansion_calls"] += 1
        latest_verification = row.get("latest_verification") or {}
        if isinstance(latest_verification, dict) and latest_verification:
            if latest_verification.get("sufficient") is False:
                metrics["verifier_insufficient_events"] += 1
            if latest_verification.get("conflicts"):
                metrics["verifier_conflict_events"] += 1
            if latest_verification.get("missing_evidence_types"):
                metrics["verifier_missing_events"] += 1
        for item in row.get("working_memory_tail") or []:
            if not isinstance(item, str):
                continue
            if item.startswith("planner_override "):
                metrics["planner_override_items"] += 1
            if item.startswith("conflict_hint type="):
                metrics["conflict_hint_items"] += 1
            if item.startswith("conflict_resolved="):
                metrics["conflict_resolved_items"] += 1
            if item.startswith("writeback_skipped "):
                metrics["writeback_skipped_items"] += 1
            if item == "session_writeback_skipped reason=conflict":
                metrics["session_writeback_skipped_items"] += 1
    metrics["trace_rows"] = len(rows)
    return {"counts": dict(metrics), "tool_counts": dict(tool_counts)}


def summarize_session_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"counts": {}}
    counts = Counter()
    session_memory = payload.get("session_memory")
    if isinstance(session_memory, dict):
        for item in session_memory.get("working_memory") or []:
            if not isinstance(item, str):
                continue
            if item.startswith("reuse:"):
                counts["memory_reuse_items"] += 1
            if item.startswith("reuse_relation:"):
                counts["relation_reuse_items"] += 1
            if item.startswith("deterministic_finalize prediction="):
                counts["deterministic_finalize_items"] += 1
            if item.startswith("reuse_cached_artifact="):
                counts["cached_artifact_reuse_items"] += 1
        counts["persisted_working_memory_entries"] = len(session_memory.get("working_memory") or [])
        counts["persisted_evidence_entries"] = len(session_memory.get("evidence_bundle") or [])
    if payload.get("question_count") is not None:
        counts["question_count"] = int(payload.get("question_count") or 0)
    return {"counts": dict(counts)}


def build_requirement_report(
    record_metrics: dict[str, Any],
    trace_metrics: dict[str, Any],
    session_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    record_counts = record_metrics.get("counts", {})
    trace_counts = trace_metrics.get("counts", {})
    session_counts = session_metrics.get("counts", {})

    def total_count(key: str) -> int:
        return int(record_counts.get(key, 0)) + int(trace_counts.get(key, 0)) + int(session_counts.get(key, 0))

    requirements = [
        requirement(
            "memory_reuse",
            "同视频记忆复用",
            total_count("memory_reuse_items"),
            "需要看到 reuse: 记忆进入 working memory 或 session memory。",
        ),
        requirement(
            "relation_expansion",
            "图关系扩展",
            total_count("relation_expansion_calls")
            + total_count("relation_reuse_items")
            + total_count("relation_reuse_edges"),
            "需要看到 expand_graph_context 或 reuse_relation: 证据。",
        ),
        requirement(
            "raw_revisit",
            "原始多模态回看",
            total_count("raw_revisit_tool_calls"),
            "需要看到抽帧、局部区域、OCR、视觉检查等 raw revisit 工具调用。",
        ),
        requirement(
            "verifier_block",
            "Verifier 阻止过过早结束",
            total_count("verifier_blocked_finish_items") + total_count("verifier_insufficient_events"),
            "需要看到 verifier 不允许 finish，或 planner_override verifier_blocked_finish。",
        ),
        requirement(
            "planner_override",
            "Planner 覆盖模型原始决策",
            total_count("planner_override_items"),
            "需要看到 planner_override 轨迹。",
        ),
        requirement(
            "tool_failure_recovery",
            "工具失败后恢复",
            total_count("failed_tool_recovery_count"),
            "需要看到某工具失败后转向其他工具继续完成。",
        ),
        requirement(
            "ineffective_tool_avoidance",
            "空转工具后规避",
            total_count("ineffective_tool_avoidance_count"),
            "需要看到 tool_ineffective 后切换其他路径。",
        ),
        requirement(
            "conflict_detection",
            "冲突检测",
            total_count("conflict_detected_items")
            + total_count("conflict_hint_items")
            + total_count("open_conflict_items")
            + total_count("verifier_conflict_events"),
            "需要看到 conflict:*、conflict_detected= 或 verifier conflict。",
        ),
        requirement(
            "conflict_resolution",
            "冲突恢复",
            total_count("conflict_resolved_items"),
            "需要看到 conflict_resolved=...。",
        ),
        requirement(
            "writeback_hygiene",
            "冲突时脏写回阻止",
            total_count("writeback_skipped_items"),
            "需要看到 writeback_skipped ...。",
        ),
        requirement(
            "session_writeback_hygiene",
            "冲突时 session 写回阻止",
            total_count("session_writeback_skipped_items"),
            "需要看到 session_writeback_skipped reason=conflict。",
        ),
        requirement(
            "open_query_structured_summary",
            "开放问答结构化总结",
            total_count("freeform_structured_summary_items"),
            "需要看到 freeform_answer_mode=structured_summary。",
        ),
        requirement(
            "deterministic_finalize",
            "结构化最终收口",
            total_count("deterministic_finalize_items"),
            "需要看到 deterministic_finalize prediction=...，表示最终答案直接由结构化证据收口。",
        ),
        requirement(
            "cached_artifact_reuse",
            "缓存原始证据复用",
            total_count("cached_artifact_reuse_items"),
            "需要看到 cached_artifact ... 或 reuse_cached_artifact=...，表示复用了已缓存的原始 artifact。",
        ),
        requirement(
            "session_persistence",
            "Session 持久化",
            int(session_counts.get("question_count", 0)) + int(session_counts.get("persisted_working_memory_entries", 0)),
            "需要看到 session_state 里保存的问题计数或持久化记忆。",
        ),
    ]
    return requirements


def requirement(requirement_id: str, label: str, observed_count: int, expectation: str) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "label": label,
        "satisfied": observed_count > 0,
        "observed_count": observed_count,
        "expectation": expectation,
    }


def load_records(prediction_files: list[Path], result_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in prediction_files:
        for item in load_jsonl_records(path):
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    for path in result_files:
        payload = load_json(path)
        if isinstance(payload, dict):
            key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                records.append(payload)
    return records


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
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


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.as_posix()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
