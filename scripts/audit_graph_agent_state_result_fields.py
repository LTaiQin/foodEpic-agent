#!/usr/bin/env python3
"""Audit GraphAgent state/result/session field coverage from real artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


AGENT_STATE_SESSION_KEYS = [
    "video_id",
    "working_memory",
    "evidence_bundle",
    "retrieved_frames",
    "visited_times",
    "artifacts",
    "retrieved_node_ids",
    "retrieved_nodes",
    "hypotheses",
    "open_questions",
    "tool_failures",
    "ineffective_tools",
    "verification_history",
    "confidence",
]

RESULT_KEYS = [
    "vqa_id",
    "video_id",
    "task_family",
    "prediction",
    "answer_text",
    "confidence",
    "elapsed_seconds",
    "usage",
    "tool_trace",
    "evidence_bundle",
    "working_memory",
    "retrieved_frames",
    "visited_times",
    "artifacts",
    "verification_history",
    "latest_verification",
    "tool_failures",
    "ineffective_tools",
    "open_questions",
    "tool_calls",
    "tool_call_count",
    "failure_count",
    "ineffective_tool_count",
    "verification_count",
    "raw_model_output",
    "question",
    "choices_json",
    "inputs_json",
]

SESSION_STATE_TOP_KEYS = [
    "video_id",
    "question_count",
    "last_vqa_id",
    "last_task_family",
    "last_prediction",
    "last_elapsed_seconds",
    "updated_at",
    "session_memory",
]

SESSION_TRACE_KEYS = [
    "vqa_id",
    "task_family",
    "prediction",
    "elapsed_seconds",
    "question_count",
    "tool_calls",
    "visited_times_tail",
    "artifacts_tail",
    "tool_failures",
    "ineffective_tools",
    "latest_verification",
    "open_questions_tail",
    "working_memory_tail",
    "evidence_tail",
]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=defaults.output_root / "results")
    parser.add_argument("--sessions-root", type=Path, default=defaults.graph_agent_sessions_root)
    parser.add_argument("--runs-root", type=Path, default=defaults.graph_agent_runs_root)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=defaults.output_root / "reports" / "graph_agent_state_result_field_audit.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=defaults.output_root / "reports" / "graph_agent_state_result_field_audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit_report(
        results_root=args.results_root,
        sessions_root=args.sessions_root,
        runs_root=args.runs_root,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_audit_report(*, results_root: Path, sessions_root: Path, runs_root: Path) -> dict[str, Any]:
    prediction_rows = load_prediction_rows(results_root)
    session_state_payloads = load_json_paths(sessions_root.glob("*/session_state.json"))
    session_trace_rows = load_jsonl_paths(sessions_root.glob("*/session_trace.jsonl"))
    result_payloads = load_result_payloads(runs_root, prediction_rows)

    sections = {
        "graph_agent_result": audit_required_keys(RESULT_KEYS, result_payloads),
        "session_state_top_level": audit_required_keys(SESSION_STATE_TOP_KEYS, session_state_payloads),
        "session_state_memory": audit_required_keys(
            AGENT_STATE_SESSION_KEYS,
            [
                payload.get("session_memory")
                for payload in session_state_payloads
                if isinstance(payload, dict) and isinstance(payload.get("session_memory"), dict)
            ],
        ),
        "session_trace": audit_required_keys(SESSION_TRACE_KEYS, session_trace_rows),
    }
    linkage = audit_linkage(prediction_rows, result_payloads, session_state_payloads, session_trace_rows)
    summary = build_summary(sections, linkage, prediction_rows, result_payloads, session_state_payloads, session_trace_rows)
    return {
        "summary": summary,
        "sections": sections,
        "linkage": linkage,
        "artifacts": {
            "prediction_row_count": len(prediction_rows),
            "result_payload_count": len(result_payloads),
            "session_state_count": len(session_state_payloads),
            "session_trace_row_count": len(session_trace_rows),
        },
    }


def load_prediction_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in results_root.glob("**/predictions_graph_agent.jsonl"):
        rows.extend(load_jsonl(path))
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("task_family") or ""), str(row.get("vqa_id") or ""))
        deduped[key] = row
    return list(deduped.values())


def load_result_payloads(runs_root: Path, prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for row in prediction_rows:
        task_family = str(row.get("task_family") or "")
        vqa_id = str(row.get("vqa_id") or "")
        if not task_family or not vqa_id:
            continue
        path = runs_root / task_family / f"{safe_filename(vqa_id)}.json"
        if path.exists() and path not in seen:
            payloads.append(load_json(path))
            seen.add(path)
    return payloads


def audit_required_keys(required_keys: list[str], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    key_presence = Counter()
    type_examples: dict[str, str] = {}
    payload_count = len(payloads)
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in required_keys:
            if key in payload:
                key_presence[key] += 1
                if key not in type_examples:
                    type_examples[key] = type(payload.get(key)).__name__
    missing = [key for key in required_keys if key_presence.get(key, 0) == 0]
    partial = [key for key in required_keys if 0 < key_presence.get(key, 0) < payload_count]
    return {
        "payload_count": payload_count,
        "required_keys": required_keys,
        "present_in_all": [key for key in required_keys if payload_count and key_presence.get(key, 0) == payload_count],
        "missing_in_all": missing,
        "partial_coverage": {key: key_presence.get(key, 0) for key in partial},
        "type_examples": type_examples,
    }


def audit_linkage(
    prediction_rows: list[dict[str, Any]],
    result_payloads: list[dict[str, Any]],
    session_state_payloads: list[dict[str, Any]],
    session_trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    real_prediction_rows = [row for row in prediction_rows if is_real_artifact_row(row)]
    real_result_payloads = [row for row in result_payloads if is_real_artifact_row(row)]
    real_session_trace_rows = [row for row in session_trace_rows if is_real_artifact_row(row)]
    real_session_state_payloads = [row for row in session_state_payloads if is_real_session_state(row)]

    result_ids = {str(item.get("vqa_id") or "") for item in real_result_payloads if isinstance(item, dict)}
    prediction_ids = {str(item.get("vqa_id") or "") for item in real_prediction_rows if isinstance(item, dict)}
    trace_ids = {str(item.get("vqa_id") or "") for item in real_session_trace_rows if isinstance(item, dict)}
    state_video_ids = {
        str(item.get("video_id") or "")
        for item in real_session_state_payloads
        if isinstance(item, dict) and item.get("video_id")
    }
    prediction_video_ids = {
        str(item.get("video_id") or "")
        for item in real_prediction_rows
        if isinstance(item, dict) and item.get("video_id")
    }
    return {
        "real_prediction_count": len(real_prediction_rows),
        "prediction_without_result_count": len(prediction_ids - result_ids),
        "prediction_without_session_trace_count": len(prediction_ids - trace_ids),
        "prediction_video_without_session_state_count": len(prediction_video_ids - state_video_ids),
        "prediction_without_result_examples": sorted(prediction_ids - result_ids)[:10],
        "prediction_without_session_trace_examples": sorted(prediction_ids - trace_ids)[:10],
        "prediction_video_without_session_state_examples": sorted(prediction_video_ids - state_video_ids)[:10],
    }


def build_summary(
    sections: dict[str, dict[str, Any]],
    linkage: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    result_payloads: list[dict[str, Any]],
    session_state_payloads: list[dict[str, Any]],
    session_trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_required = 0
    total_missing = 0
    partial_keys = 0
    for section in sections.values():
        total_required += len(section["required_keys"])
        total_missing += len(section["missing_in_all"])
        partial_keys += len(section["partial_coverage"])
    return {
        "prediction_row_count": len(prediction_rows),
        "result_payload_count": len(result_payloads),
        "session_state_count": len(session_state_payloads),
        "session_trace_row_count": len(session_trace_rows),
        "required_key_count": total_required,
        "missing_key_count": total_missing,
        "partial_key_count": partial_keys,
        "all_required_keys_observed": total_missing == 0,
        "linkage_ok": all(
            int(value) == 0
            for key, value in linkage.items()
            if key.endswith("_count") and key != "real_prediction_count"
        ),
    }


def is_real_artifact_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    vqa_id = str(row.get("vqa_id") or "")
    video_id = str(row.get("video_id") or "")
    if not vqa_id or vqa_id.startswith("mechanism:"):
        return False
    if video_id.startswith("vid_"):
        return False
    return True


def is_real_session_state(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    video_id = str(payload.get("video_id") or "")
    return bool(video_id) and not video_id.startswith("vid_")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Graph Agent State/Result Field Audit", ""]
    summary = report["summary"]
    lines.extend(
        [
            "## Summary",
            f"- prediction rows: {summary['prediction_row_count']}",
            f"- result payloads: {summary['result_payload_count']}",
            f"- session states: {summary['session_state_count']}",
            f"- session trace rows: {summary['session_trace_row_count']}",
            f"- required keys tracked: {summary['required_key_count']}",
            f"- missing keys: {summary['missing_key_count']}",
            f"- partial keys: {summary['partial_key_count']}",
            f"- all required keys observed: {summary['all_required_keys_observed']}",
            f"- linkage ok: {summary['linkage_ok']}",
            "",
            "## Sections",
        ]
    )
    for name, section in report["sections"].items():
        lines.append(f"- {name}: payloads={section['payload_count']} missing={len(section['missing_in_all'])} partial={len(section['partial_coverage'])}")
    lines.extend(["", "## Linkage"])
    for key, value in report["linkage"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def load_json_paths(paths) -> list[dict[str, Any]]:
    return [load_json(path) for path in paths if path.exists()]


def load_jsonl_paths(paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(load_jsonl(path))
    return rows


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


def safe_filename(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


if __name__ == "__main__":
    raise SystemExit(main())
