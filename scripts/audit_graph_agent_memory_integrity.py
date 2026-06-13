#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/22liushoulong/agent/hd-epic")
OUTPUT_ROOT = ROOT / "outputs"
REPORT_PATH = OUTPUT_ROOT / "reports" / "graph_agent_memory_integrity_audit.json"

WRITEBACK_NODE_PREFIXES = (
    "observation:",
    "frame_observation:",
    "region_observation:",
    "ocr_reading:",
    "audio_writeback:",
    "timeline_summary:",
    "state_change:",
)
WRITEBACK_NODE_TYPES = {"observation", "region", "ocr_reading", "timeline_event", "state_change", "audio_event"}
LEAK_KEYS = {"gold", "correct", "correct_idx", "answer_idx", "label_idx"}
REQUIRED_ATTR_KEYS_BY_TYPE = {
    "ocr_reading": {"reading", "source_tool", "confidence"},
    "timeline_event": {"summary", "source_tool", "confidence"},
    "state_change": {"target", "source_tool", "confidence"},
    "region": {"source_tool", "confidence"},
    "observation": {"source_tool", "confidence"},
}


def main() -> int:
    graph_stats = audit_graph_memory()
    session_stats = audit_session_outputs()
    run_stats = audit_run_outputs()
    payload = {
        "graph_memory": graph_stats,
        "session_outputs": session_stats,
        "run_outputs": run_stats,
        "summary": build_summary(graph_stats, session_stats, run_stats),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"report_path={REPORT_PATH}")
    return 0


def audit_graph_memory() -> dict[str, Any]:
    graph_root = OUTPUT_ROOT / "graph_memory"
    node_type_counts = Counter()
    missing_required = defaultdict(Counter)
    leak_hits = Counter()
    writeback_nodes = 0
    evidence_missing = 0
    scanned_files = 0
    videos = set()
    for nodes_path in graph_root.glob("*/nodes.jsonl"):
        scanned_files += 1
        videos.add(nodes_path.parent.name)
        for line in nodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            node = json.loads(line)
            node_type = str(node.get("node_type") or "")
            node_type_counts[node_type] += 1
            attrs = node.get("attributes") or {}
            evidence_paths = node.get("evidence_paths") or []
            node_id = str(node.get("node_id") or "")
            if node_type in WRITEBACK_NODE_TYPES and node_id.startswith(WRITEBACK_NODE_PREFIXES):
                writeback_nodes += 1
                required = REQUIRED_ATTR_KEYS_BY_TYPE.get(node_type, set())
                for key in required:
                    if key not in attrs:
                        missing_required[node_type][key] += 1
                if node_type in {"region", "ocr_reading", "observation", "timeline_event"} and not evidence_paths:
                    evidence_missing += 1
            for key in LEAK_KEYS:
                if key in node:
                    leak_hits[f"node_field:{key}"] += 1
                if isinstance(attrs, dict) and key in attrs:
                    leak_hits[f"attr_field:{key}"] += 1
    return {
        "video_count": len(videos),
        "scanned_node_files": scanned_files,
        "node_type_counts": dict(node_type_counts),
        "writeback_node_count": writeback_nodes,
        "writeback_missing_evidence_count": evidence_missing,
        "missing_required_fields": {node_type: dict(counter) for node_type, counter in missing_required.items()},
        "leak_hits": dict(leak_hits),
    }


def audit_session_outputs() -> dict[str, Any]:
    session_root = OUTPUT_ROOT / "graph_agent_sessions"
    session_count = 0
    trace_rows = 0
    leak_hits = Counter()
    session_memory_keys = Counter()
    for session_dir in session_root.iterdir() if session_root.exists() else []:
        if not session_dir.is_dir():
            continue
        state_path = session_dir / "session_state.json"
        trace_path = session_dir / "session_trace.jsonl"
        if state_path.exists():
            session_count += 1
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            for key in payload.keys():
                session_memory_keys[key] += 1
            memory = payload.get("session_memory") or {}
            if isinstance(memory, dict):
                for key in LEAK_KEYS:
                    if key in memory:
                        leak_hits[f"session_memory:{key}"] += 1
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                trace_rows += 1
                row = json.loads(line)
                for key in LEAK_KEYS:
                    if key in row:
                        leak_hits[f"session_trace:{key}"] += 1
    return {
        "session_count": session_count,
        "trace_row_count": trace_rows,
        "session_state_top_keys": dict(session_memory_keys),
        "leak_hits": dict(leak_hits),
    }


def audit_run_outputs() -> dict[str, Any]:
    run_root = OUTPUT_ROOT / "graph_agent_runs"
    json_count = 0
    markdown_count = 0
    leak_hits = Counter()
    for path in run_root.rglob("*"):
        if path.suffix == ".json":
            json_count += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in LEAK_KEYS:
                if key in payload:
                    leak_hits[f"run_json:{key}"] += 1
        elif path.suffix == ".md":
            markdown_count += 1
            text = path.read_text(encoding="utf-8")
            if "- gold:" in text:
                leak_hits["run_md:gold"] += 1
            if "- correct:" in text:
                leak_hits["run_md:correct"] += 1
    return {
        "json_file_count": json_count,
        "markdown_file_count": markdown_count,
        "leak_hits": dict(leak_hits),
    }


def build_summary(graph_stats: dict[str, Any], session_stats: dict[str, Any], run_stats: dict[str, Any]) -> dict[str, Any]:
    total_leak_hits = (
        sum(int(v) for v in graph_stats.get("leak_hits", {}).values())
        + sum(int(v) for v in session_stats.get("leak_hits", {}).values())
        + sum(int(v) for v in run_stats.get("leak_hits", {}).values())
    )
    total_missing_required = sum(
        int(value)
        for counter in graph_stats.get("missing_required_fields", {}).values()
        for value in counter.values()
    )
    return {
        "graph_writeback_node_count": graph_stats.get("writeback_node_count", 0),
        "graph_writeback_missing_evidence_count": graph_stats.get("writeback_missing_evidence_count", 0),
        "graph_missing_required_field_count": total_missing_required,
        "session_trace_leak_hit_count": sum(int(v) for v in session_stats.get("leak_hits", {}).values()),
        "run_output_leak_hit_count": sum(int(v) for v in run_stats.get("leak_hits", {}).values()),
        "total_leak_hit_count": total_leak_hits,
    }


if __name__ == "__main__":
    raise SystemExit(main())
