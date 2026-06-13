#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/22liushoulong/agent/hd-epic")
OUTPUT_ROOT = ROOT / "outputs"
LEAK_KEYS = ("gold", "correct")


def main() -> int:
    session_trace_updates = sanitize_session_traces()
    run_json_updates = sanitize_run_json()
    run_md_updates = sanitize_run_markdown()
    print(
        json.dumps(
            {
                "session_trace_rows_sanitized": session_trace_updates,
                "run_json_files_sanitized": run_json_updates,
                "run_markdown_files_sanitized": run_md_updates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def sanitize_session_traces() -> int:
    updated_rows = 0
    for path in (OUTPUT_ROOT / "graph_agent_sessions").glob("*/session_trace.jsonl"):
        rows = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in LEAK_KEYS:
                if key in row:
                    row.pop(key, None)
                    changed = True
                    updated_rows += 1
            rows.append(row)
        if changed:
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return updated_rows


def sanitize_run_json() -> int:
    updated_files = 0
    for path in (OUTPUT_ROOT / "graph_agent_runs").rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        if isinstance(payload, dict):
            for key in LEAK_KEYS:
                if key in payload:
                    payload.pop(key, None)
                    changed = True
        if changed:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            updated_files += 1
    return updated_files


def sanitize_run_markdown() -> int:
    updated_files = 0
    for path in (OUTPUT_ROOT / "graph_agent_runs").rglob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        filtered = [line for line in lines if not line.startswith("- gold:") and not line.startswith("- correct:")]
        if filtered != lines:
            path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
            updated_files += 1
    return updated_files


if __name__ == "__main__":
    raise SystemExit(main())
