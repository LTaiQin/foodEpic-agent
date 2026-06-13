#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.memory import GraphMemoryStore, GraphNodeRecord


ROOT = Path("/22liushoulong/agent/hd-epic/outputs/graph_memory")
WRITEBACK_NODE_PREFIXES = (
    "observation:",
    "frame_observation:",
    "region_observation:",
    "ocr_reading:",
    "audio_writeback:",
    "timeline_summary:",
    "state_change:",
)


def main() -> int:
    updated = 0
    for video_dir in ROOT.iterdir() if ROOT.exists() else []:
        if not video_dir.is_dir():
            continue
        store = GraphMemoryStore(video_dir)
        nodes_path = video_dir / "nodes.jsonl"
        if not nodes_path.exists():
            continue
        for line in nodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            node = json.loads(line)
            node_id = str(node.get("node_id") or "")
            if not node_id.startswith(WRITEBACK_NODE_PREFIXES):
                continue
            attrs = dict(node.get("attributes") or {})
            changed = False
            if "source_tool" not in attrs:
                attrs["source_tool"] = "agent_writeback"
                changed = True
            if "confidence" not in attrs:
                attrs["confidence"] = 0.0
                changed = True
            if not changed:
                continue
            store.upsert_node(
                GraphNodeRecord(
                    node_id=node_id,
                    node_type=str(node.get("node_type") or ""),
                    label=str(node.get("label") or ""),
                    video_id=str(node.get("video_id") or ""),
                    start_time=node.get("start_time"),
                    end_time=node.get("end_time"),
                    attributes=attrs,
                    evidence_paths=list(node.get("evidence_paths") or []),
                    keywords=list(node.get("keywords") or []),
                )
            )
            updated += 1
    print(json.dumps({"updated_nodes": updated}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
