#!/usr/bin/env python3
"""Run the graph agent in open-query mode on one video and persist the trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--inputs-json", default="{}")
    parser.add_argument("--task-family", default="open_query")
    parser.add_argument("--query-id", default="")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--reset-session", action="store_true")
    parser.add_argument("--rebuild-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)
    if args.rebuild_graph:
        agent.rebuild_video_graph(args.video_id)
    if args.reset_session:
        agent.reset_video_session(args.video_id)
    result = agent.answer_open_query(
        video_id=args.video_id,
        question=args.question,
        inputs_json=args.inputs_json,
        task_family=args.task_family,
        max_steps=args.max_steps,
        query_id=args.query_id,
    )
    payload = result.to_dict(
        include_row={
            "question": args.question,
            "choices_json": json.dumps(["OPEN_ENDED_RESPONSE"], ensure_ascii=False),
            "inputs_json": args.inputs_json,
        }
    )
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
