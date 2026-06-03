#!/usr/bin/env python3
"""Export incorrect prediction cases from a comparison run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions_*.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Path to write exported error cases JSONL")
    parser.add_argument("--task-family", default=None, help="Optional task family filter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("correct"):
            continue
        if args.task_family and obj.get("task_family") != args.task_family:
            continue
        rows.append(obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(args.out)
    print(f"errors={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
