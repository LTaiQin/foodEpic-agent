#!/usr/bin/env python3
"""Query the local Parquet event index around a video/time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_path = args.index_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"missing event index: {events_path}")
    events = pd.read_parquet(events_path)
    subset = events[events["video_id"] == args.video_id].copy()
    if args.time is not None:
        start = args.time - args.window
        end = args.time + args.window
        subset = subset[
            (subset["start_time"].fillna(float("inf")) <= end)
            & (subset["end_time"].fillna(float("-inf")) >= start)
        ]
    subset = subset.sort_values(["start_time", "end_time"], na_position="last").head(args.limit)
    columns = ["event_id", "event_type", "start_time", "end_time", "label", "text"]
    print(subset[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

