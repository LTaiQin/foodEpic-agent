#!/usr/bin/env python3
"""Build graph memory for one HD-EPIC video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.graph import VideoGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = VideoGraphBuilder()
    store = builder.build(args.video_id)
    print(store.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
