#!/usr/bin/env python3
"""Query spatial, gaze, and audio context for a video/time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from food_agent.spatial_store import SpatialContextStore, context_to_json


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--object-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SpatialContextStore(args.index_dir)
    context = store.combined_context(args.video_id, time=args.time, object_name=args.object_name)
    print(context_to_json(context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

