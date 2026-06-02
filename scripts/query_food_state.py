#!/usr/bin/env python3
"""Query recipe, ingredient, nutrition, and anomaly state for a video/time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from food_agent.state_store import FoodStateStore, state_to_json


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--section", choices=["recipe", "ingredient", "nutrition", "anomalies", "all"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = FoodStateStore(args.index_dir)
    if args.section in {"recipe", "all"}:
        print("## recipe")
        print(state_to_json(store.recipe_state(args.video_id, args.time)))
    if args.section in {"ingredient", "all"}:
        print("## ingredient")
        print(state_to_json(store.ingredient_state(args.video_id, args.time)))
    if args.section in {"nutrition", "all"}:
        print("## nutrition")
        print(state_to_json(store.nutrition_delta(args.video_id, args.time)))
    if args.section in {"anomalies", "all"}:
        print("## anomalies")
        print(state_to_json(store.detect_anomalies(args.video_id, args.time)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

