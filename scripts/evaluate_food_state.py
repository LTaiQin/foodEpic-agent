#!/usr/bin/env python3
"""Evaluate deterministic food-state coverage over indexed videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from food_agent.state_store import FoodStateStore


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out", type=Path, default=defaults.output_root / "results" / "food_state_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = FoodStateStore(args.index_dir)
    videos = pd.read_parquet(args.index_dir / "videos.parquet")
    recipe_steps = pd.read_parquet(args.index_dir / "recipe_steps.parquet")
    ingredients = pd.read_parquet(args.index_dir / "ingredients.parquet")
    recipe_videos = set(recipe_steps["video_id"].dropna())
    ingredient_videos = set(ingredients["video_id"].dropna())
    metrics = {
        "video_count": int(len(videos)),
        "recipe_video_count": len(recipe_videos),
        "ingredient_video_count": len(ingredient_videos),
        "recipe_step_event_count": int(len(recipe_steps)),
        "ingredient_event_count": int(len(ingredients)),
        "sample_state_checks": [],
    }
    for video_id in sorted(recipe_videos)[:10]:
        subset = recipe_steps[recipe_steps["video_id"] == video_id].sort_values("start_time")
        if subset.empty:
            continue
        time = float(subset.iloc[0]["start_time"])
        recipe_state = store.recipe_state(video_id, time)
        ingredient_state = store.ingredient_state(video_id, time)
        metrics["sample_state_checks"].append(
            {
                "video_id": video_id,
                "time": time,
                "active_steps": len(recipe_state.active_steps),
                "added_ingredients": len(ingredient_state.added),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

