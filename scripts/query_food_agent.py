#!/usr/bin/env python3
"""Unified food agent demo query over state, spatial, audio, and event evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from food_agent.spatial_store import SpatialContextStore
from food_agent.state_store import FoodStateStore
from food_agent.task_router import FoodTaskRouter


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--object-name", default=None)
    return parser.parse_args()


def summarize_answer(question: str, route: str, recipe, ingredient, nutrition, anomalies, spatial) -> str:
    if route == "ingredient":
        if ingredient.added:
            names = ", ".join(row.get("label", "unknown") for row in ingredient.added[-3:])
            return f"已观察到加入的食材包括：{names}。"
        return "当前时间点前没有检索到已加入食材。"
    if route == "nutrition":
        return f"当前可累计营养为 {nutrition.totals}；未知营养字段数为 {nutrition.unknown_count}。"
    if route == "recipe":
        if recipe.active_steps:
            return f"当前步骤：{recipe.active_steps[0].get('text')}"
        if recipe.next_steps:
            return f"当前无活跃步骤；下一步可能是：{recipe.next_steps[0].get('text')}"
        return "当前未检索到 recipe step。"
    if route == "object":
        if spatial.object_tracks:
            obj = spatial.object_tracks[0]
            return f"检索到物体轨迹：{obj.get('object_name')}，时间段 {obj.get('start_time')} 到 {obj.get('end_time')}。"
        return "当前未检索到匹配物体轨迹。"
    if route == "audio":
        if spatial.audio_events:
            labels = ", ".join(row.get("label", "unknown") for row in spatial.audio_events[:5])
            return f"附近音频事件包括：{labels}。"
        return "附近未检索到音频事件。"
    if anomalies:
        return f"检测到 {len(anomalies)} 个潜在异常。"
    return "已检索 recipe、ingredient、spatial 和 audio 证据，请查看 evidence 字段。"


def main() -> int:
    args = parse_args()
    router = FoodTaskRouter()
    route = router.route(args.question)
    state_store = FoodStateStore(args.index_dir)
    spatial_store = SpatialContextStore(args.index_dir)
    recipe = state_store.recipe_state(args.video_id, args.time)
    ingredient = state_store.ingredient_state(args.video_id, args.time)
    nutrition = state_store.nutrition_delta(args.video_id, args.time)
    anomalies = state_store.detect_anomalies(args.video_id, args.time)
    spatial = spatial_store.combined_context(args.video_id, time=args.time, object_name=args.object_name)
    evidence_ids = []
    for row in recipe.active_steps + recipe.completed_steps[-3:] + ingredient.added[-3:] + spatial.audio_events[:3]:
        event_id = row.get("event_id")
        if event_id:
            evidence_ids.append(event_id)
    result = {
        "question": args.question,
        "video_id": args.video_id,
        "time": args.time,
        "task_family": route.task_family,
        "answer": summarize_answer(args.question, route.task_family, recipe, ingredient, nutrition, anomalies, spatial),
        "evidence_ids": evidence_ids,
        "recipe_state": asdict(recipe),
        "ingredient_state": asdict(ingredient),
        "nutrition_delta": asdict(nutrition),
        "anomalies": [asdict(item) for item in anomalies],
        "spatial_context": asdict(spatial),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

