from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/22liushoulong/agent/hd-epic")
INDEX_PATH = ROOT / "outputs" / "event_index" / "vqa_samples.parquet"
OUT_PATH = ROOT / "outputs" / "reports" / "graph_agent_task_coverage_audit.json"


@dataclass(frozen=True)
class CoverageRow:
    task_family: str
    sample_count: int
    has_direct_structured_inference: bool
    has_temporal_specialized_path: bool
    has_segment_visual_inference: bool
    has_bbox_specialized_path: bool
    has_viewpoint_specialized_path: bool
    has_weight_specialized_path: bool
    has_nutrition_specialized_path: bool
    has_open_query_specialized_path: bool
    likely_uses_generic_rank: bool
    notes: list[str]


def _row_for_family(task_family: str, sample_count: int) -> CoverageRow:
    notes: list[str] = []
    has_direct_structured = task_family in {
        "recipe_recipe_recognition",
        "recipe_multi_recipe_recognition",
        "nutrition_video_nutrition_estimation",
        "ingredient_ingredients_order",
        "ingredient_ingredient_retrieval",
        "ingredient_ingredient_recognition",
        "ingredient_exact_ingredient_recognition",
    }
    has_temporal_specialized = task_family in {
        "fine_grained_action_localization",
        "ingredient_ingredient_adding_localization",
        "recipe_multi_step_localization",
        "recipe_prep_localization",
        "recipe_rough_step_localization",
        "recipe_step_localization",
    }
    has_segment_visual = task_family in {
        "gaze_interaction_anticipation",
        "fine_grained_action_recognition",
        "recipe_step_recognition",
        "fine_grained_how_recognition",
        "fine_grained_why_recognition",
        "recipe_following_activity_recognition",
    }
    has_bbox_specialized = task_family in {
        "object_motion_object_movement_itinerary",
        "object_motion_object_movement_counting",
        "object_motion_stationary_object_localization",
        "3d_perception_object_location",
        "3d_perception_object_contents_retrieval",
    }
    has_viewpoint_specialized = task_family in {
        "3d_perception_fixture_location",
        "gaze_gaze_estimation",
        "3d_perception_fixture_interaction_counting",
    }
    has_weight_specialized = task_family == "ingredient_ingredient_weight"
    has_nutrition_specialized = task_family in {
        "nutrition_nutrition_change",
        "nutrition_image_nutrition_estimation",
    }
    has_open_query_specialized = task_family.startswith("open_query")

    likely_uses_generic_rank = not any(
        (
            has_direct_structured,
            has_temporal_specialized,
            has_segment_visual,
            has_bbox_specialized,
            has_viewpoint_specialized,
            has_weight_specialized,
            has_nutrition_specialized,
            has_open_query_specialized,
        )
    )

    if has_direct_structured:
        notes.append("has_direct_structured_inference")
    if has_temporal_specialized:
        notes.append("has_temporal_specialized_path")
    if has_segment_visual:
        notes.append("has_segment_visual_inference")
    if has_bbox_specialized:
        notes.append("has_bbox_specialized_path")
    if has_viewpoint_specialized:
        notes.append("has_viewpoint_specialized_path")
    if has_weight_specialized:
        notes.append("has_weight_specialized_path")
    if has_nutrition_specialized:
        notes.append("has_nutrition_specialized_path")
    if has_open_query_specialized:
        notes.append("has_open_query_specialized_path")
    if likely_uses_generic_rank:
        notes.append("likely_generic_rank_residual")

    return CoverageRow(
        task_family=task_family,
        sample_count=sample_count,
        has_direct_structured_inference=has_direct_structured,
        has_temporal_specialized_path=has_temporal_specialized,
        has_segment_visual_inference=has_segment_visual,
        has_bbox_specialized_path=has_bbox_specialized,
        has_viewpoint_specialized_path=has_viewpoint_specialized,
        has_weight_specialized_path=has_weight_specialized,
        has_nutrition_specialized_path=has_nutrition_specialized,
        has_open_query_specialized_path=has_open_query_specialized,
        likely_uses_generic_rank=likely_uses_generic_rank,
        notes=notes,
    )


def main() -> int:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"missing index file: {INDEX_PATH}")
    df = pd.read_parquet(INDEX_PATH)
    counts = df["task_family"].value_counts().sort_index()
    rows = [_row_for_family(str(task_family), int(sample_count)) for task_family, sample_count in counts.items()]
    summary = {
        "task_family_count": len(rows),
        "likely_generic_rank_residual_count": sum(1 for row in rows if row.likely_uses_generic_rank),
        "likely_generic_rank_residuals": [row.task_family for row in rows if row.likely_uses_generic_rank],
    }
    payload: dict[str, Any] = {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
