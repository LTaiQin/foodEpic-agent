"""Shared artifact reuse policy for planner and session recovery."""

from __future__ import annotations


def artifact_reuse_prefixes_for_task(task_family: str) -> tuple[str, ...]:
    normalized = str(task_family or "").strip()
    prefixes: list[str] = [normalized]
    if normalized in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
        prefixes.append("view")
    if normalized in {
        "gaze_interaction_anticipation",
        "fine_grained_action_recognition",
        "fine_grained_how_recognition",
        "fine_grained_why_recognition",
        "recipe_step_recognition",
    }:
        prefixes.append("segment")
    if normalized == "ingredient_ingredient_weight":
        prefixes.extend(["range", "ocr"])
    if normalized in {
        "3d_perception_object_location",
        "3d_perception_object_contents_retrieval",
        "object_motion_object_movement_itinerary",
        "object_motion_object_movement_counting",
        "object_motion_stationary_object_localization",
    }:
        prefixes.extend(["anchor", "bbox", "crop", "contents"])
    if normalized.startswith("open_query"):
        prefixes.append("open_query")
    deduped: list[str] = []
    for item in prefixes:
        item = str(item).strip()
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped)
