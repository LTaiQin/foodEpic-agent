from pathlib import Path

import pandas as pd

from food_agent.spatial_store import SpatialContextStore


def test_object_context() -> None:
    store = SpatialContextStore()
    ctx = store.object_context("P01-20240202-110250", time=360.0)
    assert ctx["video_id"] == "P01-20240202-110250"
    assert isinstance(ctx["object_tracks"], list)


def test_combined_context() -> None:
    store = SpatialContextStore()
    ctx = store.combined_context("P01-20240202-110250", time=360.0)
    assert ctx.video_id == "P01-20240202-110250"
    assert isinstance(ctx.object_tracks, list)
    assert isinstance(ctx.audio_events, list)


def test_gaze_context_sorts_by_time_distance() -> None:
    store = SpatialContextStore(index_dir=Path("/tmp/unused"))
    store._cache["videos"] = pd.DataFrame(
        [{"video_id": "vid", "fps": 10.0}]
    )
    store._cache["gaze_priming"] = pd.DataFrame(
        [
            {"video_id": "vid", "frame": 10, "frame_primed": 5},
            {"video_id": "vid", "frame": 30, "frame_primed": 20},
            {"video_id": "vid", "frame": 21, "frame_primed": 18},
        ]
    )
    ctx = store.gaze_context("vid", time=2.0, limit=3)
    frames = [row["frame"] for row in ctx["gaze_priming"]]
    assert frames[0] == 21


def test_resolve_object_reference_prefers_high_iou() -> None:
    store = SpatialContextStore(index_dir=Path("/tmp/unused"))
    store._cache["videos"] = pd.DataFrame(
        [{"video_id": "vid", "fps": 10.0}]
    )
    store._cache["object_masks"] = pd.DataFrame(
        [
            {"video_id": "vid", "frame_number": 20, "bbox_json": "[0, 0, 10, 10]"},
            {"video_id": "vid", "frame_number": 21, "bbox_json": "[1, 1, 9, 9]"},
            {"video_id": "vid", "frame_number": 22, "bbox_json": "[20, 20, 30, 30]"},
        ]
    )
    matches = store.resolve_object_reference("vid", [1, 1, 9, 9], time=2.0, limit=2)
    assert len(matches) == 2
    assert matches[0]["frame_number"] == 21
    assert matches[0]["iou"] >= matches[1]["iou"]


def test_resolve_object_reference_handles_swapped_bbox_axes() -> None:
    store = SpatialContextStore(index_dir=Path("/tmp/unused"))
    store._cache["videos"] = pd.DataFrame(
        [{"video_id": "vid", "fps": 10.0}]
    )
    store._cache["object_masks"] = pd.DataFrame(
        [
            {"video_id": "vid", "frame_number": 20, "bbox_json": "[734.085, 787.036, 791.85, 972.362]"},
        ]
    )
    matches = store.resolve_object_reference("vid", [787.036, 734.085, 972.362, 791.85], time=2.0, limit=1)
    assert len(matches) == 1
    assert matches[0]["iou"] > 0.9
