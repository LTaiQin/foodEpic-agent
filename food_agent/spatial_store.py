"""Spatial / gaze / audio context queries over the event index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ProjectPaths


@dataclass(frozen=True)
class SpatialContext:
    video_id: str
    time: float | None
    object_tracks: list[dict[str, Any]]
    object_masks: list[dict[str, Any]]
    gaze_priming: list[dict[str, Any]]
    audio_events: list[dict[str, Any]]


class SpatialContextStore:
    """Simple retrieval layer for object, gaze, and audio evidence."""

    def __init__(self, index_dir: Path | None = None):
        paths = ProjectPaths.from_env()
        self.index_dir = index_dir or paths.output_root / "event_index"
        self._cache: dict[str, pd.DataFrame] = {}

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.index_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing index table: {path}")
            self._cache[name] = pd.read_parquet(path)
        return self._cache[name]

    def _video_fps(self, video_id: str) -> float | None:
        videos = self._table("videos")
        row = videos[videos["video_id"] == video_id].head(1)
        if row.empty:
            return None
        fps = row.iloc[0].get("fps")
        try:
            return float(fps) if fps is not None else None
        except (TypeError, ValueError):
            return None

    def object_context(
        self,
        video_id: str,
        time: float | None = None,
        object_name: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        tracks = self._table("object_tracks")
        masks = self._table("object_masks")
        track_subset = tracks[tracks["video_id"] == video_id].copy()
        if object_name:
            track_subset = track_subset[track_subset["object_name"].str.contains(object_name, case=False, na=False)]
        if time is not None:
            track_subset = track_subset[
                (track_subset["start_time"].fillna(float("inf")) <= float(time))
                & (track_subset["end_time"].fillna(float("-inf")) >= float(time))
            ]
        track_subset = track_subset.sort_values(["start_time", "end_time"]).head(limit)
        mask_subset = masks[masks["video_id"] == video_id].copy()
        if time is not None and not mask_subset.empty:
            fps = self._video_fps(video_id)
            if fps:
                target_frame = float(time) * fps
                mask_subset["frame_distance"] = mask_subset["frame_number"].apply(lambda value: abs(float(value) - target_frame))
                mask_subset = mask_subset.sort_values(["frame_distance", "frame_number"], na_position="last")
            mask_subset = mask_subset.head(limit)
        return {
            "video_id": video_id,
            "time": time,
            "object_tracks": _records(track_subset),
            "object_masks": _records(mask_subset.head(limit)),
        }

    def gaze_context(self, video_id: str, time: float | None = None, limit: int = 20) -> dict[str, Any]:
        gaze = self._table("gaze_priming")
        subset = gaze[gaze["video_id"] == video_id].copy()
        fps = self._video_fps(video_id)
        if fps:
            subset["frame_time"] = subset["frame"].astype(float) / fps
            subset["frame_primed_time"] = subset["frame_primed"].apply(
                lambda value: float(value) / fps if pd.notna(value) and float(value) >= 0 else None
            )
        if time is not None:
            if fps and not subset.empty:
                subset["time_distance"] = subset["frame_time"].apply(lambda value: abs(float(value) - float(time)))
                subset = subset.sort_values(["time_distance", "frame_time", "frame_primed_time"], na_position="last")
            else:
                subset = subset.head(limit)
        return {
            "video_id": video_id,
            "time": time,
            "gaze_priming": _records(subset.head(limit)),
        }

    def audio_context(self, video_id: str, start_time: float, end_time: float, limit: int = 20) -> dict[str, Any]:
        audio = self._table("audio_events")
        subset = audio[
            (audio["video_id"] == video_id)
            & (audio["end_time"].fillna(float("inf")) >= float(start_time))
            & (audio["start_time"].fillna(float("-inf")) <= float(end_time))
        ]
        subset = subset.sort_values(["start_time", "end_time"]).head(limit)
        return {
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "audio_events": _records(subset),
        }

    def combined_context(
        self,
        video_id: str,
        time: float | None = None,
        object_name: str | None = None,
        audio_window: float = 5.0,
        limit: int = 20,
    ) -> SpatialContext:
        object_ctx = self.object_context(video_id, time=time, object_name=object_name, limit=limit)
        gaze_ctx = self.gaze_context(video_id, time=time, limit=limit)
        if time is None:
            audio_ctx = {"audio_events": []}
        else:
            audio_ctx = self.audio_context(video_id, max(0.0, time - audio_window), time + audio_window, limit=limit)
        return SpatialContext(
            video_id=video_id,
            time=time,
            object_tracks=object_ctx["object_tracks"],
            object_masks=object_ctx["object_masks"],
            gaze_priming=gaze_ctx["gaze_priming"],
            audio_events=audio_ctx["audio_events"],
        )

    def resolve_object_reference(
        self,
        video_id: str,
        bbox: list[float],
        time: float | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        masks = self._table("object_masks")
        subset = masks[masks["video_id"] == video_id].copy()
        fps = self._video_fps(video_id)
        if time is not None and fps and not subset.empty:
            target_frame = float(time) * fps
            subset["frame_distance"] = subset["frame_number"].apply(lambda value: abs(float(value) - target_frame))
            subset = subset.sort_values(["frame_distance", "frame_number"])
            subset = subset.head(max(limit * 20, 20))
        scored: list[dict[str, Any]] = []
        target_bbox = [float(value) for value in bbox]
        for row in _records(subset):
            row_bbox = json.loads(row.get("bbox_json") or "[]")
            if len(row_bbox) != 4:
                continue
            iou = _best_bbox_iou(target_bbox, [float(value) for value in row_bbox])
            if iou <= 0:
                continue
            enriched = dict(row)
            enriched["iou"] = iou
            scored.append(enriched)
        scored.sort(
            key=lambda row: (
                float(row.get("iou") or 0.0),
                -float(row.get("frame_distance") or 0.0),
            ),
            reverse=True,
        )
        return scored[:limit]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_variants(bbox: list[float]) -> list[list[float]]:
    if len(bbox) != 4:
        return [bbox]
    x1, y1, x2, y2 = bbox
    return [
        [x1, y1, x2, y2],
        [y1, x1, y2, x2],
    ]


def _best_bbox_iou(a: list[float], b: list[float]) -> float:
    best = 0.0
    for a_variant in _bbox_variants(a):
        for b_variant in _bbox_variants(b):
            best = max(best, _bbox_iou(a_variant, b_variant))
    return best


def context_to_json(context: SpatialContext | dict[str, Any]) -> str:
    if hasattr(context, "__dataclass_fields__"):
        return json.dumps(asdict(context), ensure_ascii=False, indent=2)
    return json.dumps(context, ensure_ascii=False, indent=2)
