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
        if time is not None:
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


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def context_to_json(context: SpatialContext | dict[str, Any]) -> str:
    if hasattr(context, "__dataclass_fields__"):
        return json.dumps(asdict(context), ensure_ascii=False, indent=2)
    return json.dumps(context, ensure_ascii=False, indent=2)

