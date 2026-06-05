"""LightAgent-compatible tools backed by the local event index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ProjectPaths


class HDEpicToolset:
    """Factory for HD-EPIC query tools with LightAgent tool_info metadata."""

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

    def tools(self) -> list:
        return [
            _make_tool("get_video_metadata", self.get_video_metadata),
            _make_tool("retrieve_events", self.retrieve_events),
            _make_tool("get_recipe_state", self.get_recipe_state),
            _make_tool("get_ingredient_state", self.get_ingredient_state),
            _make_tool("get_object_state", self.get_object_state),
            _make_tool("resolve_object_reference", self.resolve_object_reference),
            _make_tool("get_gaze_hand_context", self.get_gaze_hand_context),
            _make_tool("get_audio_events", self.get_audio_events),
        ]

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

    def get_video_metadata(self, video_id: str) -> str:
        videos = self._table("videos")
        row = videos[videos["video_id"] == video_id].head(1)
        if row.empty:
            return json.dumps({"status": "not_found", "video_id": video_id}, ensure_ascii=False)
        return row.iloc[0].to_json(force_ascii=False)

    def retrieve_events(
        self,
        video_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        event_types: list[str] | None = None,
        limit: int = 20,
    ) -> str:
        events = self._table("events")
        subset = events[events["video_id"] == video_id].copy()
        if start_time is not None:
            subset = subset[subset["end_time"].fillna(float("inf")) >= float(start_time)]
        if end_time is not None:
            subset = subset[subset["start_time"].fillna(float("-inf")) <= float(end_time)]
        if event_types:
            subset = subset[subset["event_type"].isin(event_types)]
        subset = subset.sort_values(["start_time", "end_time"], na_position="last").head(limit)
        return subset.to_json(orient="records", force_ascii=False)

    def get_recipe_state(self, video_id: str, time: float) -> str:
        events = self._table("recipe_steps")
        subset = events[
            (events["video_id"] == video_id)
            & (events["start_time"].fillna(float("inf")) <= float(time))
            & (events["end_time"].fillna(float("-inf")) >= float(time))
        ].copy()
        return subset.sort_values(["start_time", "end_time"]).to_json(orient="records", force_ascii=False)

    def get_ingredient_state(self, video_id: str, time: float) -> str:
        events = self._table("ingredients")
        subset = events[(events["video_id"] == video_id) & (events["start_time"].fillna(float("inf")) <= float(time))]
        return subset.sort_values(["start_time", "end_time"]).to_json(orient="records", force_ascii=False)

    def get_object_state(self, video_id: str, object_name: str | None = None, time: float | None = None) -> str:
        tracks = self._table("object_tracks")
        subset = tracks[tracks["video_id"] == video_id].copy()
        if object_name:
            subset = subset[subset["object_name"].str.contains(object_name, case=False, na=False)]
        if time is not None:
            subset = subset[
                (subset["start_time"].fillna(float("inf")) <= float(time))
                & (subset["end_time"].fillna(float("-inf")) >= float(time))
            ]
        return subset.head(20).to_json(orient="records", force_ascii=False)

    def resolve_object_reference(self, video_id: str, bbox: list[float], time: float | None = None, limit: int = 5) -> str:
        masks = self._table("object_masks")
        subset = masks[masks["video_id"] == video_id].copy()
        fps = self._video_fps(video_id)
        if time is not None and fps and not subset.empty:
            target_frame = float(time) * fps
            subset["frame_distance"] = subset["frame_number"].apply(lambda value: abs(float(value) - target_frame))
            subset = subset.sort_values(["frame_distance", "frame_number"]).head(max(limit * 20, 20))
        rows = []
        target_bbox = [float(value) for value in bbox]
        for row in subset.to_dict(orient="records"):
            row_bbox = json.loads(row.get("bbox_json") or "[]")
            iou = _best_bbox_iou(target_bbox, [float(value) for value in row_bbox]) if len(row_bbox) == 4 else 0.0
            if iou <= 0:
                continue
            row["iou"] = iou
            rows.append(row)
        rows.sort(key=lambda row: row.get("iou", 0.0), reverse=True)
        return json.dumps(rows[:limit], ensure_ascii=False)

    def get_gaze_hand_context(self, video_id: str, time: float | None = None) -> str:
        gaze = self._table("gaze_priming")
        subset = gaze[gaze["video_id"] == video_id].copy()
        fps = self._video_fps(video_id)
        if time is not None and fps and not subset.empty:
            subset["frame_time"] = subset["frame"].astype(float) / fps
            subset["time_distance"] = subset["frame_time"].apply(lambda value: abs(float(value) - float(time)))
            subset = subset.sort_values(["time_distance", "frame_time"], na_position="last")
        return subset.head(20).to_json(orient="records", force_ascii=False)

    def get_audio_events(self, video_id: str, start_time: float, end_time: float, limit: int = 20) -> str:
        audio = self._table("audio_events")
        subset = audio[
            (audio["video_id"] == video_id)
            & (audio["end_time"].fillna(float("inf")) >= float(start_time))
            & (audio["start_time"].fillna(float("-inf")) <= float(end_time))
        ]
        return subset.sort_values(["start_time", "end_time"]).head(limit).to_json(orient="records", force_ascii=False)


def _make_tool(name: str, call: Any) -> Any:
    def tool(**kwargs: Any) -> Any:
        return call(**kwargs)

    tool.__name__ = name
    tool.tool_info = _tool_info(name)
    return tool


def _tool_info(name: str) -> dict[str, Any]:
    infos = {
        "get_video_metadata": (
            "Return metadata for a HD-EPIC video.",
            [{"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True}],
        ),
        "retrieve_events": (
            "Retrieve structured events for a video and optional time range.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "start_time", "description": "Start time in seconds.", "type": "number", "required": False},
                {"name": "end_time", "description": "End time in seconds.", "type": "number", "required": False},
                {
                    "name": "event_types",
                    "description": "Optional event type names.",
                    "type": "array",
                    "items": {"type": "string"},
                    "required": False,
                },
                {"name": "limit", "description": "Maximum number of events.", "type": "integer", "required": False},
            ],
        ),
        "get_recipe_state": (
            "Return active recipe step events at a timestamp.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "time", "description": "Time in seconds.", "type": "number", "required": True},
            ],
        ),
        "get_ingredient_state": (
            "Return ingredient events observed up to a timestamp.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "time", "description": "Time in seconds.", "type": "number", "required": True},
            ],
        ),
        "get_object_state": (
            "Return object tracks for a video.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "object_name", "description": "Optional object name filter.", "type": "string", "required": False},
                {"name": "time", "description": "Optional time in seconds.", "type": "number", "required": False},
            ],
        ),
        "resolve_object_reference": (
            "Resolve a bbox/time object reference to matching object-mask candidates.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {
                    "name": "bbox",
                    "description": "Bounding box [x1, y1, x2, y2].",
                    "type": "array",
                    "items": {"type": "number"},
                    "required": True,
                },
                {"name": "time", "description": "Optional time in seconds.", "type": "number", "required": False},
                {"name": "limit", "description": "Maximum number of matches.", "type": "integer", "required": False},
            ],
        ),
        "get_gaze_hand_context": (
            "Return gaze priming context for a video.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "time", "description": "Optional time in seconds.", "type": "number", "required": False},
            ],
        ),
        "get_audio_events": (
            "Return audio events for a video and time range.",
            [
                {"name": "video_id", "description": "HD-EPIC video id.", "type": "string", "required": True},
                {"name": "start_time", "description": "Start time in seconds.", "type": "number", "required": True},
                {"name": "end_time", "description": "End time in seconds.", "type": "number", "required": True},
                {"name": "limit", "description": "Maximum number of events.", "type": "integer", "required": False},
            ],
        ),
    }
    description, params = infos[name]
    return {"tool_name": name, "tool_description": description, "tool_params": params}


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
