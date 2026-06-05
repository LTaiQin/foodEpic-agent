"""Build a per-video graph memory from event indices and observed evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from food_agent.memory import GraphEdgeRecord, GraphMemoryStore, GraphNodeRecord
from food_agent.paths import ProjectPaths


GRAPH_NODE_TYPES = {
    "events.parquet": "segment",
    "audio_events.parquet": "audio_event",
    "ingredients.parquet": "ingredient_event",
    "recipe_steps.parquet": "recipe_step",
    "object_tracks.parquet": "object_track",
}


class VideoGraphBuilder:
    """Build a graph memory store for a single video."""

    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths.from_env()
        self.index_dir = self.paths.output_root / "event_index"
        self.probe_root = self.paths.output_root / "single_video_agent_probe"
        self.graph_root = self.paths.output_root / "graph_memory"

    def build(self, video_id: str) -> GraphMemoryStore:
        store = GraphMemoryStore(self.graph_root / video_id)
        nodes: list[GraphNodeRecord] = []
        edges: list[GraphEdgeRecord] = []
        nodes.append(self._video_node(video_id))
        nodes.extend(self._indexed_event_nodes(video_id))
        edges.extend(self._video_contains_edges(nodes[1:], video_id))
        frame_nodes, frame_edges = self._frame_memory_nodes(video_id)
        nodes.extend(frame_nodes)
        edges.extend(frame_edges)
        store.replace_graph(nodes, edges)
        return store

    def _video_node(self, video_id: str) -> GraphNodeRecord:
        videos_df = pd.read_parquet(self.index_dir / "videos.parquet")
        row = videos_df[videos_df["video_id"] == video_id].iloc[0]
        attrs = {
            "participant_id": self._safe_value(row["participant_id"]),
            "path": self._safe_value(row["path"]),
            "relative_path": self._safe_value(row["relative_path"]),
            "fps": self._safe_value(row["fps"]),
            "frame_count": self._safe_value(row["frame_count"]),
            "width": self._safe_value(row["width"]),
            "height": self._safe_value(row["height"]),
            "duration_sec": self._safe_value(row["duration_sec"]),
        }
        return GraphNodeRecord(
            node_id=f"video:{video_id}",
            node_type="video",
            label=video_id,
            video_id=video_id,
            start_time=0.0,
            end_time=float(row["duration_sec"]),
            attributes=attrs,
            evidence_paths=[str(self._safe_value(row["path"]))],
            keywords=[video_id.lower(), str(self._safe_value(row["participant_id"])).lower()],
        )

    def _indexed_event_nodes(self, video_id: str) -> list[GraphNodeRecord]:
        nodes: list[GraphNodeRecord] = []
        for filename, node_type in GRAPH_NODE_TYPES.items():
            df = pd.read_parquet(self.index_dir / filename)
            if "video_id" not in df.columns:
                continue
            video_df = df[df["video_id"] == video_id].copy()
            for _, row in video_df.iterrows():
                attrs = {key: self._safe_value(row[key]) for key in row.index if key not in {"event_id", "video_id", "start_time", "end_time"}}
                raw_id = str(row["event_id"] if "event_id" in row.index else f"{node_type}:{video_id}:{len(nodes)}")
                node_id = f"{node_type}:{raw_id}"
                label = str(row.get("text") or row.get("label") or raw_id)
                keywords = self._keyword_list([label, attrs.get("label"), attrs.get("text"), attrs.get("object_name")])
                nodes.append(
                    GraphNodeRecord(
                        node_id=node_id,
                        node_type=node_type,
                        label=label,
                        video_id=video_id,
                        start_time=float(row["start_time"]) if pd.notna(row.get("start_time")) else None,
                        end_time=float(row["end_time"]) if pd.notna(row.get("end_time")) else None,
                        attributes=attrs,
                        evidence_paths=[str(row.get("source_file", ""))] if row.get("source_file") else [],
                        keywords=keywords,
                    )
                )
        return nodes

    def _frame_memory_nodes(self, video_id: str) -> tuple[list[GraphNodeRecord], list[GraphEdgeRecord]]:
        video_dir = self.probe_root / video_id
        frame_files = [video_dir / "frame_observations.json", video_dir / "probe_frame_observations.json"]
        frame_rows: list[dict[str, Any]] = []
        for path in frame_files:
            if path.exists():
                frame_rows.extend(json.loads(path.read_text(encoding="utf-8")))
        video_memory_path = video_dir / "video_memory.json"
        nodes: list[GraphNodeRecord] = []
        edges: list[GraphEdgeRecord] = []
        seen: set[str] = set()
        for row in frame_rows:
            frame_id = str(row["frame_id"])
            if frame_id in seen:
                continue
            seen.add(frame_id)
            obs = row.get("observation", {})
            label = str(obs.get("ongoing_action") or obs.get("possible_step") or frame_id)
            keywords = self._keyword_list(
                [
                    label,
                    obs.get("scene_location"),
                    *obs.get("visible_ingredients", []),
                    *obs.get("visible_tools", []),
                    *obs.get("attention_targets", []),
                    obs.get("state_change_hint"),
                ]
            )
            node_id = f"frame:{video_id}:{frame_id}"
            nodes.append(
                GraphNodeRecord(
                    node_id=node_id,
                    node_type="frame",
                    label=label,
                    video_id=video_id,
                    start_time=float(row["time_s"]),
                    end_time=float(row["time_s"]),
                    attributes=self._safe_value({"source": row.get("source"), "observation": obs, "frame_id": frame_id}),
                    evidence_paths=[str(row.get("path", ""))] if row.get("path") else [],
                    keywords=keywords,
                )
            )
            edges.append(
                GraphEdgeRecord(
                    edge_id=f"contains:video:{video_id}:{frame_id}",
                    source_id=f"video:{video_id}",
                    target_id=node_id,
                    edge_type="contains",
                    video_id=video_id,
                    attributes=self._safe_value({"source": row.get("source")}),
                )
            )
        if video_memory_path.exists():
            payload = json.loads(video_memory_path.read_text(encoding="utf-8"))
            for index, event in enumerate(payload.get("timeline_events", [])):
                event_id = f"timeline:{video_id}:{index}"
                label = str(event.get("event") or event_id)
                nodes.append(
                    GraphNodeRecord(
                        node_id=event_id,
                        node_type="timeline_event",
                        label=label,
                        video_id=video_id,
                        start_time=self._parse_hms_to_seconds(event.get("time_hms")),
                        end_time=self._parse_hms_to_seconds(event.get("time_hms")),
                        attributes=self._safe_value(event),
                        evidence_paths=[],
                        keywords=self._keyword_list([label, *event.get("evidence_frames", [])]),
                    )
                )
                edges.append(
                    GraphEdgeRecord(
                        edge_id=f"contains:{event_id}",
                        source_id=f"video:{video_id}",
                        target_id=event_id,
                        edge_type="contains",
                        video_id=video_id,
                        attributes={},
                    )
                )
        return nodes, edges

    def _video_contains_edges(self, nodes: list[GraphNodeRecord], video_id: str) -> list[GraphEdgeRecord]:
        return [
            GraphEdgeRecord(
                edge_id=f"contains:{video_id}:{index}",
                source_id=f"video:{video_id}",
                target_id=node.node_id,
                edge_type="contains",
                video_id=video_id,
                attributes={},
            )
            for index, node in enumerate(nodes)
        ]

    def _safe_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_value(item) for item in value]
        if pd.isna(value):
            return None
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _keyword_list(self, values: list[Any]) -> list[str]:
        tokens: set[str] = set()
        for value in values:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            tokens.add(text)
            for part in text.replace("/", " ").replace(",", " ").split():
                if len(part) >= 2:
                    tokens.add(part)
        return sorted(tokens)

    def _parse_hms_to_seconds(self, value: Any) -> float | None:
        if not value:
            return None
        text = str(value).strip()
        if "-" in text:
            text = text.split("-", 1)[0]
        hours, minutes, seconds = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
