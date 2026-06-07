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
        self.graph_root = self.paths.graph_memory_root

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
        edges.extend(self._relation_edges(nodes, video_id))
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

    def _relation_edges(self, nodes: list[GraphNodeRecord], video_id: str) -> list[GraphEdgeRecord]:
        edges: dict[str, GraphEdgeRecord] = {}
        temporal_nodes = [
            node
            for node in nodes
            if node.node_type != "video" and node.start_time is not None
        ]
        temporal_nodes.sort(key=lambda item: (float(item.start_time or 0.0), float(item.end_time or item.start_time or 0.0), item.node_id))
        for index, node in enumerate(temporal_nodes):
            if index > 0:
                previous = temporal_nodes[index - 1]
                self._add_temporal_pair(edges, previous=previous, current=node, video_id=video_id)
            overlap_budget = 0
            for next_index in range(index + 1, min(len(temporal_nodes), index + 7)):
                candidate = temporal_nodes[next_index]
                if self._temporal_gap(node, candidate) > 4.0:
                    break
                if self._overlaps(node, candidate):
                    self._add_bidirectional_edge(
                        edges,
                        edge_type="co_occurs",
                        source_id=node.node_id,
                        target_id=candidate.node_id,
                        video_id=video_id,
                        attributes={"source": "builder", "reason": "time_overlap"},
                    )
                    overlap_budget += 1
                if overlap_budget >= 4:
                    break

        step_like_nodes = [node for node in temporal_nodes if node.node_type in {"segment", "recipe_step", "timeline_event"}]
        step_like_nodes.sort(key=lambda item: (float(item.start_time or 0.0), float(item.end_time or item.start_time or 0.0), item.node_id))
        step_groups: dict[str, list[GraphNodeRecord]] = {}
        for node in step_like_nodes:
            step_groups.setdefault(str(node.label), []).append(node)
        for siblings in step_groups.values():
            if len(siblings) < 2:
                continue
            for left_index, left in enumerate(siblings):
                for right in siblings[left_index + 1 : left_index + 3]:
                    self._add_bidirectional_edge(
                        edges,
                        edge_type="same_step",
                        source_id=left.node_id,
                        target_id=right.node_id,
                        video_id=video_id,
                        attributes={"source": "builder", "reason": "shared_step_label"},
                    )
        for node in temporal_nodes:
            if node.node_type in {"segment", "recipe_step", "timeline_event"}:
                continue
            linked = 0
            nearest_candidates: list[tuple[float, GraphNodeRecord]] = []
            for candidate in step_like_nodes:
                if self._overlaps(node, candidate, tolerance=2.5):
                    self._add_bidirectional_edge(
                        edges,
                        edge_type="same_step",
                        source_id=node.node_id,
                        target_id=candidate.node_id,
                        video_id=video_id,
                        attributes={"source": "builder", "reason": "shared_window"},
                    )
                    linked += 1
                    if linked >= 3:
                        break
                else:
                    distance = self._center_distance(node, candidate)
                    if distance <= 5.0:
                        nearest_candidates.append((distance, candidate))
            if linked == 0 and nearest_candidates:
                nearest_candidates.sort(key=lambda item: (item[0], item[1].node_id))
                for distance, candidate in nearest_candidates[:2]:
                    self._add_bidirectional_edge(
                        edges,
                        edge_type="same_step",
                        source_id=node.node_id,
                        target_id=candidate.node_id,
                        video_id=video_id,
                        attributes={"source": "builder", "reason": "nearest_step_fallback", "distance_s": round(distance, 3)},
                    )
        return list(edges.values())

    def _add_temporal_pair(
        self,
        edges: dict[str, GraphEdgeRecord],
        *,
        previous: GraphNodeRecord,
        current: GraphNodeRecord,
        video_id: str,
    ) -> None:
        self._add_edge(
            edges,
            GraphEdgeRecord(
                edge_id=f"before:{previous.node_id}:{current.node_id}",
                source_id=previous.node_id,
                target_id=current.node_id,
                edge_type="before",
                video_id=video_id,
                attributes={"source": "builder", "reason": "adjacent_temporal_order"},
            ),
        )
        self._add_edge(
            edges,
            GraphEdgeRecord(
                edge_id=f"after:{current.node_id}:{previous.node_id}",
                source_id=current.node_id,
                target_id=previous.node_id,
                edge_type="after",
                video_id=video_id,
                attributes={"source": "builder", "reason": "adjacent_temporal_order"},
            ),
        )

    def _add_bidirectional_edge(
        self,
        edges: dict[str, GraphEdgeRecord],
        *,
        edge_type: str,
        source_id: str,
        target_id: str,
        video_id: str,
        attributes: dict[str, Any],
    ) -> None:
        self._add_edge(
            edges,
            GraphEdgeRecord(
                edge_id=f"{edge_type}:{source_id}:{target_id}",
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                video_id=video_id,
                attributes=attributes,
            ),
        )
        self._add_edge(
            edges,
            GraphEdgeRecord(
                edge_id=f"{edge_type}:{target_id}:{source_id}",
                source_id=target_id,
                target_id=source_id,
                edge_type=edge_type,
                video_id=video_id,
                attributes=attributes,
            ),
        )

    def _add_edge(self, edges: dict[str, GraphEdgeRecord], edge: GraphEdgeRecord) -> None:
        edges.setdefault(edge.edge_id, edge)

    def _overlaps(self, left: GraphNodeRecord, right: GraphNodeRecord, tolerance: float = 0.0) -> bool:
        left_start = float(left.start_time or 0.0)
        left_end = float(left.end_time if left.end_time is not None else left_start)
        right_start = float(right.start_time or 0.0)
        right_end = float(right.end_time if right.end_time is not None else right_start)
        return max(left_start, right_start) <= min(left_end, right_end) + tolerance

    def _temporal_gap(self, left: GraphNodeRecord, right: GraphNodeRecord) -> float:
        left_end = float(left.end_time if left.end_time is not None else left.start_time or 0.0)
        right_start = float(right.start_time or 0.0)
        return max(0.0, right_start - left_end)

    def _center_distance(self, left: GraphNodeRecord, right: GraphNodeRecord) -> float:
        left_center = self._center_time(left)
        right_center = self._center_time(right)
        return abs(left_center - right_center)

    def _center_time(self, node: GraphNodeRecord) -> float:
        start = float(node.start_time or 0.0)
        end = float(node.end_time if node.end_time is not None else start)
        return (start + end) / 2.0

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
