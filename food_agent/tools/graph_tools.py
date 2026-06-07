"""Graph query tools."""

from __future__ import annotations

import json
from typing import Any

from food_agent.memory import GraphEdgeRecord, GraphMemoryStore, GraphNodeRecord
from food_agent.retrieval import EventRetriever, ObjectRetriever, TimeRetriever


class GraphToolbox:
    def __init__(self, store: GraphMemoryStore):
        self.store = store
        self.time_retriever = TimeRetriever(store)
        self.object_retriever = ObjectRetriever(store)
        self.event_retriever = EventRetriever(store)

    def query_time(self, *, video_id: str, start_time: float | None, end_time: float | None, limit: int = 20) -> list[dict[str, Any]]:
        return self.time_retriever.retrieve(video_id=video_id, start_time=start_time, end_time=end_time, limit=limit)

    def query_object(self, *, video_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.object_retriever.retrieve(video_id=video_id, query=query, limit=limit)

    def query_event(
        self,
        *,
        video_id: str,
        event_types: list[str] | None = None,
        keyword: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.event_retriever.retrieve(
            video_id=video_id,
            event_types=event_types,
            keyword=keyword,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def get_neighbors(self, *, node_ids: list[str], edge_types: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.get_neighbors(node_ids=node_ids, edge_types=edge_types, limit=limit)

    def query_state(
        self,
        *,
        video_id: str,
        state_keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.store.query_nodes(
            video_id=video_id,
            node_types=["frame", "timeline_event", "observation", "ingredient_event", "recipe_step"],
            keyword=state_keyword,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )

    def query_location(
        self,
        *,
        video_id: str,
        location_keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.store.query_nodes(
            video_id=video_id,
            node_types=["frame", "segment", "observation", "object_track", "timeline_event"],
            keyword=location_keyword,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )

    def query_region(
        self,
        *,
        video_id: str,
        object_hint: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.store.query_nodes(
            video_id=video_id,
            node_types=["object_track", "frame", "observation"],
            keyword=object_hint,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )

    def query_ocr(
        self,
        *,
        video_id: str,
        keyword: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.store.query_nodes(
            video_id=video_id,
            node_types=["ocr_reading", "observation", "ingredient_event", "audio_event", "segment", "recipe_step", "object_track", "timeline_event", "frame"],
            keyword=keyword,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )

    def write_node(
        self,
        *,
        node_id: str,
        node_type: str,
        label: str,
        video_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        attributes: dict[str, Any] | None = None,
        evidence_paths: list[str] | None = None,
        keywords: list[str] | None = None,
        edge_type: str = "supports",
        edge_attributes: dict[str, Any] | None = None,
        source_tool: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        payload_attributes = dict(attributes or {})
        if "source_tool" not in payload_attributes:
            payload_attributes["source_tool"] = str(source_tool or "agent_writeback")
        if "confidence" not in payload_attributes:
            payload_attributes["confidence"] = float(confidence) if confidence is not None else 0.0
        node = GraphNodeRecord(
            node_id=node_id,
            node_type=node_type,
            label=label,
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            attributes=payload_attributes,
            evidence_paths=evidence_paths or [],
            keywords=keywords or [],
        )
        self.store.upsert_node(node)
        self.store.upsert_edge(
            GraphEdgeRecord(
                edge_id=f"{edge_type}:{node_id}",
                source_id=f"video:{video_id}",
                target_id=node_id,
                edge_type=edge_type,
                video_id=video_id,
                attributes=edge_attributes or {"source": "agent_writeback"},
            )
        )
        return self.store.get_node(node_id) or {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "attributes": payload_attributes,
            "evidence_paths": evidence_paths or [],
            "keywords": keywords or [],
        }

    def write_edge(
        self,
        *,
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        video_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.store.upsert_edge(
            GraphEdgeRecord(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                video_id=video_id,
                attributes=attributes or {},
            )
        )
        return {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "video_id": video_id,
            "attributes": attributes or {},
        }

    def debug_dump_node(self, *, node_id: str) -> str:
        node = self.store.get_node(node_id)
        return json.dumps(node, ensure_ascii=False, indent=2) if node else ""
