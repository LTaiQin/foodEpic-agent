"""Graph query tools."""

from __future__ import annotations

from typing import Any

from food_agent.memory import GraphMemoryStore
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
