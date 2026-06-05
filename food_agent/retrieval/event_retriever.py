"""Event-centric retrieval over graph memory."""

from __future__ import annotations

from typing import Any

from food_agent.memory import GraphMemoryStore


class EventRetriever:
    def __init__(self, store: GraphMemoryStore):
        self.store = store

    def retrieve(
        self,
        *,
        video_id: str,
        event_types: list[str] | None = None,
        keyword: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.store.query_nodes(
            video_id=video_id,
            node_types=event_types,
            keyword=keyword,
            time_start=start_time,
            time_end=end_time,
            limit=limit,
        )
