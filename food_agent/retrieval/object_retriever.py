"""Object-centric retrieval over graph memory."""

from __future__ import annotations

from typing import Any

from food_agent.memory import GraphMemoryStore


class ObjectRetriever:
    def __init__(self, store: GraphMemoryStore):
        self.store = store

    def retrieve(self, *, video_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.query_nodes(video_id=video_id, keyword=query, limit=limit)
