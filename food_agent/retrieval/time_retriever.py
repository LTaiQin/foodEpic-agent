"""Time-window retrieval over graph memory."""

from __future__ import annotations

from typing import Any

from food_agent.memory import GraphMemoryStore


class TimeRetriever:
    def __init__(self, store: GraphMemoryStore):
        self.store = store

    def retrieve(self, *, video_id: str, start_time: float | None, end_time: float | None, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.query_nodes(video_id=video_id, time_start=start_time, time_end=end_time, limit=limit)
