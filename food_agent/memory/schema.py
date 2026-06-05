"""Schema primitives for the graph-backed video memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNodeRecord:
    node_id: str
    node_type: str
    label: str
    video_id: str
    start_time: float | None = None
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdgeRecord:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    video_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
