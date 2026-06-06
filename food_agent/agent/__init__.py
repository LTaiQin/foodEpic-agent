"""Graph agent components."""

from __future__ import annotations

from typing import Any


__all__ = ["GraphAgent", "GraphAgentVideoSession"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .graph_agent import GraphAgent, GraphAgentVideoSession

        exports = {
            "GraphAgent": GraphAgent,
            "GraphAgentVideoSession": GraphAgentVideoSession,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
