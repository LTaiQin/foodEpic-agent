"""Controlled wrapper around LightAgent for fair baselines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .hd_epic_tools import HDEpicToolset
from .task_router import FoodTaskRouter


LIGHTAGENT_ROOT = Path("/22liushoulong/agent/agent-context-isolation/third_party/LightAgent")
if LIGHTAGENT_ROOT.exists() and LIGHTAGENT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, LIGHTAGENT_ROOT.as_posix())


class LightAgentLike(Protocol):
    tool_registry: Any
    loaded_tools: dict

    def run(self, *args: Any, **kwargs: Any) -> Any:
        ...


def import_lightagent_class() -> Any:
    from LightAgent import LightAgent  # noqa: PLC0415

    return LightAgent


@dataclass
class FoodAgentRun:
    content: str
    task_family: str
    exposed_tools: list[str]
    trace: list[dict[str, Any]]
    raw_result: Any


class FoodAgentLightWrapper:
    """Wrap LightAgent with explicit tool exposure and trace capture."""

    def __init__(
        self,
        agent: LightAgentLike,
        toolset: HDEpicToolset | None = None,
        router: FoodTaskRouter | None = None,
    ):
        self.agent = agent
        self.toolset = toolset or HDEpicToolset()
        self.router = router or FoodTaskRouter()
        self._all_tools = {tool.tool_info["tool_name"]: tool for tool in self.toolset.tools()}

    def clear_default_tools(self) -> None:
        """Remove LightAgent's auto-registered default tools for clean baselines."""
        self.agent.tool_registry.function_mappings.clear()
        self.agent.tool_registry.function_info.clear()
        self.agent.tool_registry.openai_function_schemas.clear()
        self.agent.loaded_tools.clear()

    def run(
        self,
        question: str,
        *,
        baseline: str,
        task_family: str | None = None,
        history: list[dict[str, str]] | None = None,
        user_id: str = "default_user",
    ) -> FoodAgentRun:
        baseline = baseline.lower()
        route = self.router.route(question, task_family=task_family)
        tools = self._select_tools(baseline, route.tool_names)
        exposed = [tool.tool_info["tool_name"] for tool in tools]
        self.clear_default_tools()
        result = self.agent.run(
            question,
            tools=tools or None,
            history=history or [],
            user_id=user_id,
            use_skills=False,
            result_format="object",
            trace=True,
        )
        return FoodAgentRun(
            content=result.content,
            task_family=route.task_family,
            exposed_tools=exposed,
            trace=result.trace,
            raw_result=result,
        )

    def _select_tools(self, baseline: str, routed_names: list[str]) -> list:
        if baseline in {"original", "textonly", "text_only"}:
            return []
        if baseline in {"hdtools", "hd_tools"}:
            return list(self._all_tools.values())
        if baseline in {"rag", "foodmemory", "ours", "ours-lightagent"}:
            return [self._all_tools[name] for name in routed_names if name in self._all_tools]
        raise ValueError(f"unknown baseline: {baseline}")
