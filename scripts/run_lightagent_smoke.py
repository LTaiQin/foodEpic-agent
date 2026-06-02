#!/usr/bin/env python3
"""Smoke test the controlled LightAgent wrapper with a mock model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.lightagent_wrapper import FoodAgentLightWrapper, import_lightagent_class


class FakeRegistry:
    def __init__(self) -> None:
        self.function_mappings = {"execute_python_code": object()}
        self.function_info = {"execute_python_code": {}}
        self.openai_function_schemas = [{"function": {"name": "execute_python_code"}}]


class FakeAgent:
    def __init__(self) -> None:
        self.tool_registry = FakeRegistry()
        self.loaded_tools = {"execute_python_code": object()}
        self.calls: list[dict[str, Any]] = []

    def run(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        tools = kwargs.get("tools") or []
        trace = [{"type": "model_request", "data": {"tools": [tool.tool_info["tool_name"] for tool in tools]}}]
        return SimpleNamespace(content="mock answer", trace=trace)


class StaticCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="textonly")
    parser.add_argument("--question", default="What ingredient was added around 16 seconds?")
    parser.add_argument("--real-lightagent", action="store_true", help="Import and run the real LightAgent package.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completions = None
    if args.real_lightagent:
        LightAgent = import_lightagent_class()
        agent = LightAgent(
            model="mock-model",
            api_key="test-key",
            base_url="http://127.0.0.1:9/v1",
            auto_discover_skills=False,
        )
        completions = StaticCompletions("mock answer")
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    else:
        agent = FakeAgent()
    wrapper = FoodAgentLightWrapper(agent)
    result = wrapper.run(args.question, baseline=args.baseline)
    call = completions.calls[0] if completions else agent.calls[0]
    request_tools = call.get("tools") or []
    print("content:", result.content)
    print("task_family:", result.task_family)
    print("exposed_tools:", result.exposed_tools)
    print("request_tools:", [getattr(t, "tool_info", {}).get("tool_name") for t in request_tools])
    print("trace_types:", [event["type"] for event in result.trace])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
