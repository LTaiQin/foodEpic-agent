from types import SimpleNamespace

from food_agent.lightagent_wrapper import FoodAgentLightWrapper


class FakeRegistry:
    def __init__(self) -> None:
        self.function_mappings = {"execute_python_code": object()}
        self.function_info = {"execute_python_code": {}}
        self.openai_function_schemas = [{"function": {"name": "execute_python_code"}}]


class FakeAgent:
    def __init__(self) -> None:
        self.tool_registry = FakeRegistry()
        self.loaded_tools = {"execute_python_code": object()}
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        tools = kwargs.get("tools") or []
        trace = [{"type": "model_request", "data": {"tools": [tool.tool_info["tool_name"] for tool in tools]}}]
        return SimpleNamespace(content="ok", trace=trace)


def make_wrapper() -> tuple[FoodAgentLightWrapper, FakeAgent]:
    agent = FakeAgent()
    return FoodAgentLightWrapper(agent), agent


def test_textonly_exposes_no_tools() -> None:
    wrapper, agent = make_wrapper()
    result = wrapper.run("What is happening?", baseline="textonly")
    assert result.exposed_tools == []
    assert agent.calls[0]["tools"] is None
    assert agent.loaded_tools == {}


def test_hdtools_exposes_data_tools() -> None:
    wrapper, agent = make_wrapper()
    result = wrapper.run("What ingredient was added?", baseline="hdtools")
    assert "get_video_metadata" in result.exposed_tools
    assert agent.calls[0]["tools"]
    assert "execute_python_code" not in {tool.tool_info["tool_name"] for tool in agent.calls[0]["tools"]}
