from types import SimpleNamespace

from food_agent.config import ModelConfig
from food_agent.model_client import OpenAICompatibleModelClient


def test_model_client_complete_with_mock(monkeypatch) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")

    class MockCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "mock"
            msg = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions()))
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"

