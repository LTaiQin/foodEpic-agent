from types import SimpleNamespace

from openai import APIStatusError

from food_agent.config import ModelConfig
from food_agent.model_client import ModelResponse, OpenAICompatibleModelClient


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


def test_model_client_retries_on_retryable_status(monkeypatch) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1", max_retries=2, retry_backoff_seconds=0.0)
    calls = {"count": 0}

    class MockResponse:
        request = None
        status_code = 503
        headers = {}

    class MockCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise APIStatusError("temporary", response=MockResponse(), body={})
            msg = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr("food_agent.model_client.time.sleep", lambda *_args, **_kwargs: None)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions()))
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert calls["count"] == 3


def test_model_client_complete_with_responses_mode() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="responses",
    )

    class MockResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "mock"
            return SimpleNamespace(output_text="ok")

    client.client = SimpleNamespace(responses=MockResponses())
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"


def test_model_client_extracts_sse_string_payload() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")
    client.client = None
    raw = (
        'data: {"choices":[{"delta":{"content":"O"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"K"}}]}\n\n'
        'data: [DONE]\n\n'
    )
    assert client._extract_content(raw) == "OK"


def test_model_client_complete_json_extracts_object() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")
    client.complete = lambda messages, temperature=0.0: ModelResponse(content='{"tool":"finish","args":{"prediction":1}}', raw={})  # type: ignore[method-assign]
    payload = client.complete_json([{"role": "user", "content": "hi"}])
    assert payload["tool"] == "finish"
    assert payload["args"]["prediction"] == 1
