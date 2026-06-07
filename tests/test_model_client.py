from pathlib import Path
from types import SimpleNamespace

from openai import APIStatusError

from food_agent.config import ModelConfig
from food_agent.model_client import ModelResponse, OpenAICompatibleModelClient


def test_model_client_complete_with_mock(monkeypatch) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="chat_completions",
    )

    class MockCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "mock"
            msg = SimpleNamespace(content="ok")
            usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions()))
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_model_client_usage_snapshot_accumulates_usage_and_cost() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )
    client._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    client._estimated_cost_total = 0.0
    client._record_usage(ModelResponse(content="ok", raw={}, usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, estimated_cost=0.0002))
    snapshot = client.usage_snapshot()
    assert snapshot["prompt_tokens"] == 100.0
    assert snapshot["completion_tokens"] == 50.0
    assert snapshot["total_tokens"] == 150.0
    assert snapshot["estimated_cost"] == 0.0002


def test_model_client_retries_on_retryable_status(monkeypatch) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="chat_completions",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
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


def test_model_client_extracts_responses_completed_sse_payload() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")
    client.client = None
    raw = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","output":[]}}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message","content":[{"type":"output_text","text":"{\\"ok\\":true}"}]}]}}\n\n'
        'data: [DONE]\n\n'
    )
    assert client._extract_content(raw) == '{"ok":true}'


def test_model_client_complete_json_extracts_object() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")
    client.complete = lambda messages, temperature=0.0: ModelResponse(content='{"tool":"finish","args":{"prediction":1}}', raw={})  # type: ignore[method-assign]
    payload = client.complete_json([{"role": "user", "content": "hi"}])
    assert payload["tool"] == "finish"
    assert payload["args"]["prediction"] == 1


def test_model_client_inspect_images_with_chat_completions(monkeypatch, tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1", provider_mode="chat_completions")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpg")

    captured: dict[str, object] = {}

    def fake_request(messages, *, temperature, timeout_seconds):
        captured["messages"] = messages
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client._request_with_timeout_for_mode = lambda messages, *, temperature, timeout_seconds, mode: fake_request(messages, temperature=temperature, timeout_seconds=timeout_seconds)  # type: ignore[method-assign]
    response = client.inspect_images(prompt="look", image_paths=[image_path])
    assert response.content == "ok"
    messages = captured["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert str(content[1]["image_url"]["url"]).startswith("data:image/jpeg;base64,")


def test_model_client_inspect_images_with_responses_mode(monkeypatch, tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1", provider_mode="responses")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-png")

    captured: dict[str, object] = {}

    def fake_request(messages, *, temperature, timeout_seconds):
        captured["messages"] = messages
        return SimpleNamespace(output_text="ok")

    client._request_with_timeout_for_mode = lambda messages, *, temperature, timeout_seconds, mode: fake_request(messages, temperature=temperature, timeout_seconds=timeout_seconds)  # type: ignore[method-assign]
    response = client.inspect_images(prompt="look", image_paths=[image_path])
    assert response.content == "ok"
    messages = captured["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert str(content[1]["image_url"]).startswith("data:image/png;base64,")


def test_model_client_inspect_images_falls_back_to_alternate_mode_on_empty_content(tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1", provider_mode="chat_completions")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpg")

    calls: list[str] = []

    def fake_request(messages, *, temperature, timeout_seconds, mode):
        calls.append(mode)
        if mode == "chat_completions":
            return 'data: {"choices":[],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n'
        return SimpleNamespace(output_text="fallback-ok")

    client._request_with_timeout_for_mode = fake_request  # type: ignore[method-assign]
    response = client.inspect_images(prompt="look", image_paths=[image_path])
    assert response.content == "fallback-ok"
    assert calls == ["chat_completions", "responses"]


def test_model_client_prefers_responses_for_cctq_vision_auto_mode() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="https://www.cctq.ai/v1",
        provider_mode="chat_completions",
        vision_provider_mode="auto",
    )
    assert client._preferred_vision_mode() == "responses"


def test_model_client_prefers_chat_completions_for_right_codes_vision_auto_mode() -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="https://right.codes/codex/v1",
        provider_mode="responses",
        vision_provider_mode="auto",
    )
    assert client._preferred_vision_mode() == "chat_completions"


def test_model_client_inspect_images_uses_vision_timeout(tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="chat_completions",
        request_timeout_seconds=120.0,
        vision_timeout_seconds=17.0,
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpg")

    captured: dict[str, object] = {}

    def fake_request(messages, *, temperature, timeout_seconds):
        captured["timeout_seconds"] = timeout_seconds
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client._request_with_timeout_for_mode = lambda messages, *, temperature, timeout_seconds, mode: fake_request(messages, temperature=temperature, timeout_seconds=timeout_seconds)  # type: ignore[method-assign]
    response = client.inspect_images(prompt="look", image_paths=[image_path])
    assert response.content == "ok"
    assert captured["timeout_seconds"] == 17.0


def test_model_client_inspect_images_uses_vision_retry_budget(monkeypatch, tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="chat_completions",
        max_retries=3,
        vision_max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._vision_support_state = None
    client._vision_disable_reason = ""
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpg")
    calls = {"count": 0}

    class MockResponse:
        request = None
        status_code = 503
        headers = {}

    def fake_request(messages, *, temperature, timeout_seconds):
        calls["count"] += 1
        raise APIStatusError("temporary", response=MockResponse(), body={})

    monkeypatch.setattr("food_agent.model_client.time.sleep", lambda *_args, **_kwargs: None)
    client._request_with_timeout_for_mode = lambda messages, *, temperature, timeout_seconds, mode: fake_request(messages, temperature=temperature, timeout_seconds=timeout_seconds)  # type: ignore[method-assign]
    try:
        client.inspect_images(prompt="look", image_paths=[image_path])
    except RuntimeError as exc:
        assert "after 2 attempts" in str(exc)
    else:
        raise AssertionError("expected runtime error")
    assert calls["count"] == 2


def test_model_client_inspect_images_maps_image_generation_disabled_to_vision_not_supported(tmp_path: Path) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(
        model="mock",
        api_key="key",
        base_url="http://example.com/v1",
        provider_mode="chat_completions",
        max_retries=0,
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake-jpg")

    class MockResponse:
        request = None
        status_code = 403
        headers = {}

    def fake_request(messages, *, temperature, timeout_seconds):
        raise APIStatusError("Image generation is not enabled for this group", response=MockResponse(), body={})

    client._request_with_timeout_for_mode = lambda messages, *, temperature, timeout_seconds, mode: fake_request(messages, temperature=temperature, timeout_seconds=timeout_seconds)  # type: ignore[method-assign]
    try:
        client.inspect_images(prompt="look", image_paths=[image_path])
    except RuntimeError as exc:
        assert "vision_not_supported:image_generation_disabled" in str(exc)
    else:
        raise AssertionError("expected vision_not_supported error")
    assert client.supports_vision_requests() is False
    assert client.vision_disable_reason() == "image_generation_disabled"


def test_model_client_supports_vision_requests_can_be_disabled_by_env(monkeypatch) -> None:
    client = OpenAICompatibleModelClient.__new__(OpenAICompatibleModelClient)
    client.config = ModelConfig(model="mock", api_key="key", base_url="http://example.com/v1")
    client._vision_support_state = None
    client._vision_disable_reason = ""
    monkeypatch.setenv("FOOD_AGENT_DISABLE_VISION", "1")
    assert client.supports_vision_requests() is False
    assert client.vision_disable_reason() == "disabled_by_env"
