from food_agent.config import ModelConfig


def test_model_config_from_env_uses_normalized_right_codes_defaults(monkeypatch) -> None:
    monkeypatch.delenv("FOOD_AGENT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("FOOD_AGENT_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("FOOD_AGENT_VISION_PROVIDER_MODE", raising=False)

    cfg = ModelConfig.from_env()

    assert cfg.model == "gpt-5.4"
    assert cfg.base_url == "https://right.codes/codex/v1"
    assert cfg.provider_mode == "responses"
    assert cfg.vision_provider_mode == "auto"


def test_model_config_from_env_strips_whitespace_and_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("FOOD_AGENT_MODEL", " gpt-5.4 ")
    monkeypatch.setenv("OPENAI_API_KEY", " sk-test ")
    monkeypatch.setenv("OPENAI_BASE_URL", " https://right.codes/codex/v1/ ")
    monkeypatch.setenv("FOOD_AGENT_PROVIDER_MODE", " responses ")
    monkeypatch.setenv("FOOD_AGENT_VISION_PROVIDER_MODE", " auto ")

    cfg = ModelConfig.from_env()

    assert cfg.model == "gpt-5.4"
    assert cfg.api_key == "sk-test"
    assert cfg.base_url == "https://right.codes/codex/v1"
    assert cfg.provider_mode == "responses"
    assert cfg.vision_provider_mode == "auto"
