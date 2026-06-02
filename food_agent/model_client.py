"""OpenAI-compatible model client used by direct baselines and our agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import ModelConfig


@dataclass(frozen=True)
class ModelResponse:
    content: str
    raw: Any


class OpenAICompatibleModelClient:
    """Small wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: ModelConfig | None = None, use_env_proxy: bool = False):
        if not use_env_proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                os.environ.pop(key, None)
        self.config = config or ModelConfig.from_env()
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
        )
        return ModelResponse(content=response.choices[0].message.content or "", raw=response)

