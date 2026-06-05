"""OpenAI-compatible model client used by direct baselines and our agent."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI

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

    def complete(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._request(messages, temperature=temperature)
                return ModelResponse(content=self._extract_content(response), raw=response)
            except (APIConnectionError, InternalServerError) as exc:
                last_error = exc
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise
            if attempt >= self.config.max_retries:
                break
            time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(
            f"model request failed after {self.config.max_retries + 1} attempts for model={self.config.model}: {last_error}"
        ) from last_error

    def complete_json(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> dict[str, Any]:
        response = self.complete(messages, temperature=temperature)
        return self._extract_json_object(response.content)

    def inspect_images(self, *, prompt: str, image_paths: list[Path], temperature: float = 0.0) -> ModelResponse:
        if not image_paths:
            raise ValueError("inspect_images requires at least one image path")
        mode = (self.config.provider_mode or "chat_completions").strip().lower()
        if mode == "responses":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            content.extend(self._responses_image_part(path) for path in image_paths)
            messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            content = [{"type": "text", "text": prompt}]
            content.extend(self._chat_image_part(path) for path in image_paths)
            messages = [{"role": "user", "content": content}]
        return self.complete(messages, temperature=temperature)

    def _request(self, messages: list[dict[str, Any]], temperature: float) -> Any:
        mode = (self.config.provider_mode or "chat_completions").strip().lower()
        if mode == "responses":
            return self.client.responses.create(
                model=self.config.model,
                input=self._normalize_responses_input(messages),
                temperature=temperature,
                timeout=self.config.request_timeout_seconds,
            )
        if mode != "chat_completions":
            raise RuntimeError(f"unsupported provider mode: {self.config.provider_mode}")
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=self._normalize_chat_messages(messages),
            temperature=temperature,
            timeout=self.config.request_timeout_seconds,
        )

    def _extract_content(self, response: Any) -> str:
        if isinstance(response, str):
            return _extract_sse_text(response)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    item_text = getattr(item, "text", None)
                    if isinstance(item_text, str):
                        text_parts.append(item_text)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                if text_parts:
                    return "".join(text_parts)
        output = getattr(response, "output", None)
        if output:
            text = _extract_response_output_text(output)
            if text:
                return text
        if hasattr(response, "model_dump_json"):
            dumped = response.model_dump_json()
            sse_text = _extract_sse_text(dumped)
            if sse_text:
                return sse_text
        raise RuntimeError(f"unsupported model response format: {type(response).__name__}")

    def _normalize_responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append({"role": role, "content": [{"type": "input_text", "text": content}]})
                continue
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append({"type": "input_text", "text": str(item.get("text", ""))})
                    else:
                        parts.append(item)
                normalized.append({"role": role, "content": parts})
                continue
            normalized.append({"role": role, "content": [{"type": "input_text", "text": str(content)}]})
        return normalized

    def _normalize_chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "input_text":
                        parts.append({"type": "text", "text": str(item.get("text", ""))})
                    elif item_type == "input_image":
                        parts.append({"type": "image_url", "image_url": {"url": str(item.get("image_url", ""))}})
                    else:
                        parts.append(item)
                normalized.append({"role": role, "content": parts})
                continue
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def _responses_image_part(self, path: Path) -> dict[str, str]:
        payload = _image_to_data_url(path)
        return {"type": "input_image", "image_url": payload}

    def _chat_image_part(self, path: Path) -> dict[str, Any]:
        payload = _image_to_data_url(path)
        return {"type": "image_url", "image_url": {"url": payload}}

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"model did not return a JSON object: {text[:300]}")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise RuntimeError(f"model JSON payload is not an object: {payload!r}")
        return payload


SSE_DATA_PATTERN = re.compile(r"^data:\s*(.+)$", re.MULTILINE)


def _extract_sse_text(raw: str) -> str:
    text_parts: list[str] = []
    for line in SSE_DATA_PATTERN.findall(raw):
        line = line.strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _extract_response_output_text(payload.get("output")) if isinstance(payload, dict) else ""
        if text:
            text_parts.append(text)
            continue
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    text_parts.append(delta["content"])
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    text_parts.append(message["content"])
    return "".join(text_parts)


def _extract_response_output_text(output: Any) -> str:
    text_parts: list[str] = []
    if not isinstance(output, list):
        return ""
    for item in output:
        if hasattr(item, "type"):
            item_type = getattr(item, "type", None)
            content = getattr(item, "content", None)
        elif isinstance(item, dict):
            item_type = item.get("type")
            content = item.get("content")
        else:
            continue
        if item_type != "message" or not isinstance(content, list):
            continue
        for part in content:
            if hasattr(part, "type"):
                part_type = getattr(part, "type", None)
                part_text = getattr(part, "text", None)
            elif isinstance(part, dict):
                part_type = part.get("type")
                part_text = part.get("text")
            else:
                continue
            if part_type == "output_text" and isinstance(part_text, str):
                text_parts.append(part_text)
    return "".join(text_parts)


def _image_to_data_url(path: Path) -> str:
    import base64

    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"
