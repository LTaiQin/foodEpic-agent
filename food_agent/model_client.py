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
    usage: dict[str, int] | None = None
    estimated_cost: float | None = None


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
        self._vision_support_state: bool | None = None
        self._vision_disable_reason: str = ""
        self._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._estimated_cost_total = 0.0

    def supports_vision_requests(self) -> bool:
        disable_flag = os.environ.get("FOOD_AGENT_DISABLE_VISION", "").strip().lower()
        if disable_flag in {"1", "true", "yes", "on"}:
            self._vision_support_state = False
            self._vision_disable_reason = "disabled_by_env"
            return False
        return getattr(self, "_vision_support_state", None) is not False

    def vision_disable_reason(self) -> str:
        return getattr(self, "_vision_disable_reason", "")

    def complete(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._request(messages, temperature=temperature)
                usage = self._extract_usage(response)
                model_response = ModelResponse(
                    content=self._extract_content(response),
                    raw=response,
                    usage=usage,
                    estimated_cost=self._estimate_cost_from_usage(usage),
                )
                self._record_usage(model_response)
                return model_response
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

    def usage_snapshot(self) -> dict[str, float]:
        return {
            "prompt_tokens": float(self._usage_totals.get("prompt_tokens") or 0),
            "completion_tokens": float(self._usage_totals.get("completion_tokens") or 0),
            "total_tokens": float(self._usage_totals.get("total_tokens") or 0),
            "estimated_cost": float(self._estimated_cost_total or 0.0),
        }

    def inspect_images(self, *, prompt: str, image_paths: list[Path], temperature: float = 0.0) -> ModelResponse:
        if not image_paths:
            raise ValueError("inspect_images requires at least one image path")
        if not self.supports_vision_requests():
            reason = self.vision_disable_reason() or "vision_disabled"
            raise RuntimeError(f"vision_not_supported:{reason}")
        timeout_seconds = float(getattr(self.config, "vision_timeout_seconds", self.config.request_timeout_seconds))
        preferred_mode = self._preferred_vision_mode()
        tried_modes: list[str] = []
        for mode in self._vision_request_modes(preferred_mode):
            tried_modes.append(mode)
            messages = self._vision_messages_for_mode(mode=mode, prompt=prompt, image_paths=image_paths)
            response = self._complete_with_timeout(
                messages,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                vision_request=True,
                request_mode=mode,
            )
            if not self._is_empty_vision_response(response):
                self._remember_working_vision_mode(mode)
                return response
        raise RuntimeError(f"vision request returned empty content for modes={tried_modes}")

    def _request(self, messages: list[dict[str, Any]], temperature: float) -> Any:
        return self._request_with_timeout(messages, temperature=temperature, timeout_seconds=self.config.request_timeout_seconds)

    def _request_with_timeout(self, messages: list[dict[str, Any]], *, temperature: float, timeout_seconds: float) -> Any:
        mode = (self.config.provider_mode or "chat_completions").strip().lower()
        return self._request_with_timeout_for_mode(messages, temperature=temperature, timeout_seconds=timeout_seconds, mode=mode)

    def _request_with_timeout_for_mode(self, messages: list[dict[str, Any]], *, temperature: float, timeout_seconds: float, mode: str) -> Any:
        if mode == "responses":
            return self.client.responses.create(
                model=self.config.model,
                input=self._normalize_responses_input(messages),
                temperature=temperature,
                timeout=timeout_seconds,
            )
        if mode != "chat_completions":
            raise RuntimeError(f"unsupported provider mode: {self.config.provider_mode}")
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=self._normalize_chat_messages(messages),
            temperature=temperature,
            timeout=timeout_seconds,
        )

    def _complete_with_timeout(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        timeout_seconds: float,
        vision_request: bool = False,
        request_mode: str | None = None,
    ) -> ModelResponse:
        last_error: Exception | None = None
        max_retries = (
            int(getattr(self.config, "vision_max_retries", self.config.max_retries))
            if vision_request
            else self.config.max_retries
        )
        for attempt in range(max_retries + 1):
            try:
                mode = request_mode or (self.config.provider_mode or "chat_completions").strip().lower()
                response = self._request_with_timeout_for_mode(
                    messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    mode=mode,
                )
                usage = self._extract_usage(response)
                model_response = ModelResponse(
                    content=self._extract_content(response),
                    raw=response,
                    usage=usage,
                    estimated_cost=self._estimate_cost_from_usage(usage),
                )
                self._record_usage(model_response)
                return model_response
            except (APIConnectionError, InternalServerError) as exc:
                last_error = exc
            except APIStatusError as exc:
                last_error = exc
                if vision_request:
                    unsupported_reason = self._classify_vision_unsupported_error(exc)
                    if unsupported_reason:
                        self._mark_vision_unavailable(unsupported_reason)
                        raise RuntimeError(f"vision_not_supported:{unsupported_reason}") from exc
                if exc.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise
            if attempt >= max_retries:
                break
            time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        request_kind = "vision" if vision_request else "model"
        raise RuntimeError(
            f"{request_kind} request failed after {max_retries + 1} attempts for model={self.config.model}: {last_error}"
        ) from last_error

    def _mark_vision_unavailable(self, reason: str) -> None:
        self._vision_support_state = False
        self._vision_disable_reason = reason

    def _classify_vision_unsupported_error(self, exc: APIStatusError) -> str | None:
        message = str(exc).lower()
        if exc.status_code == 403 and "image generation" in message:
            return "image_generation_disabled"
        keywords = (
            "vision",
            "image input",
            "input_image",
            "multimodal",
            "image_url",
            "does not support images",
            "not support image",
            "unsupported content type",
        )
        if exc.status_code in {400, 403, 404, 415, 422} and any(keyword in message for keyword in keywords):
            return "provider_rejected_image_input"
        return None

    def _extract_content(self, response: Any) -> str:
        if isinstance(response, str):
            sse_text = _extract_sse_text(response)
            if sse_text:
                return sse_text
            if SSE_DATA_PATTERN.search(response):
                return ""
            return response.strip()
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

    def _vision_messages_for_mode(self, *, mode: str, prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
        if mode == "responses":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            content.extend(self._responses_image_part(path) for path in image_paths)
            return [{"role": "user", "content": content}]
        content = [{"type": "text", "text": prompt}]
        content.extend(self._chat_image_part(path) for path in image_paths)
        return [{"role": "user", "content": content}]

    def _vision_request_modes(self, preferred_mode: str) -> list[str]:
        normalized = preferred_mode if preferred_mode in {"chat_completions", "responses"} else "chat_completions"
        fallback = "responses" if normalized == "chat_completions" else "chat_completions"
        return [normalized, fallback]

    def _preferred_vision_mode(self) -> str:
        cached = getattr(self, "_working_vision_mode", "")
        if cached in {"chat_completions", "responses"}:
            return cached
        configured = (getattr(self.config, "vision_provider_mode", "auto") or "auto").strip().lower()
        if configured in {"chat_completions", "responses"}:
            return configured
        base_url = str(getattr(self.config, "base_url", "") or "").lower()
        if "cctq.ai" in base_url:
            return "responses"
        return (self.config.provider_mode or "chat_completions").strip().lower()

    def _remember_working_vision_mode(self, mode: str) -> None:
        if mode in {"chat_completions", "responses"}:
            self._working_vision_mode = mode

    def _is_empty_vision_response(self, response: ModelResponse) -> bool:
        content = str(response.content or "").strip()
        if content:
            return False
        raw = response.raw
        if isinstance(raw, str):
            lowered = raw.lower()
            return '"completion_tokens":0' in lowered or "[done]" in lowered or '"choices":[]' in lowered
        output_text = getattr(raw, "output_text", None)
        if isinstance(output_text, str):
            return not output_text.strip()
        return True

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"model did not return a JSON object: {text[:300]}")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise RuntimeError(f"model JSON payload is not an object: {payload!r}")
        return payload

    def _extract_usage(self, response: Any) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage = usage.dict()
        if not isinstance(usage, dict):
            return None
        prompt_tokens = _coerce_int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = _coerce_int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = _coerce_int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _estimate_cost_from_usage(self, usage: dict[str, int] | None) -> float | None:
        if not usage:
            return None
        input_rate = float(getattr(self.config, "input_cost_per_million_tokens", 0.0) or 0.0)
        output_rate = float(getattr(self.config, "output_cost_per_million_tokens", 0.0) or 0.0)
        if input_rate <= 0.0 and output_rate <= 0.0:
            return None
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return (prompt_tokens / 1_000_000.0) * input_rate + (completion_tokens / 1_000_000.0) * output_rate

    def _record_usage(self, response: ModelResponse) -> None:
        if not hasattr(self, "_usage_totals"):
            self._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not hasattr(self, "_estimated_cost_total"):
            self._estimated_cost_total = 0.0
        usage = response.usage or {}
        self._usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._usage_totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self._usage_totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        self._estimated_cost_total += float(response.estimated_cost or 0.0)


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


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return 0
