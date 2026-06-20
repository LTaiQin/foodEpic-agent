"""API client for LLM with retry, caching, and error handling.

Supports OpenAI (Chat Completions / Response API) and Anthropic (Messages API).
Anthropic supports native tool-calling for more reliable tool use.
"""

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from food_agent.utils.cache import CacheManager


class MimoClient:
    """API client supporting OpenAI and Anthropic formats.

    Supports both text and vision (image+text) requests with
    automatic retry, caching, and error handling.

    For Anthropic, supports native tool-calling via call_with_tools().
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider_mode: Optional[str] = None,
        max_retries: int = 3,
        cache_ttl: float = 3600.0,
    ):
        from food_agent.config import load_env_file

        # Load .env if exists
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_env_file(env_path)

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("FOOD_AGENT_MODEL", "gpt-5.4")
        self.provider_mode = provider_mode or os.environ.get("FOOD_AGENT_PROVIDER_MODE", "chat_completions")
        self.max_retries = max_retries
        self._cache = CacheManager(default_ttl=cache_ttl)
        self._client = None
        self._anthropic_client = None

    @property
    def is_anthropic(self) -> bool:
        """Check if using Anthropic API."""
        return "anthropic" in self.base_url.lower() or self.provider_mode == "anthropic"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0,
            )
        return self._client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            kwargs = {"api_key": self.api_key}
            # Use custom base URL if provided (for proxies)
            if self.base_url and "anthropic" in self.base_url.lower():
                kwargs["base_url"] = self.base_url
            self._anthropic_client = anthropic.Anthropic(**kwargs)
        return self._anthropic_client

    def _cache_key(self, prompt: str, image_hash: str = "") -> str:
        raw = f"{self.model}:{prompt}:{image_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a BGR numpy array to base64 JPEG."""
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _extract_response_text(self, response) -> str:
        """Extract text from API response (handles both formats)."""
        # Try Response API format
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content in item.content:
                        if hasattr(content, 'text'):
                            return content.text
        # Try Chat Completions format
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content or ""
        # Try Anthropic format
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        return ""

    def call_text(self, prompt: str, system: str = "") -> str:
        """Send a text-only request to the LLM."""
        cache_key = self._cache_key(prompt)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        for attempt in range(self.max_retries):
            try:
                if self.is_anthropic:
                    text = self._call_anthropic_text(prompt, system)
                elif self.provider_mode == "responses":
                    text = self._call_openai_responses_text(prompt, system)
                else:
                    text = self._call_openai_chat_text(prompt, system)

                self._cache.put(cache_key, text)
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"API error: {e}"

        return "API error: max retries exceeded"

    def call_vision(self, image: np.ndarray, prompt: str, system: str = "") -> str:
        """Send a vision (image + text) request to the LLM."""
        img_b64 = self._encode_image(image)
        img_hash = hashlib.md5(img_b64[:100].encode()).hexdigest()
        cache_key = self._cache_key(prompt, img_hash)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        for attempt in range(self.max_retries):
            try:
                if self.is_anthropic:
                    text = self._call_anthropic_vision(img_b64, prompt, system)
                elif self.provider_mode == "responses":
                    text = self._call_openai_responses_vision(img_b64, prompt, system)
                else:
                    text = self._call_openai_chat_vision(img_b64, prompt, system)

                self._cache.put(cache_key, text)
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"API error: {e}"

        return "API error: max retries exceeded"

    def call_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_executor: Callable,
        system: str = "",
        max_rounds: int = 5,
    ) -> Dict:
        """Call LLM with native tool-calling support.

        For Anthropic: uses native tool_use blocks.
        For OpenAI: uses function_call in chat completions.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (Anthropic or OpenAI format).
            tool_executor: Function(tool_name, tool_input) -> result.
            system: System prompt.
            max_rounds: Max tool-calling rounds.

        Returns:
            Dict with 'answer', 'tool_calls', 'rounds'.
        """
        if self.is_anthropic:
            return self._call_anthropic_with_tools(messages, tools, tool_executor, system, max_rounds)
        else:
            return self._call_openai_with_tools(messages, tools, tool_executor, system, max_rounds)

    # --- OpenAI implementations ---

    def _call_openai_chat_text(self, prompt: str, system: str = "") -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    def _call_openai_responses_text(self, prompt: str, system: str = "") -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.responses.create(
            model=self.model, input=messages, max_output_tokens=2048,
        )
        return self._extract_response_text(response)

    def _call_openai_chat_vision(self, img_b64: str, prompt: str, system: str = "") -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        })
        response = client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    def _call_openai_responses_vision(self, img_b64: str, prompt: str, system: str = "") -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        })
        response = client.responses.create(
            model=self.model, input=messages, max_output_tokens=2048,
        )
        return self._extract_response_text(response)

    def _call_openai_with_tools(self, messages, tools, tool_executor, system, max_rounds):
        """OpenAI tool-calling (chat completions)."""
        client = self._get_client()
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        tool_calls_log = []

        for round_idx in range(max_rounds):
            response = client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                tools=tools if tools else None,
                max_tokens=2048,
            )

            msg = response.choices[0].message

            # If no tool calls, we're done
            if not msg.tool_calls:
                return {
                    "answer": msg.content or "",
                    "tool_calls": tool_calls_log,
                    "rounds": round_idx + 1,
                }

            # Execute tool calls
            all_messages.append(msg)
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                result = tool_executor(tool_name, tool_input)
                tool_calls_log.append({"tool": tool_name, "input": tool_input, "result": str(result)[:500]})

                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)[:2000],
                })

        return {
            "answer": "Max tool rounds reached",
            "tool_calls": tool_calls_log,
            "rounds": max_rounds,
        }

    # --- Anthropic implementations ---

    def _call_anthropic_text(self, prompt: str, system: str = "") -> str:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system if system else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_response_text(response)

    def _call_anthropic_vision(self, img_b64: str, prompt: str, system: str = "") -> str:
        client = self._get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system if system else anthropic.NOT_GIVEN,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return self._extract_response_text(response)

    def _call_anthropic_with_tools(self, messages, tools, tool_executor, system, max_rounds):
        """Anthropic native tool-calling."""
        import anthropic
        client = self._get_anthropic_client()

        # Convert tools to Anthropic format if needed
        anthropic_tools = self._convert_tools_to_anthropic(tools)

        all_messages = list(messages)
        tool_calls_log = []

        for round_idx in range(max_rounds):
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": all_messages,
                "tools": anthropic_tools,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            # Check if model wants to use tools
            if response.stop_reason == "tool_use":
                # Extract tool use blocks
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        result = tool_executor(tool_name, tool_input)
                        tool_calls_log.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": str(result)[:500],
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)[:2000],
                        })

                # Add assistant message and tool results
                all_messages.append({"role": "assistant", "content": response.content})
                all_messages.append({"role": "user", "content": tool_results})
            else:
                # Model finished, extract text
                text = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        text += block.text
                return {
                    "answer": text,
                    "tool_calls": tool_calls_log,
                    "rounds": round_idx + 1,
                }

        return {
            "answer": "Max tool rounds reached",
            "tool_calls": tool_calls_log,
            "rounds": max_rounds,
        }

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool:
                # Already in Anthropic format
                anthropic_tools.append(tool)
        return anthropic_tools
