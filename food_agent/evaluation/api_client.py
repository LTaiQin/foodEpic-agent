"""API client for MiMo2.5 with retry, caching, and error handling."""

import base64
import hashlib
import time
from typing import Optional

import cv2
import numpy as np

from food_agent.utils.cache import CacheManager


class MimoClient:
    """OpenAI-compatible API client for MiMo2.5.

    Supports both text and vision (image+text) requests with
    automatic retry, caching, and error handling.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        cache_ttl: float = 3600.0,
    ):
        from food_agent.config import ProjectConfig
        cfg = ProjectConfig.from_env()

        self.base_url = base_url or cfg.model.base_url
        self.api_key = api_key or cfg.model.api_key
        self.model = model or cfg.model.model_name
        self.max_retries = max_retries
        self._cache = CacheManager(default_ttl=cache_ttl)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def _cache_key(self, prompt: str, image_hash: str = "") -> str:
        raw = f"{self.model}:{prompt}:{image_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a BGR numpy array to base64 JPEG."""
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def call_text(self, prompt: str, system: str = "") -> str:
        """Send a text-only request to the LLM.

        Args:
            prompt: User message.
            system: System message (optional).

        Returns:
            Response text string.
        """
        cache_key = self._cache_key(prompt)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                )
                text = response.choices[0].message.content or ""
                self._cache.put(cache_key, text)
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"API error: {e}"

        return "API error: max retries exceeded"

    def call_vision(
        self, image: np.ndarray, prompt: str, system: str = ""
    ) -> str:
        """Send a vision (image + text) request to the LLM.

        Args:
            image: BGR numpy array (H, W, 3).
            prompt: Text prompt about the image.
            system: System message (optional).

        Returns:
            Response text string.
        """
        img_b64 = self._encode_image(image)
        img_hash = hashlib.md5(img_b64[:100].encode()).hexdigest()
        cache_key = self._cache_key(prompt, img_hash)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        })

        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                )
                text = response.choices[0].message.content or ""
                self._cache.put(cache_key, text)
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"API error: {e}"

        return "API error: max retries exceeded"
