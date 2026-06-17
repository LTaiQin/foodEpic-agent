"""Simple in-memory cache with TTL support."""

import time
from typing import Any, Optional


class CacheManager:
    """Key-value cache with optional TTL (time-to-live) in seconds."""

    def __init__(self, default_ttl: Optional[float] = None):
        self._store: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing/expired."""
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if expiry and time.monotonic() > expiry:
            del self._store[key]
            return None
        return value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value with optional TTL override."""
        t = ttl if ttl is not None else self._default_ttl
        expiry = time.monotonic() + t if t else 0.0
        self._store[key] = (value, expiry)

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
