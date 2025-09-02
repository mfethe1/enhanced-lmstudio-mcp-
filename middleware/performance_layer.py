from __future__ import annotations

import asyncio
import json
import hashlib
from typing import Any, Dict, Optional


class MemoryCache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class PerformanceLayer:
    """Lightweight performance helpers: cache, dedup, simple batching stub."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.cache = MemoryCache()
        self._pending: Dict[str, asyncio.Future] = {}

    def generate_cache_key(self, name: str, request: Dict[str, Any]) -> str:
        try:
            payload = json.dumps({"name": name, "request": request}, sort_keys=True)
        except Exception:
            payload = f"{name}:{str(request)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def deduplicate(self, key: str, coro_factory):
        # Simple in-flight de-duplication: share same future if a request is ongoing
        fut = self._pending.get(key)
        if fut is not None:
            return await fut
        fut = asyncio.ensure_future(coro_factory())
        self._pending[key] = fut
        try:
            return await fut
        finally:
            self._pending.pop(key, None)

