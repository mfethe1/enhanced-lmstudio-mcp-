from __future__ import annotations

import time
from typing import Any, Dict, Optional


class ContextManager:
    """Lightweight context enrichment and scratchpad store.

    This module can later integrate with enhanced_mcp_storage_v2 and semantic memory.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._store: Dict[str, Any] = {}

    async def enrich_tool_context(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(args)
        # Example: attach a timestamp and previous result if present
        enriched["_ts"] = int(time.time())
        last_key = f"last_result:{tool_name}"
        if last_key in self._store:
            enriched["_last_result"] = self._store[last_key]
        return enriched

    async def add_tool_result(self, tool_name: str, result: Any) -> None:
        self._store[f"last_result:{tool_name}"] = result

