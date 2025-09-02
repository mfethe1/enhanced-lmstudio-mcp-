from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable

from .router import IntelligentRouter
from .registry import ToolRegistry
from .context_manager import ContextManager

logger = logging.getLogger(__name__)


class EnhancedMCPServer:
    """Production-ready orchestrator for MCP tool execution.

    This class presents a minimal, clean interface while delegating real work to
    dedicated subsystems (router, registry, context manager). It is designed to be
    embedded inside the existing monolithic server until full migration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.router = IntelligentRouter(self.config)
        self.registry = ToolRegistry()
        self.context = ContextManager(self.config)
        # Simple in-memory cache; pluggable later
        self._cache: Dict[str, Any] = {}

    # --- Public API ---
    async def handle_tool_call(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Unified tool execution with context enrichment and simple caching.

        Message shape (subset):
        { "params": { "name": str, "arguments": dict } }
        """
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name:
            return self._error(-32602, "Missing tool name")

        # Context enrichment
        enriched_args = await self.context.enrich_tool_context(tool_name, arguments)

        # Cache check
        cache_key = self._cache_key(tool_name, enriched_args)
        if cache_key in self._cache:
            logger.debug("Cache hit for %s", cache_key)
            return {"result": self._cache[cache_key]}

        # Resolve handler
        handler = self.registry.get_handler(tool_name)
        if handler is None:
            return self._error(-32601, f"Unknown tool: {tool_name}")

        # Execute
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(enriched_args, self)
            else:
                # Allow sync handlers
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, enriched_args, self)
        except Exception as e:  # surface as standard error
            logger.exception("Tool %s failed", tool_name)
            return self._error(-32000, f"Tool execution failed: {e}")

        # Cache store
        self._cache[cache_key] = result
        return {"result": result}

    # --- Utilities ---
    def _cache_key(self, name: str, args: Dict[str, Any]) -> str:
        try:
            return f"{name}:{json.dumps(args, sort_keys=True)[:512]}"
        except Exception:
            return f"{name}:{str(args)[:512]}"

    @staticmethod
    def _error(code: int, message: str) -> Dict[str, Any]:
        return {"error": {"code": code, "message": message}}

