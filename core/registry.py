from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class ToolRegistry:
    """Simple tool registry mapping names to handlers and metadata.

    Handlers take (arguments: dict, server?) and can be sync or async.
    """

    def __init__(self) -> None:
        # name -> (handler, needs_server)
        self._handlers: Dict[str, tuple[Callable[..., Any], bool]] = {}

    def register(self, name: str, handler: Callable[..., Any], *, needs_server: bool = False) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string")
        self._handlers[name] = (handler, needs_server)

    def get_handler(self, name: str) -> Optional[tuple[Callable[..., Any], bool]]:
        return self._handlers.get(name)

    def list_tools(self) -> Dict[str, Callable[..., Any]]:
        return {k: v[0] for k, v in self._handlers.items()}

    def get_all_tool_schemas(self) -> list[dict]:
        """Placeholder for compatibility. Returns empty list until wired to schemas."""
        return []

