from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Simple tool registry mapping names to handlers and metadata.

    Handlers take (arguments: dict, server?) and can be sync or async.

    Enhanced with duplicate detection for Claude Sonnet 4.5 compatibility.
    """

    def __init__(self) -> None:
        # name -> (handler, needs_server)
        self._handlers: Dict[str, tuple[Callable[..., Any], bool]] = {}
        self._duplicate_warnings: set[str] = set()

    def register(self, name: str, handler: Callable[..., Any], *, needs_server: bool = False) -> None:
        """Register a tool handler with duplicate detection.

        Args:
            name: Unique tool name
            handler: Callable handler function
            needs_server: Whether handler requires server instance

        Raises:
            ValueError: If name is invalid

        Warns:
            If a tool with the same name is already registered (overwrites existing)
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string")

        # Detect and warn about duplicates (Claude Sonnet 4.5 compatibility)
        if name in self._handlers and name not in self._duplicate_warnings:
            logger.warning(
                f"Duplicate tool registration detected: '{name}'. "
                f"This may cause errors with Claude Sonnet 4.5. "
                f"Previous handler will be overwritten."
            )
            self._duplicate_warnings.add(name)

        self._handlers[name] = (handler, needs_server)

    def get_handler(self, name: str) -> Optional[tuple[Callable[..., Any], bool]]:
        return self._handlers.get(name)

    def list_tools(self) -> Dict[str, Callable[..., Any]]:
        return {k: v[0] for k, v in self._handlers.items()}

    def has_duplicates(self) -> bool:
        """Check if any duplicate registrations were detected."""
        return len(self._duplicate_warnings) > 0

    def get_duplicate_names(self) -> list[str]:
        """Get list of tool names that had duplicate registrations."""
        return sorted(self._duplicate_warnings)

    def get_all_tool_schemas(self) -> list[dict]:
        """Placeholder for compatibility. Returns empty list until wired to schemas."""
        return []

