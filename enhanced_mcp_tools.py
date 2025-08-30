from __future__ import annotations

from typing import Any, Dict, List


def merged_tools(existing: Dict[str, Any]) -> Dict[str, Any]:
    """Return a merged tool registry compatible with MCP tools/list response.
    Accepts an existing schema dict from server.get_all_tools() and returns a new one.
    Keeping this function small and explicit maintains backward compatibility.
    """
    # For now, we simply return existing. Hooks for future expansion can insert
    # or modify entries while preserving the result shape.
    return existing

