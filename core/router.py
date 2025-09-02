from __future__ import annotations

from typing import Any, Dict, Optional


class IntelligentRouter:
    """Minimal placeholder for routing logic.

    In the monolith this chooses between LM Studio, OpenAI, Anthropic, etc.
    Here we keep the API stable so we can plug it in later without breaking callers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def select_backend(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        # Simple heuristic: default to 'lmstudio'
        return "lmstudio"

    def score_confidence(self, task: str, backend: str) -> float:
        # Stub confidence logic for now
        return 0.75

