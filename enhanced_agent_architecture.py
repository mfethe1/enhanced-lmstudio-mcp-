from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass(frozen=True)
class BackendChoice:
    backend: str  # "lmstudio" | "openai" | "anthropic"
    reason: str


def available_backends() -> dict[str, bool]:
    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        # LM Studio exposes OpenAI-compatible base; allow LMSTUDIO_ or OPENAI_ base as local
        "lmstudio": bool(os.getenv("LMSTUDIO_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("LM_STUDIO_URL")),
    }


def classify_task_complexity(text: str) -> ComplexityLevel:
    """Heuristic multi-tier classification.
    - Length and keywords drive the class; conservative upward bias when unclear.
    """
    if not isinstance(text, str) or not text.strip():
        return ComplexityLevel.SIMPLE

    t = text.lower()
    length_score = min(len(t) // 200, 5)  # coarse proxy
    keywords_expert = [
        "formal proof", "rlhf", "distributed systems", "concurrency", "race condition",
        "security audit", "cryptograph", "post-quantum", "complexity analysis", "compiler",
        "multi-agent", "mcp protocol", "observability", "zero-downtime migration",
    ]
    keywords_complex = [
        "refactor architecture", "performance tuning", "vector db", "retrieval",
        "kubernetes", "terraform", "oauth", "sso", "websocket", "protocol",
        "parallel", "asyncio", "playwright", "end-to-end", "multi-file",
    ]
    keywords_simple = ["typo", "rename", "docs", "comment", "format", "lint"]

    if any(k in t for k in keywords_expert) or length_score >= 4:
        return ComplexityLevel.EXPERT
    if any(k in t for k in keywords_complex) or length_score >= 2:
        return ComplexityLevel.COMPLEX
    if any(k in t for k in keywords_simple) and length_score == 0:
        return ComplexityLevel.SIMPLE
    return ComplexityLevel.MODERATE


def _parse_overrides(raw: str) -> dict[str, str]:
    # Format examples: "all=openai", "security=anthropic,planner=lmstudio"
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        if not part.strip():
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out


def decide_backend(role: str, task_desc: str) -> BackendChoice:
    """Central routing rule-set.
    - Honors AGENT_BACKEND_OVERRIDE if present
    - Routes by complexity and role
    - Falls back to available providers; default is lmstudio
    """
    role_key = (role or "").strip().lower()
    overrides = _parse_overrides(os.getenv("AGENT_BACKEND_OVERRIDE", ""))
    if overrides:
        if "all" in overrides:
            return BackendChoice(overrides["all"], "override:all")
        if role_key in overrides:
            return BackendChoice(overrides[role_key], f"override:{role_key}")

    comp = classify_task_complexity(task_desc)
    avail = available_backends()

    def pick(preferred: list[str], reason: str) -> BackendChoice:
        for b in preferred:
            if avail.get(b):
                return BackendChoice(b, reason)
        # If none available, default to lmstudio as safe local fallback
        return BackendChoice("lmstudio", reason + "+fallback:lmstudio")

    # Security, advanced reasoning -> prefer Anthropic, then OpenAI
    if role_key in {"security reviewer", "security", "architect"} or comp in {ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT}:
        return pick(["anthropic", "openai", "lmstudio"], f"role:{role_key}|comp:{comp}")

    # Reviewer/coder/planner defaults to local for speed/cost
    return pick(["lmstudio", "openai", "anthropic"], f"role:{role_key}|comp:{comp}")


__all__ = [
    "ComplexityLevel",
    "BackendChoice",
    "available_backends",
    "classify_task_complexity",
    "decide_backend",
]

