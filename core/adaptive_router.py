from __future__ import annotations

import math
import statistics
import time
from collections import deque, defaultdict
from typing import Any, Deque, Dict, List, Optional, Tuple


class AdaptiveRouter:
    """Lightweight ML-inspired router with online learning and performance tracking.

    - Tracks latency distributions (P50/P99) per backend and per tool
    - Tracks success rates to compute confidence
    - Chooses tools using simple keyword/complexity signals with feedback from artifacts
    """

    def __init__(self, window_size: int = 200) -> None:
        self.window_size = window_size
        self.latency_ms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.successes: Dict[str, int] = defaultdict(int)
        self.calls: Dict[str, int] = defaultdict(int)
        self.tool_successes: Dict[str, int] = defaultdict(int)
        self.tool_calls: Dict[str, int] = defaultdict(int)
        self.artifact_bias: Dict[str, float] = defaultdict(float)  # e.g., push towards certain tools

    # --- tracking ---
    def record_result(self, key: str, latency_ms: float, success: bool) -> None:
        self.latency_ms[key].append(float(latency_ms))
        self.calls[key] += 1
        if success:
            self.successes[key] += 1

    def record_tool_result(self, tool: str, latency_ms: float, success: bool) -> None:
        k = f"tool::{tool}"
        self.record_result(k, latency_ms, success)
        self.tool_calls[tool] += 1
        if success:
            self.tool_successes[tool] += 1

    # --- metrics ---
    def percentile(self, key: str, p: float) -> Optional[float]:
        arr = list(self.latency_ms.get(key, []))
        if not arr:
            return None
        arr_sorted = sorted(arr)
        idx = min(len(arr_sorted)-1, max(0, int(round((p/100.0) * (len(arr_sorted)-1)))))
        return arr_sorted[idx]

    def p50(self, key: str) -> Optional[float]:
        return self.percentile(key, 50)

    def p99(self, key: str) -> Optional[float]:
        return self.percentile(key, 99)

    def conf(self, key: str) -> float:
        n = self.calls.get(key, 0)
        if n == 0:
            return 0.0
        return self.successes.get(key, 0) / float(n)

    def tool_conf(self, tool: str) -> float:
        n = self.tool_calls.get(tool, 0)
        if n == 0:
            return 0.0
        return self.tool_successes.get(tool, 0) / float(n)

    # --- selection ---
    def analyze_complexity(self, instruction: str, context: str) -> float:
        txt = (instruction + "\n" + context).lower()
        score = 0.0
        for kw in ("optimiz", "performance", "concurrent", "async", "security", "deep research", "benchmark", "profile"):
            if kw in txt:
                score += 1.0
        score += min(3.0, len(txt) / 2000.0)  # coarse length prior
        return score

    def choose_tool(self, instruction: str, context: str, tool_names: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
        t = instruction.lower()
        # Heuristic seeding + learned artifact bias
        if any(k in t for k in ["refactor", "cleanup", "modular", "restructure"]):
            seed = "agent_team_refactor"
        elif any(k in t for k in ["test", "pytest", "unit test", "coverage", "assert"]):
            seed = "agent_team_review_and_test"
        elif any(k in t for k in ["research", "survey", "compare", "latest", "find papers", "deep research"]):
            seed = "deep_research"
        else:
            seed = "agent_team_plan_and_code"
        # Combine with artifact bias
        bias_tool = max(tool_names, key=lambda name: self.artifact_bias.get(name, 0.0)) if tool_names else None
        if bias_tool and self.artifact_bias.get(bias_tool, 0.0) > 0.5:
            seed = bias_tool
        conf = min(0.9, 0.6 + 0.2 * self.tool_conf(seed))
        return (seed if seed in tool_names else (tool_names[0] if tool_names else None), {"confidence": conf, "rationale": f"seeded:{seed}"})

    def choose_backend(self, complexity: float) -> str:
        # Simple policy: low complexity -> LM Studio, medium -> OpenAI, high -> Anthropic
        if complexity < 1.5:
            return "lmstudio"
        if complexity < 3.0:
            return "openai"
        return "anthropic"

    def learn_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        # Very simple update: if performance_tuner produced artifacts, increase bias to plan_and_code (execution focus)
        if "performance_tuner" in artifacts:
            self.artifact_bias["agent_team_plan_and_code"] += 0.1
        if "security_auditor" in artifacts:
            self.artifact_bias["agent_team_review_and_test"] += 0.1
        # keep values bounded
        for k in list(self.artifact_bias.keys()):
            self.artifact_bias[k] = float(min(2.0, max(-2.0, self.artifact_bias[k])))


# Global singleton
adaptive_router = AdaptiveRouter(window_size=int(float(__import__('os').getenv('ROUTER_METRICS_WINDOW', '200'))))

