from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional
from collections import deque
import random


@dataclass
class Decision:
    context: Dict[str, Any]
    action: str
    confidence: float
    outcome: Optional[float] = None


class AdaptiveAgent:
    """Simple performance-aware decision helper per role.

    This is intentionally lightweight; it can be swapped out for a more
    sophisticated learner later while keeping the same interface.
    """

    def __init__(self, role: str, learning_rate: float = 0.1) -> None:
        self.role = role
        self.learning_rate = learning_rate
        self.decision_history: Deque[Decision] = deque(maxlen=1000)
        self.success_bias = 0.5  # baseline success estimate

    async def make_informed_decision(self, context: Dict[str, Any], options: List[str]) -> Decision:
        # Naive scoring using recent success bias
        scores = {opt: self.success_bias for opt in options}
        best = max(scores, key=scores.get)
        conf = scores[best]
        # epsilon-greedy exploration
        if random.random() < 0.1 and len(options) > 1:
            best = random.choice([o for o in options if o != best])
            conf *= 0.5
        d = Decision(context=context, action=best, confidence=conf)
        self.decision_history.append(d)
        return d

    async def update_from_outcome(self, decision: Decision, outcome: float) -> None:
        decision.outcome = outcome
        # Adjust bias toward outcome
        self.success_bias = (1 - self.learning_rate) * self.success_bias + self.learning_rate * outcome

