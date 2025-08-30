import asyncio
import os

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Deque, DefaultDict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Available LLM backends"""
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Future: add more as needed


@dataclass
class BackendProfile:
    """Profile of a backend's capabilities and performance"""
    backend: Backend
    models: List[str]
    strengths: List[str]  # e.g., ["reasoning", "code", "vision"]
    weaknesses: List[str]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_context_length: int
    supports_streaming: bool
    reliability_score: float = 0.95  # 0-1, updated based on success rate


@dataclass
class RoutingDecision:
    """A routing decision made by the router"""
    backend: Backend
    model: str
    confidence: float
    reasoning: str
    fallback_chain: List[Backend]
    cache_key: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskProfile:
    """Profile of an incoming task/request"""
    content: str
    intent: Optional[str] = None
    role: Optional[str] = None
    complexity: Optional[str] = None
    estimated_tokens: int = 0
    requires_vision: bool = False
    requires_tools: bool = False
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None


class RouterAgent:
    """Intelligent router agent for LLM backend selection"""

    def __init__(self, storage, lm_studio_url: str = "http://localhost:1234"):
        self.storage = storage
        self.lm_studio_url = lm_studio_url

        # Backend profiles (can be loaded from config)
        self.backend_profiles = {
            Backend.LMSTUDIO: BackendProfile(
                backend=Backend.LMSTUDIO,
                models=["openai/gpt-oss-20b"],
                strengths=["fast", "local", "cost-effective", "coding"],
                weaknesses=["complex-reasoning", "vision"],
                cost_per_1k_tokens=0.0,
                avg_latency_ms=200,
                max_context_length=32768,
                supports_streaming=True
            ),
            Backend.OPENAI: BackendProfile(
                backend=Backend.OPENAI,
                models=["gpt5"],
                strengths=["general", "vision", "tools", "reliability"],
                weaknesses=["cost"],
                cost_per_1k_tokens=0.15,  # gpt-4o-mini
                avg_latency_ms=800,
                max_context_length=128000,
                supports_streaming=True
            ),
            Backend.ANTHROPIC: BackendProfile(
                backend=Backend.ANTHROPIC,
                models=["claude-4-sonnet", "claude-4-opus"],
                strengths=["reasoning", "analysis", "safety", "nuanced-tasks"],
                weaknesses=["cost", "rate-limits"],
                cost_per_1k_tokens=3.0,  # sonnet
                avg_latency_ms=1200,
                max_context_length=200000,
                supports_streaming=True
            )
        }

        # Performance tracking
        self.performance_history = defaultdict(lambda: {
            "successes": 0,
            "failures": 0,
            "total_latency": 0,
            "recent_latencies": deque(maxlen=100)
        })

        # Decision cache (TTL = 5 minutes)
        self.decision_cache: Dict[str, Tuple[RoutingDecision, float]] = {}
        self.cache_ttl = 300  # seconds

        # Specialization mappings
        self.role_specializations = {
            # Security & Safety
            "security": [Backend.ANTHROPIC, Backend.OPENAI],
            "safety": [Backend.ANTHROPIC],
            "compliance": [Backend.ANTHROPIC, Backend.OPENAI],

            # Technical roles
            "coder": [Backend.LMSTUDIO, Backend.OPENAI],
            "debugger": [Backend.LMSTUDIO, Backend.OPENAI],
            "architect": [Backend.ANTHROPIC, Backend.OPENAI],
            "reviewer": [Backend.ANTHROPIC, Backend.LMSTUDIO],

            # Analysis & Research
            "researcher": [Backend.ANTHROPIC, Backend.OPENAI],
            "analyst": [Backend.ANTHROPIC, Backend.OPENAI],
            "scientist": [Backend.ANTHROPIC],

            # Creative & Communication
            "writer": [Backend.ANTHROPIC, Backend.OPENAI],
            "teacher": [Backend.ANTHROPIC, Backend.OPENAI],
            "planner": [Backend.OPENAI, Backend.LMSTUDIO],
        }

    async def analyze_task(self, prompt: str, intent: Optional[str] = None,
                          role: Optional[str] = None) -> TaskProfile:
        """Analyze incoming task to build a profile"""
        # Token estimation (rough)
        estimated_tokens = len(prompt.split()) * 1.3

        # Complexity detection
        complexity_markers = {
            "high": ["theorem", "proof", "architecture", "vulnerability", "formal", "rigorous"],
            "medium": ["analyze", "design", "optimize", "refactor", "review"],
            "low": ["list", "summarize", "explain", "simple", "basic"]
        }

        prompt_lower = prompt.lower()
        detected_complexity = "medium"  # default

        for level, markers in complexity_markers.items():
            if any(marker in prompt_lower for marker in markers):
                detected_complexity = level
                break

        # Vision detection
        requires_vision = any(marker in prompt_lower
                            for marker in ["image", "picture", "visual", "screenshot"])

        # Tool usage detection
        requires_tools = any(marker in prompt_lower
                           for marker in ["search", "browse", "fetch", "api", "database"])

        return TaskProfile(
            content=prompt,
            intent=intent,
            role=role,
            complexity=detected_complexity,
            estimated_tokens=int(estimated_tokens),
            requires_vision=requires_vision,
            requires_tools=requires_tools
        )

    def _check_cache(self, task_hash: str) -> Optional[RoutingDecision]:
        """Check if we have a recent cached decision"""
        if task_hash in self.decision_cache:
            decision, timestamp = self.decision_cache[task_hash]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Router cache hit for {task_hash[:8]}")
                return decision
            else:
                del self.decision_cache[task_hash]
        return None

    def _update_performance(self, backend: Backend, latency_ms: float, success: bool):
        """Update backend performance metrics"""
        stats = self.performance_history[backend]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_latency"] += latency_ms
        stats["recent_latencies"].append(latency_ms)

        # Update reliability score in profile
        total = stats["successes"] + stats["failures"]
        if total > 10:  # Only update after sufficient data
            self.backend_profiles[backend].reliability_score = (
                stats["successes"] / total
            )

    async def decide_routing(self, task: TaskProfile) -> RoutingDecision:
        """Main routing decision logic"""
        # Generate cache key
        cache_key = hashlib.md5(
            f"{task.content[:200]}{task.intent}{task.role}".encode()
        ).hexdigest()

        # Check cache first
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Build candidate backends
        candidates = self._get_candidates(task)

        # Score each candidate
        scores = {}
        for backend in candidates:
            score = self._score_backend(backend, task)
            scores[backend] = score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            # Fallback to LMStudio
            decision = RoutingDecision(
                backend=Backend.LMSTUDIO,
                model="openai/gpt-oss-20b",
                confidence=0.5,
                reasoning="No suitable backends found, defaulting to LMStudio",
                fallback_chain=[Backend.OPENAI, Backend.ANTHROPIC]
            )
        else:
            # Select best backend
            best_backend, best_score = ranked[0]
            profile = self.backend_profiles[best_backend]

            # Build fallback chain from remaining candidates
            fallback_chain = [b for b, _ in ranked[1:]]

            # Determine model based on complexity
            model = self._select_model(best_backend, task)

            decision = RoutingDecision(
                backend=best_backend,
                model=model,
                confidence=min(best_score / 100, 1.0),  # Normalize to 0-1
                reasoning=self._generate_reasoning(best_backend, task, scores),
                fallback_chain=fallback_chain,
                cache_key=cache_key
            )

        # Cache decision
        self.decision_cache[cache_key] = (decision, time.time())

        # Log decision for analytics
        await self._log_decision(decision, task)

        return decision

    def _get_candidates(self, task: TaskProfile) -> List[Backend]:
        """Get candidate backends based on task requirements"""
        candidates = []

        # Check availability
        available = {
            Backend.LMSTUDIO: True,  # Always available
            Backend.OPENAI: bool(os.getenv("OPENAI_API_KEY")),
            Backend.ANTHROPIC: bool(os.getenv("ANTHROPIC_API_KEY"))
        }

        # Role-based filtering
        if task.role:
            role_key = task.role.lower().split()[0]
            if role_key in self.role_specializations:
                for backend in self.role_specializations[role_key]:
                    if available.get(backend, False):
                        candidates.append(backend)

        # Add all available backends if no role match
        if not candidates:
            candidates = [b for b, avail in available.items() if avail]

        # Filter by requirements
        if task.requires_vision:
            # Only OpenAI and Anthropic support vision currently
            candidates = [c for c in candidates
                         if c in [Backend.OPENAI, Backend.ANTHROPIC]]

        return candidates

    def _score_backend(self, backend: Backend, task: TaskProfile) -> float:
        """Score a backend for a given task (0-100)"""
        profile = self.backend_profiles[backend]
        score = 50.0  # Base score

        # Complexity matching
        complexity_bonus = {
            "high": {Backend.ANTHROPIC: 30, Backend.OPENAI: 20, Backend.LMSTUDIO: 5},
            "medium": {Backend.OPENAI: 20, Backend.ANTHROPIC: 15, Backend.LMSTUDIO: 15},
            "low": {Backend.LMSTUDIO: 30, Backend.OPENAI: 10, Backend.ANTHROPIC: 5}
        }
        score += complexity_bonus.get(task.complexity, {}).get(backend, 0)

        # Reliability bonus
        score += profile.reliability_score * 20

        # Latency penalty (if requirement specified)
        if task.max_latency_ms:
            if profile.avg_latency_ms > task.max_latency_ms:
                score -= 30

        # Cost penalty (if requirement specified)
        if task.max_cost:
            estimated_cost = (task.estimated_tokens / 1000) * profile.cost_per_1k_tokens
            if estimated_cost > task.max_cost:
                score -= 40

        # Intent/specialization bonus
        if task.intent:
            intent_lower = task.intent.lower()
            for strength in profile.strengths:
                if strength in intent_lower:
                    score += 15

        # Context length check
        if task.estimated_tokens > profile.max_context_length:
            score -= 50

        return max(0, score)

    def _select_model(self, backend: Backend, task: TaskProfile) -> str:
        """Select specific model within a backend"""
        profile = self.backend_profiles[backend]

        if backend == Backend.LMSTUDIO:
            return "openai/gpt-oss-20b"

        elif backend == Backend.OPENAI:
            # Use o1 for complex reasoning, gpt-4o for vision, else mini
            return "gpt5"

        elif backend == Backend.ANTHROPIC:
            # Use opus for highest complexity, else sonnet
            if task.complexity == "high":
                return "claude-4-opus"
            return "claude-4-sonnet"

        # Default to first available
        return profile.models[0]

    def _generate_reasoning(self, backend: Backend, task: TaskProfile,
                          scores: Dict[Backend, float]) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []

        # Explain why this backend was chosen
        if task.role:
            reasons.append(f"Role '{task.role}' specialization")
        if task.complexity == "high":
            reasons.append("High complexity task requiring advanced reasoning")
        elif task.complexity == "low":
            reasons.append("Simple task suitable for fast local processing")

        # Performance consideration
        profile = self.backend_profiles[backend]
        if profile.reliability_score > 0.9:
            reasons.append(f"High reliability ({profile.reliability_score:.0%})")

        # Show score differential
        if len(scores) > 1:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) > 1:
                margin = sorted_scores[0][1] - sorted_scores[1][1]
                if margin > 20:
                    reasons.append(f"Clear winner (margin: {margin:.0f})")
                else:
                    reasons.append(f"Close decision (margin: {margin:.0f})")

        return "; ".join(reasons) if reasons else "Default selection"

    async def _log_decision(self, decision: RoutingDecision, task: TaskProfile):
        """Log routing decision for analytics"""
        try:
            log_entry = {
                "timestamp": decision.timestamp,
                "backend": decision.backend.value,
                "model": decision.model,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "task_complexity": task.complexity,
                "task_role": task.role,
                "task_intent": task.intent,
                "estimated_tokens": task.estimated_tokens
            }

            # Store in memory for analytics
            self.storage.store_memory(
                key=f"routing_{int(decision.timestamp)}",
                value=json.dumps(log_entry),
                category="router_decisions"
            )
        except Exception as e:
            logger.warning(f"Failed to log routing decision: {e}")

    async def get_router_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get analytics on router performance"""
        cutoff = time.time() - (hours * 3600)

        # Retrieve recent decisions
        decisions = self.storage.retrieve_memory(
            category="router_decisions",
            limit=1000
        )

        # Filter by time
        recent = []
        for mem in decisions:
            try:
                data = json.loads(mem["value"])
                if data["timestamp"] > cutoff:
                    recent.append(data)
            except:
                continue

        # Compute analytics
        backend_counts = defaultdict(int)
        model_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        avg_confidence = 0

        for decision in recent:
            backend_counts[decision["backend"]] += 1
            model_counts[decision["model"]] += 1
            complexity_counts[decision.get("task_complexity", "unknown")] += 1
            avg_confidence += decision["confidence"]

        # Backend performance
        backend_perf = {}
        for backend in Backend:
            stats = self.performance_history[backend]
            if stats["recent_latencies"]:
                avg_latency = sum(stats["recent_latencies"]) / len(stats["recent_latencies"])
            else:
                avg_latency = 0

            total = stats["successes"] + stats["failures"]
            success_rate = stats["successes"] / total if total > 0 else 0

            backend_perf[backend.value] = {
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "total_requests": total
            }

        return {
            "total_decisions": len(recent),
            "avg_confidence": avg_confidence / len(recent) if recent else 0,
            "backend_distribution": dict(backend_counts),
            "model_distribution": dict(model_counts),
            "complexity_distribution": dict(complexity_counts),
            "backend_performance": backend_perf,
            "cache_size": len(self.decision_cache),
            "hours_analyzed": hours
        }


# Integration with main server
async def create_router_agent(server) -> RouterAgent:
    """Create and initialize router agent"""
    router = RouterAgent(server.storage)
    logger.info("Router agent initialized with %d backends", len(router.backend_profiles))
    return router