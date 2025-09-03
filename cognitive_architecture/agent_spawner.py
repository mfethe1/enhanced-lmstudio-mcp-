"""
Dynamic Agent Spawning System for Advanced MCP Cognitive Architecture

This module integrates with server.py patterns (LLM routing, storage, retry) and provides:
- Capability analysis for tasks
- On-demand agent spawning/reuse with pooling and lifecycle
- Coordination strategies (sequential, parallel, hierarchical, consensus-ready)
- Performance tracking to inform future routing decisions

Safe-by-default: optional numpy; pure-Python fallback if unavailable.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import numpy as _np  # type: ignore
except Exception:  # fallback to minimal vector ops
    _np = None


class AgentCapability(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    RESEARCH = "research"


@dataclass
class AgentProfile:
    agent_id: str
    capabilities: Set[AgentCapability]
    expertise_scores: Dict[str, float]
    active_tasks: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    created_at: float = 0.0
    last_active: float = 0.0
    llm_backend: str = "lmstudio"


class DynamicAgentSpawner:
    """Dynamic agent spawning with capability matching and load-balancing."""

    def __init__(self, server: Any, max_agents: int = 20) -> None:
        self.server = server
        self.max_agents = max_agents
        self.active_agents: Dict[str, AgentProfile] = {}
        self.agent_pool: List[AgentProfile] = []
        self.capability_matrix = self._initialize_capability_matrix()
        self.task_queue: asyncio.Queue[str] = asyncio.Queue()
        self.performance_tracker: Dict[str, List[float]] = {}

    # --------------------- Capability Analysis ---------------------
    def _initialize_capability_matrix(self):
        if _np is not None:
            return _np.random.rand(len(AgentCapability), 100)
        # Pure Python fallback: list of lists
        import random as _r
        return [[_r.random() for _ in range(100)] for _ in range(len(AgentCapability))]

    def _extract_task_features(self, task_description: str):
        indices_map = {
            "refactor": [0, 1, 2],
            "optimize": [3, 4, 5],
            "test": [6, 7, 8],
            "security": [9, 10, 11],
            "debug": [12, 13, 14],
            "implement": [15, 16, 17],
            "review": [18, 19, 20],
            "research": [21, 22, 23],
            "document": [24, 25, 26],
            "design": [27, 28, 29],
        }
        feats = [0.0] * 100
        t = (task_description or "").lower()
        for k, idxs in indices_map.items():
            if k in t:
                for i in idxs:
                    feats[i] = 1.0
        if _np is not None:
            return _np.array(feats)
        return feats

    async def analyze_task_requirements(self, task_description: str, context: Dict[str, Any]) -> Set[AgentCapability]:
        feats = self._extract_task_features(task_description)
        # dot product
        if _np is not None:
            scores = _np.dot(self.capability_matrix, feats)  # type: ignore
            scores_list = [float(s) for s in list(scores)]
        else:
            scores_list: List[float] = []
            for row in self.capability_matrix:
                s = sum(a * b for a, b in zip(row, feats))
                scores_list.append(float(s))
        threshold = 0.6
        req: Set[AgentCapability] = set()
        caps = list(AgentCapability)
        for i, sc in enumerate(scores_list):
            if sc > threshold:
                req.add(caps[i])
        # Context boosts
        if "security" in (context.get("tags") or []):
            req.add(AgentCapability.SECURITY)
        if "performance" in (context.get("concerns") or []):
            req.add(AgentCapability.PERFORMANCE)
        if not req:
            req.add(AgentCapability.CODE_GENERATION)
        return req

    # --------------------- Lifecycle & Pooling ---------------------
    async def spawn_or_retrieve_agent(self, capabilities: Set[AgentCapability], task_id: str) -> AgentProfile:
        # Try to find a free matching agent
        best: Optional[AgentProfile] = None
        best_score = 0.0
        need = len(capabilities) or 1
        for ag in self.agent_pool:
            if ag.active_tasks:
                continue
            score = len(ag.capabilities.intersection(capabilities)) / float(need)
            if score > best_score:
                best = ag
                best_score = score
        if best and best_score > 0.8:
            best.active_tasks.append(task_id)
            best.last_active = time.time()
            return best
        # Spawn new if capacity
        if len(self.active_agents) < self.max_agents:
            na = await self._spawn_agent(capabilities)
            na.active_tasks.append(task_id)
            self.active_agents[na.agent_id] = na
            self.agent_pool.append(na)
            return na
        # Queue: for simplicity, reuse least-loaded agent
        least = min(self.agent_pool, key=lambda a: len(a.active_tasks)) if self.agent_pool else None
        if least:
            least.active_tasks.append(task_id)
            return least
        # As a last resort, spawn
        return await self._spawn_agent(capabilities)

    async def _spawn_agent(self, capabilities: Set[AgentCapability]) -> AgentProfile:
        aid = hashlib.md5(f"{capabilities}|{time.time()}".encode()).hexdigest()[:12]
        expertise = {cap.value: 0.7 for cap in capabilities}
        profile = AgentProfile(
            agent_id=aid,
            capabilities=capabilities,
            expertise_scores=expertise,
            created_at=time.time(),
            llm_backend=self._select_backend(capabilities),
        )
        try:
            self.server.storage.store_memory(
                key=f"agent_spawn_{aid}",
                value=json.dumps({"capabilities": [c.value for c in capabilities], "ts": time.time()}),
                category="agents",
            )
        except Exception:
            pass
        return profile

    def _select_backend(self, capabilities: Set[AgentCapability]) -> str:
        # Heuristic backend selection; integrates with router if present
        if AgentCapability.SECURITY in capabilities:
            return "anthropic"
        if AgentCapability.PERFORMANCE in capabilities:
            return "openai"
        return "lmstudio"

    # --------------------- Execution ---------------------
    async def _execute_agent_task(self, agent: AgentProfile, task: str) -> Any:
        prompt = (
            f"Role: {','.join([c.value for c in agent.capabilities])}\n"
            f"Backend: {agent.llm_backend}\n"
            f"Task: {task}"
        )
        # Use production optimizer cache if available to reduce tokens
        try:
            if getattr(self.server, "production", None) is not None:
                out = await self.server.production.cached_llm(prompt, temperature=0.2, intent="agent_task", role=agent.llm_backend)
            else:
                out = await self.server.make_llm_request_with_retry(prompt, temperature=0.2)
        except Exception as e:
            out = f"error: {e}"
        agent.last_active = time.time()
        return out

    def _merge_results(self, results: List[Any]) -> Dict[str, Any]:
        texts = [str(r) for r in results]
        return {"combined": "\n\n".join(texts)[:8000], "parts": texts}

    async def execute_with_agents(self, task: str, agents: List[AgentProfile], coordination_strategy: str = "sequential") -> Dict[str, Any]:
        if not agents:
            return {"error": "no agents"}
        if coordination_strategy == "parallel":
            res = await asyncio.gather(*[self._execute_agent_task(a, task) for a in agents])
            return self._merge_results(res)
        if coordination_strategy == "hierarchical":
            lead = agents[0]
            plan = await self._execute_agent_task(lead, f"Break down into {len(agents)-1} subtasks: {task}")
            # naive split
            subs = [f"subtask_{i}" for i in range(max(0, len(agents)-1))]
            sub_results = []
            for i, a in enumerate(agents[1:]):
                st = subs[i] if i < len(subs) else task
                sub_results.append(await self._execute_agent_task(a, f"{st}: {task}"))
            synth = await self._execute_agent_task(lead, f"Synthesize results: {sub_results}")
            return {"plan": plan, "synthesis": synth}
        # sequential
        prev = None
        for a in agents:
            prev = await self._execute_agent_task(a, f"{task}\nPrevious: {prev}" if prev else task)
        return {"result": prev}


# --------------------- Integration helpers ---------------------
_spawner_singleton: Dict[int, DynamicAgentSpawner] = {}


def get_spawner(server: Any) -> DynamicAgentSpawner:
    sid = id(server)
    if sid not in _spawner_singleton:
        _spawner_singleton[sid] = DynamicAgentSpawner(server)
    return _spawner_singleton[sid]


def integrate_agent_spawner(server: Any) -> None:
    try:
        if getattr(server, "agent_spawner", None) is None:
            server.agent_spawner = get_spawner(server)
    except Exception:
        pass

