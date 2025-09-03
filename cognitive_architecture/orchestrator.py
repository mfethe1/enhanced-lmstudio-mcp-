"""
Cognitive Orchestrator: Cross-component integration pipeline (MCTS + Knowledge Graph + Agent Spawner + Validation + Coder)

High-level flow:
1) Build knowledge context for the task
2) Bias MCTS planning with knowledge graph (patterns/similarities)
3) Spawn specialized agents based on planned actions
4) Execute agents to gather insights
5) One-shot code synthesis (cognitive_coder)
6) Validate artifacts with validation_engine
7) Record artifacts/decisions into knowledge graph
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from cognitive_architecture.mcts_planner import MCTSCodePlanner
from cognitive_architecture.knowledge_graph import CodeKnowledgeGraph
from cognitive_architecture.agent_spawner import get_spawner, AgentCapability

try:
    from validation_engine import ContinuousValidationLoop
except Exception:  # optional
    ContinuousValidationLoop = None  # type: ignore

try:
    from cognitive_coder import OneShotCognitiveCoder
except Exception:
    OneShotCognitiveCoder = None  # type: ignore


def _derive_required_caps(actions: List[str]) -> List[AgentCapability]:
    caps: List[AgentCapability] = []
    s = " ".join(actions).lower()
    if any(k in s for k in ["optimize", "profile", "bottleneck"]):
        caps.append(AgentCapability.PERFORMANCE)
    if any(k in s for k in ["test", "fixture"]):
        caps.append(AgentCapability.TESTING)
    if any(k in s for k in ["security", "auth", "policy"]):
        caps.append(AgentCapability.SECURITY)
    if any(k in s for k in ["refactor", "design", "interfaces"]):
        caps.append(AgentCapability.REFACTORING)
        caps.append(AgentCapability.ARCHITECTURE)
    if AgentCapability.CODE_GENERATION not in caps:
        caps.append(AgentCapability.CODE_GENERATION)
    return list(dict.fromkeys(caps))


async def orchestrate_task_async(server: Any, task: str, constraints: Optional[Dict[str, Any]] = None, initial_code: str = "", strategy: str = "hierarchical", max_iterations: int = 400) -> Dict[str, Any]:
    constraints = constraints or {}

    # 1) Knowledge context
    kg = CodeKnowledgeGraph(server.storage)
    ctx_items = kg.get_context_for_generation(task, max_context_items=5)
    # Include context in shaped prompt if optimizer is present
    context_snippets = []
    for item in ctx_items:
        if item.get("code"):
            context_snippets.append(item["code"][:400])
    if getattr(server, "production", None) is not None:
        shaped_task = server.production.augment.shape_prompt(task, context_snippets)
    else:
        shaped_task = task + ("\n\n" + "\n\n".join(context_snippets) if context_snippets else "")

    # 2) MCTS planning with bias: override initial actions to include pattern-driven actions
    planner = MCTSCodePlanner(server, max_iterations=max_iterations)
    base_get_init = planner._get_initial_actions

    def _biased_initial_actions(state: Dict[str, Any]) -> List[str]:
        acts = base_get_init(state)
        # bias for async if patterns include async
        if any(ci.get("relevance_type") == "pattern_example" and ci.get("pattern") == "async" for ci in ctx_items):
            acts.append("add_async_function")
        if any("factory" in (ci.get("pattern") or "") for ci in ctx_items):
            acts.append("define_interfaces")
        # dedupe
        return list(dict.fromkeys(acts))

    planner._get_initial_actions = _biased_initial_actions  # type: ignore

    plan = await planner.plan_solution(shaped_task, constraints, initial_code)

    # 3) Spawn agents based on plan
    spawner = get_spawner(server)
    required_caps = _derive_required_caps(plan.get("action_sequence", []))
    # Use spawner capability analysis too
    req_set = set(required_caps)
    req_set |= await spawner.analyze_task_requirements(task, {"tags": [], "concerns": []})

    agents = []
    for i in range(max(2, min(5, len(req_set) or 2))):
        ag = await spawner.spawn_or_retrieve_agent(req_set, task_id=f"{task[:24]}_{i}")
        agents.append(ag)

    # 4) Execute agents to synthesize guidance/partial outputs
    agent_exec = await spawner.execute_with_agents(task, agents, coordination_strategy=strategy)

    # 5) One-shot code synthesis
    artifacts = {"files": {}, "tests": {}, "docs": ""}
    if OneShotCognitiveCoder is not None:
        coder = OneShotCognitiveCoder(server, getattr(server, "storage", None))
        spec = f"{task}\n\nPlan: {json.dumps(plan)}\n\nContext: {'; '.join(context_snippets[:3])}"
        synth = await coder.synthesize(specification=spec, module_path="auto_tool.py")
        artifacts = {"files": synth.code_files, "tests": synth.test_files, "docs": synth.docs}

    # 6) Validation
    validation_report = None
    if ContinuousValidationLoop is not None:
        loop = ContinuousValidationLoop(server)
        changed_paths = list(artifacts.get("files", {}).keys()) + list(artifacts.get("tests", {}).keys())
        validation_report = loop.run_full_validation(changed_paths) if changed_paths else loop.realtime_validate_snippet("# no files")

    # 7) Record artifacts into KG
    try:
        for path, code in artifacts.get("files", {}).items():
            kg.add_code_artifact(code, "module", {"path": path, "source": "orchestrate"})
        for path, code in artifacts.get("tests", {}).items():
            kg.add_code_artifact(code, "test", {"path": path, "source": "orchestrate"})
    except Exception:
        pass

    return {
        "plan": plan,
        "agents": [{"agent_id": a.agent_id, "backend": a.llm_backend, "caps": [c.value for c in a.capabilities]} for a in agents],
        "agent_outputs": agent_exec,
        "artifacts": artifacts,
        "validation": getattr(validation_report, "__dict__", validation_report) if validation_report else None,
        "context_used": ctx_items,
    }


def orchestrate_task(server: Any, task: str, constraints: Optional[Dict[str, Any]] = None, initial_code: str = "", strategy: str = "hierarchical", max_iterations: int = 400) -> Dict[str, Any]:
    try:
        try:
            return asyncio.get_event_loop().run_until_complete(orchestrate_task_async(server, task, constraints, initial_code, strategy, max_iterations))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(orchestrate_task_async(server, task, constraints, initial_code, strategy, max_iterations))
            finally:
                loop.close()
    except Exception as e:
        return {"error": str(e)}

