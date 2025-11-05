from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

from core.executor import async_executor

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is in requirements, but keep soft
    yaml = None  # type: ignore

# Soft import CrewAI with a helper so tests can monkeypatch
from typing import Any as _Any, cast as _cast


def _import_crewai_any() -> Tuple[_Any, _Any, _Any]:
    try:
        from crewai import Agent as _Agent, Crew as _Crew, Task as _Task  # type: ignore
    except Exception:
        try:
            from crewai import Agent as _Agent  # type: ignore
            from crewai import Crew as _Crew  # type: ignore
            from crewai import Task as _Task  # type: ignore
        except Exception as e:
            # Return simple fakes that let tests run without CrewAI
            class _Agent:  # type: ignore
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class _Task:  # type: ignore
                def __init__(self, description: str, agent: Any, context: Any = None, expected_output: Optional[str] = None):
                    self.description = description
                    self.agent = agent
                    self.context = context
                    self.expected_output = expected_output

            class _Crew:  # type: ignore
                def __init__(self, agents, tasks, process: Any | None = None, manager_agent: Any | None = None, memory: bool = False, cache: bool = False, embedder: Any | None = None, verbose: bool = False):
                    self.agents = agents
                    self.tasks = tasks

                async def kickoff(self):
                    # Basic fake that returns predictable artifacts
                    return {"results": [getattr(t, "description", "") for t in self.tasks]}

                async def execute_task(self, task: Any):
                    return {"result": getattr(task, "description", "")}

            return _cast(_Any, _Agent), _cast(_Any, _Crew), _cast(_Any, _Task)
    return _cast(_Any, _Agent), _cast(_Any, _Crew), _cast(_Any, _Task)


@dataclass
class AgentSpec:
    role: str
    goal: str
    backstory: str
    tools: Optional[list[str]] = None
    max_iter: Optional[int] = None
    max_execution_time: Optional[int] = None


class ConfigLoader:
    def __init__(self, agents_path: str = "config/agents.yaml", tasks_path: str = "config/tasks.yaml"):
        self.agents_path = Path(agents_path)
        self.tasks_path = Path(tasks_path)
        self._agents: Dict[str, AgentSpec] | None = None
        self._tasks: Dict[str, Dict[str, Any]] | None = None

    def load_agents(self) -> Dict[str, AgentSpec]:
        if self._agents is not None:
            return self._agents
        data: Dict[str, Any] = {}
        if yaml and self.agents_path.exists():
            data = yaml.safe_load(self.agents_path.read_text(encoding="utf-8")) or {}
        out: Dict[str, AgentSpec] = {}
        for key, val in data.items():
            out[key] = AgentSpec(
                role=val.get("role", key),
                goal=val.get("goal", ""),
                backstory=val.get("backstory", ""),
                tools=val.get("tools"),
                max_iter=val.get("max_iter"),
                max_execution_time=val.get("max_execution_time"),
            )
        self._agents = out
        return out

    def load_tasks(self) -> Dict[str, Dict[str, Any]]:
        if self._tasks is not None:
            return self._tasks
        data: Dict[str, Any] = {}
        if yaml and self.tasks_path.exists():
            data = yaml.safe_load(self.tasks_path.read_text(encoding="utf-8")) or {}
        self._tasks = data
        return data


class AgentPool:
    """Simple in-memory agent pool by role key.

    - Avoids per-request construction overhead
    - Thread-safe with AsyncExecutor (all CrewAI calls run on executor loop)
    - Exposes get_agent(role_key) and run_task(agent, ...)
    """

    def __init__(self, import_crewai: Callable[[], Tuple[Any, Any, Any]] = _import_crewai_any):
        self._cache: Dict[str, Any] = {}
        self._constructed: Dict[str, int] = {}
        self._import = import_crewai

    @property
    def constructed_counts(self) -> Dict[str, int]:
        return dict(self._constructed)

    async def get_agent(self, role_key: str, spec: AgentSpec, llm: Any | None = None) -> Any:
        if role_key in self._cache:
            return self._cache[role_key]
        Agent, _Crew, _Task = self._import()
        kwargs = {"role": spec.role, "goal": spec.goal, "backstory": spec.backstory}
        if llm is not None:
            kwargs["llm"] = llm
        if spec.max_iter is not None:
            kwargs["max_iter"] = spec.max_iter
        if spec.max_execution_time is not None:
            kwargs["max_execution_time"] = spec.max_execution_time
        agent = Agent(**kwargs)
        self._cache[role_key] = agent
        self._constructed[role_key] = self._constructed.get(role_key, 0) + 1
        return agent

    async def run_task(self, agent: Any, description: str, context: Dict[str, Any] | None = None, expected_output: Optional[str] = None) -> Dict[str, Any]:
        """Run a single-task CrewAI crew with broad version compatibility.

        - Some CrewAI versions require expected_output: str (not None)
        - Some require context to be a list or simple structure rather than dict
        We attempt progressively more compatible constructions before falling back.
        """
        Agent, Crew, Task = self._import()

        # Attempt 1: as provided
        try:
            task = Task(description=description, agent=agent, context=context, expected_output=expected_output)
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
        except Exception:
            # Attempt 2: coerce expected_output and context
            coerced_expected = expected_output if isinstance(expected_output, str) and expected_output else ""
            coerced_context = context or {}
            if isinstance(coerced_context, dict):
                # Convert dict to list of simple "key: value" strings for stricter schemas
                coerced_context = [f"{k}: {v}" for k, v in coerced_context.items()]
            try:
                task = Task(description=description, agent=agent, context=coerced_context, expected_output=coerced_expected)
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
            except Exception:
                # Attempt 3: minimal task with no optional fields
                task = Task(description=description, agent=agent)
                crew = Crew(agents=[agent], tasks=[task], verbose=False)

        # Many CrewAI versions are async; ensure we await with executor when called from sync
        async def _do():
            if hasattr(crew, "execute_task") and asyncio.iscoroutinefunction(crew.execute_task):
                return await crew.execute_task(task)  # type: ignore
            # Fallback fake
            return {"result": description}

        # We're already inside async most times in server, but keep it clean
        return await _do()


class CodingCrewSystem:
    """Persistent crew system with specialized agents and a multi-stage pipeline.

    This class is import-safe without CrewAI installed and integrates with AsyncExecutor for
    sync contexts via the provided *_sync wrappers.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None, pool: Optional[AgentPool] = None) -> None:
        self.loader = config_loader or ConfigLoader()
        self.pool = pool or AgentPool()
        self._agents_conf = self.loader.load_agents()
        self._tasks_conf = self.loader.load_tasks()

    # --- role helpers ---
    async def get_agent(self, key: str) -> Any:
        spec = self._agents_conf.get(key)
        if not spec:
            # Construct a default spec if missing to keep robust
            spec = AgentSpec(role=key, goal=f"Act as {key}", backstory="")
        return await self.pool.get_agent(key, spec, llm=self.get_optimal_llm(key))

    def get_optimal_llm(self, purpose: str) -> Any | None:
        # Placeholder: in future select per-purpose LLM; return None to use defaults
        return None

    # --- pipeline ---
    stages: list[tuple[str, str]] = [
        ("requirements_analyst", "Extract and validate requirements"),
        ("architect", "Design solution architecture"),
        ("coder", "Implement solution"),
        ("test_engineer", "Generate comprehensive tests"),
        ("performance_tuner", "Optimize for performance"),
        ("security_auditor", "Security review"),
    ]

    async def execute_pipeline(self, task: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        context = context or {}
        artifacts: Dict[str, Any] = {}
        logs: list[str] = []
        for stage_key, description in self.stages:
            agent = await self.get_agent(stage_key)
            stage_desc = f"{description}: {task}"
            expected = self._tasks_conf.get(stage_key, {}).get("expected_output") if isinstance(self._tasks_conf, dict) else None
            result = await self.pool.run_task(agent, stage_desc, {"artifacts": artifacts, **context}, expected_output=expected)
            artifacts[stage_key] = result
            logs.append(stage_key)
            # Early exit conditions
            if isinstance(result, dict) and (result.get("halt") or result.get("status") == "error"):
                return {"artifacts": artifacts, "halted": stage_key, "logs": logs}
        return {"artifacts": artifacts, "logs": logs}

    # --- sync wrappers for convenience ---
    def execute_pipeline_sync(self, task: str, context: Dict[str, Any] | None = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        return async_executor.run(self.execute_pipeline(task, context), timeout=timeout)

