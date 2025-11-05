from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Tuple

from agents.crew_manager import AgentSpec, _import_crewai_any


class AgentPool:
    """Lightweight pool with reuse by role key.

    This module is provided for external imports; server and tests can import from agents.pools
    without pulling the full crew_manager pipeline logic. The implementation mirrors the
    pool used by CodingCrewSystem.
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
        Agent, Crew, Task = self._import()
        task = Task(description=description, agent=agent, context=context, expected_output=expected_output)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        if hasattr(crew, "execute_task"):
            res = await crew.execute_task(task)  # type: ignore
            return res if isinstance(res, dict) else {"result": res}
        return {"result": description}

