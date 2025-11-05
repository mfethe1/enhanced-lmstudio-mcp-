import importlib
import json
import time

import pytest
from core.executor import async_executor


def _run(coro):
    return async_executor.run(coro)


def test_agent_pool_reuse(monkeypatch):
    crew_mod = importlib.import_module('agents.crew_manager')

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, description, agent, context=None, expected_output=None):
            self.description = description
            self.agent = agent
            self.context = context
            self.expected_output = expected_output

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            self.agents = agents
            self.tasks = tasks
        async def execute_task(self, task):
            return {"result": task.description}

    def fake_import():
        return FakeAgent, FakeCrew, FakeTask

    pool = crew_mod.AgentPool(import_crewai=fake_import)
    spec = crew_mod.AgentSpec(role="Code Architect", goal="Design", backstory="")

    a1 = _run(pool.get_agent('architect', spec))
    a2 = _run(pool.get_agent('architect', spec))

    assert a1 is a2
    assert pool.constructed_counts.get('architect') == 1


def test_specialized_pipeline_flow(monkeypatch):
    crew_mod = importlib.import_module('agents.crew_manager')

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, description, agent, context=None, expected_output=None):
            self.description = description
            self.agent = agent
            self.context = context
            self.expected_output = expected_output

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            self.agents = agents
            self.tasks = tasks
        async def execute_task(self, task):
            # Return stage key embedded in description
            return {"result": f"OK::{task.description.split(':', 1)[0]}"}

    def fake_import():
        return FakeAgent, FakeCrew, FakeTask

    # Build system with fake pool wired in
    pool = crew_mod.AgentPool(import_crewai=fake_import)
    sys = crew_mod.CodingCrewSystem(pool=pool)

    out = _run(sys.execute_pipeline("Build feature X", {"priority": "high"}))
    assert 'artifacts' in out and 'logs' in out
    # confirm all 6 stages executed in order
    assert out['logs'] == [s[0] for s in sys.stages]
    for stage_key, _ in sys.stages:
        assert stage_key in out['artifacts']
        assert 'result' in out['artifacts'][stage_key]


def test_pipeline_sync_wrapper(monkeypatch):
    crew_mod = importlib.import_module('agents.crew_manager')

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, description, agent, context=None, expected_output=None):
            self.description = description
            self.agent = agent
            self.context = context
            self.expected_output = expected_output

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            self.agents = agents
            self.tasks = tasks
        async def execute_task(self, task):
            return {"result": task.description}

    def fake_import():
        return FakeAgent, FakeCrew, FakeTask

    pool = crew_mod.AgentPool(import_crewai=fake_import)
    sys = crew_mod.CodingCrewSystem(pool=pool)

    out = sys.execute_pipeline_sync("Refactor module Y", {"refactor": True}, timeout=5)
    assert 'artifacts' in out
    assert out['logs'][0] == 'requirements_analyst'

