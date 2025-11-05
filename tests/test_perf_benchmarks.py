import importlib
import time
import threading

from core.executor import async_executor


def _timeit(fn, repeat=10):
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat


def test_pooling_vs_nonpooled_latency():
    crew_mod = importlib.import_module('agents.crew_manager')

    class SlowAgent:
        def __init__(self, **kwargs):
            # simulate heavy init
            time.sleep(0.01)
            self.kwargs = kwargs

    class FakeTask:
        def __init__(self, description, agent, context=None, expected_output=None):
            self.description = description
            self.agent = agent

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            self.agents = agents
            self.tasks = tasks
        async def execute_task(self, task):
            return {"result": task.description}

    def fake_import():
        return SlowAgent, FakeCrew, FakeTask

    pool = crew_mod.AgentPool(import_crewai=fake_import)
    spec = crew_mod.AgentSpec(role="Any", goal="", backstory="")

    def pooled_once():
        async def _go():
            await pool.get_agent('role', spec)
        async_executor.run(_go())

    def non_pooled_once():
        async def _go():
            p = crew_mod.AgentPool(import_crewai=fake_import)
            await p.get_agent('role', spec)
        async_executor.run(_go())

    tp = _timeit(pooled_once, repeat=10)
    tnp = _timeit(non_pooled_once, repeat=10)
    # pooled should be faster than constructing a new agent repeatedly
    assert tp < tnp


def test_async_executor_vs_legacy_eventloop_creation():
    # compare async_executor.run vs creating a new loop per call
    async def noop():
        return 1

    def use_executor():
        async_executor.run(noop())

    def legacy_loop():
        import asyncio
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        loop.run_until_complete(noop())
        loop.close()

    te = _timeit(use_executor, repeat=50)
    tl = _timeit(legacy_loop, repeat=50)
    # executor should be at least not slower than legacy; often faster
    assert te <= tl * 1.5


def test_pipeline_completion_time_bounds():
    crew_mod = importlib.import_module('agents.crew_manager')
    sys = crew_mod.CodingCrewSystem()
    start = time.perf_counter()
    out = async_executor.run(sys.execute_pipeline('Quick task', {'prio':'low'}))
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0  # generous bound under fakes
    assert 'artifacts' in out


def test_concurrent_agent_usage_thread_safety():
    crew_mod = importlib.import_module('agents.crew_manager')
    sys = crew_mod.CodingCrewSystem()

    errs = []
    def worker(i):
        try:
            sys.execute_pipeline_sync(f"Task {i}", {"i": i}, timeout=10)
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert not errs

