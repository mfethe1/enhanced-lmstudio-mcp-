import os
import sys
import json
import asyncio
# Ensure workspace root on sys.path when invoked as `python tests/run_smoke.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    # Prefer optimizer enabled
    os.environ.setdefault('MCP_OPTIMIZER', '1')

    import server
    s = server.get_server_singleton()

    # Ensure optimizer integration
    try:
        from production_optimizer import integrate_with_server
        integrate_with_server(s)
    except Exception:
        pass

    # Monkeypatch LLM to avoid external calls
    async def fake_llm(prompt: str, temperature: float = 0.2, **kwargs):
        pl = (prompt or '').lower()
        if 'rate this code quality' in pl:
            return '0.95'  # good
        if 'plan for:' in pl or 'implement:' in pl:
            return 'def auto():\n    return "ok"'
        # generic fallback
        return 'def foo():\n    return 1'

    # Patch route_chat and optimizer LB if present, but avoid overriding make_llm_request_with_retry directly
    try:
        s.route_chat = fake_llm  # type: ignore
    except Exception:
        pass
    try:
        if getattr(s, 'production', None) is not None:
            s.production.lb.llm = lambda prompt, **kw: fake_llm(prompt, **kw)  # type: ignore
    except Exception:
        pass

    # 1) Directly test MCTS planning
    from cognitive_architecture.mcts_planner import MCTSCodePlanner
    planner = MCTSCodePlanner(s, max_iterations=16)
    try:
        result1 = asyncio.get_event_loop().run_until_complete(planner.plan_solution('Implement foo function', {}))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result1 = loop.run_until_complete(planner.plan_solution('Implement foo function', {}))
        finally:
            loop.close()
    assert 'action_sequence' in result1

    # 2) Directly test agent spawner
    from cognitive_architecture.agent_spawner import get_spawner
    sp = get_spawner(s)
    caps = asyncio.get_event_loop().run_until_complete(sp.analyze_task_requirements('Refactor and test module', {"tags": [], "concerns": []}))
    ag1 = asyncio.get_event_loop().run_until_complete(sp.spawn_or_retrieve_agent(caps, 't1'))
    out2 = asyncio.get_event_loop().run_until_complete(sp.execute_with_agents('Refactor and test module', [ag1], 'sequential'))
    assert isinstance(out2, dict)

    # 3) Directly test orchestrator
    from cognitive_architecture.orchestrator import orchestrate_task
    out3 = orchestrate_task(s, 'Implement async util and add tests', {"style": "readable"}, '', 'hierarchical', 32)
    assert isinstance(out3, dict) and 'plan' in out3

    # 4) Optimizer metrics snapshot
    prod = getattr(s, 'production', None)
    if prod is not None:
        snap = prod.metrics_snapshot()
        # after smoke run, ratio should be computable
        _ = snap.get('cache_hit_ratio', 0.0)

    print('OK: smoke tests passed')
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except AssertionError as e:
        print('ASSERTION FAILED:', e)
        code = 2
    except Exception as e:
        print('ERROR:', e)
        code = 1
    sys.exit(code)

