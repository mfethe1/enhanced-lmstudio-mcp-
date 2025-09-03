import os
import json
import importlib


def test_cognitive_orchestrate_smoke():
    server = importlib.import_module('server')
    s = server.get_server_singleton()

    # Ensure optimizer is enabled for caching/metrics
    os.environ['MCP_OPTIMIZER'] = '1'

    # Run orchestrator on a simple task
    args = {
        'task': 'Implement a simple async function and add tests',
        'constraints': {'style': 'readable'},
        'strategy': 'hierarchical',
        'max_iterations': 64,
    }
    payload = {"jsonrpc": "2.0", "id": 1, "params": {"name": "cognitive_orchestrate", "arguments": args}}
    out = server.handle_tool_call(payload)

    assert 'result' in out or 'error' in out, out
    if 'result' in out:
        res = out['result']['content'][0]['text'] if isinstance(out['result']['content'], list) else out['result']
        # Best-effort parse JSON
        try:
            data = json.loads(res) if isinstance(res, str) else res
        except Exception:
            data = res
        # Validate key fields in orchestrated response
        assert 'plan' in res or 'plan' in data
        assert 'agents' in res or 'agents' in data


def test_knowledge_graph_learning():
    kg_mod = importlib.import_module('cognitive_architecture.knowledge_graph')
    server = importlib.import_module('server')
    s = server.get_server_singleton()
    kg = kg_mod.CodeKnowledgeGraph(s.storage)
    aid = kg.add_code_artifact('def foo():\n    return 1', 'function', {'path': 'foo.py'})
    ctx = kg.get_context_for_generation('foo function')
    assert isinstance(ctx, list)


def test_mcts_planner_basic():
    server = importlib.import_module('server')
    s = server.get_server_singleton()
    planner_mod = importlib.import_module('cognitive_architecture.mcts_planner')
    planner = planner_mod.MCTSCodePlanner(s, max_iterations=32)
    try:
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(planner.plan_solution('Implement foo', {}))
    except RuntimeError:
        loop = __import__('asyncio').new_event_loop()
        try:
            __import__('asyncio').set_event_loop(loop)
            result = loop.run_until_complete(planner.plan_solution('Implement foo', {}))
        finally:
            loop.close()
    assert 'action_sequence' in result

