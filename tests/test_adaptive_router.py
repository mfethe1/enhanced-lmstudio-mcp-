import importlib

def test_adaptive_router_choose_tool_and_learn():
    mod = importlib.import_module('core.adaptive_router')
    r = mod.adaptive_router
    # ensure clean bias update
    r.learn_from_artifacts({'performance_tuner': {'result': 'ok'}})
    choice, meta = r.choose_tool('Optimize performance of function', 'context', ['agent_team_plan_and_code', 'agent_team_review_and_test', 'agent_team_refactor', 'deep_research'])
    assert choice in {'agent_team_plan_and_code', 'agent_team_review_and_test', 'agent_team_refactor', 'deep_research'}
    assert isinstance(meta, dict) and 'confidence' in meta


def test_handle_smart_task_uses_adaptive_router_path():
    srv = importlib.import_module('server')
    data = srv.handle_smart_task({'instruction': 'write unit tests for module', 'context': 'fast', 'dry_run': True}, None)
    # Returns JSON string summary
    import json
    obj = json.loads(data)
    assert 'selected' in obj and 'invoked' in obj  # invoked present from earlier behavior

