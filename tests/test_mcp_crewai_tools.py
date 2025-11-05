import importlib

def test_agent_team_plan_and_code_tool_basic():
    srv = importlib.import_module('server')
    out = srv.handle_agent_team_plan_and_code({
        'instruction': 'Plan and code a small utility function',
        'context': {'scope': 'limited'},
        'priority': 'high',
        'timeout': 10,
    }, None)
    assert isinstance(out, dict)
    assert out.get('tool') == 'agent_team_plan_and_code'
    assert out.get('invoked') is True
    assert 'artifacts' in out and 'logs' in out


def test_agent_team_review_and_test_tool_with_diff():
    srv = importlib.import_module('server')
    out = srv.handle_agent_team_review_and_test({
        'diff': 'diff --git a/a.py b/a.py\n+print("hi")',
        'context': 'safety review',
        'priority': 'normal',
        'timeout': 10,
    }, None)
    assert isinstance(out, dict)
    assert out.get('tool') == 'agent_team_review_and_test'
    assert out.get('invoked') is True
    assert 'artifacts' in out and 'logs' in out


def test_agent_team_refactor_tool_with_instruction():
    srv = importlib.import_module('server')
    out = srv.handle_agent_team_refactor({
        'instruction': 'Refactor the data access layer to improve clarity',
        'priority': 'low',
        'timeout': 10,
    }, None)
    assert isinstance(out, dict)
    assert out.get('tool') == 'agent_team_refactor'
    assert out.get('invoked') is True
    assert 'artifacts' in out and 'logs' in out

