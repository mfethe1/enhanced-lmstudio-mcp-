import importlib, json


def test_smart_task_v2_schema_inference(monkeypatch):
    srv_mod = importlib.import_module('server')

    class DummyServer:
        async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
            # Propose tool and minimal arguments missing required fields
            return json.dumps({
                'tool': 'agent_team_review_and_test',
                'arguments': {'apply_fixes': True},
                'confidence': 0.61
            })

    out = srv_mod.handle_smart_task({
        'instruction': 'review and test this change', 'context': 'safety-critical', 'dry_run': True
    }, DummyServer())
    data = json.loads(out)
    assert data['selected'] == 'agent_team_review_and_test'
    assert data['invoked'] is False
    # Should auto-fill required fields like diff from instruction/context
    assert 'diff' in data['arguments'] and isinstance(data['arguments']['diff'], str)
    assert data['valid'] is True or isinstance(data['errors'], list)


def test_smart_plan_execute_dry_plan(monkeypatch):
    srv_mod = importlib.import_module('server')

    class DummyServer:
        async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
            # Plan two steps with partial args
            return json.dumps({
                'steps': [
                    {'tool': 'tool_match', 'arguments': {'task': 'find tool'}, 'note': 'find candidates'},
                    {'tool': 'deep_research', 'arguments': {'query': 'MCP best practices'}, 'note': 'research'}
                ]
            })

    out = srv_mod.handle_smart_plan_execute({
        'instruction': 'Find relevant tools and research MCP best practices',
        'context': '',
        'dry_run': True,
        'max_steps': 2
    }, DummyServer())
    plan = json.loads(out)
    assert 'plan' in plan and isinstance(plan['plan'], list) and len(plan['plan']) == 2

