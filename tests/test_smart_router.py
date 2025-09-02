import importlib
import json


def test_smart_task_rules(monkeypatch):
    srv_mod = importlib.import_module('server')

    # Dry run: rule-based
    resp = srv_mod.handle_smart_task({
        'instruction': 'Please review and test this change with pytest',
        'context': 'safety critical',
        'dry_run': True
    }, server=object())
    data = json.loads(resp)
    assert data['selected'] == 'agent_team_review_and_test'
    assert data['invoked'] is False


def test_smart_task_llm_fallback(monkeypatch):
    srv_mod = importlib.import_module('server')

    class DummyServer:
        async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
            return json.dumps({'tool': 'web_search'})

    # No rule words; fallback to LLM picks web_search - use dry_run to avoid external calls
    out_json = srv_mod.handle_smart_task({'instruction': 'find docs about MCP discovery', 'context': '', 'dry_run': True}, DummyServer())
    data = json.loads(out_json)
    assert data['selected'] == 'web_search'
    assert data['invoked'] is False

