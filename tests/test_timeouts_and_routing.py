import importlib
import json


def test_timeouts_doubled_in_server():
    # Verify that server timeouts and max tokens were doubled in code paths
    import inspect
    srv_src = inspect.getsource(importlib.import_module('server'))
    assert 'timeout=(60, 240)' in srv_src
    assert 'ROUTER_MAX_TOKENS' in srv_src and '"4000"' in srv_src


def test_router_battery_and_diagnostics_models(monkeypatch):
    srv_mod = importlib.import_module('server')

    class DummyServer:
        def __init__(self):
            self._router_log = []
        async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
            # Simulate decisions based on role/intent
            backend = 'anthropic' if (intent == 'analysis') else 'openai' if (role == 'Planner') else 'lmstudio'
            model = 'claude-4-sonnet' if backend == 'anthropic' else 'gpt5' if backend == 'openai' else 'openai/gpt-oss-20b'
            self._router_log.append({"backend": backend, "model": model, "intent": intent, "role": role})
            return "ok"

    server = DummyServer()
    out = srv_mod.handle_router_battery({
        "cases": [
            {"task": "Deep analysis", "intent": "analysis", "role": "Reviewer"},
            {"task": "Plan", "intent": "synthesis", "role": "Planner"},
            {"task": "Code", "intent": "implementation", "role": "Developer"}
        ]
    }, server)
    data = json.loads(out)
    assert isinstance(data.get('report'), list) and len(data['report']) == 3

    diags = srv_mod.handle_router_diagnostics({"limit": 10}, server)
    # Fallback diagnostics returns raw decisions; validate models appear
    decisions = diags.get('decisions', [])
    models = {d.get('model') for d in decisions if isinstance(d, dict)}
    assert {'claude-4-sonnet', 'gpt5', 'openai/gpt-oss-20b'} & models

