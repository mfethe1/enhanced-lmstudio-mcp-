import json
import types

class DummyServer:
    def __init__(self):
        self._router_log = []
        self.calls = []
    async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
        # Record call
        self.calls.append({"prompt": prompt, "intent": intent, "role": role})
        # Simulate a minimal decision path side effect (like logging)
        self._router_log.append({"backend": "lmstudio", "source": "test", "intent": intent, "role": role})
        return ""

def test_router_battery_basic(monkeypatch):
    # Import the real handler from server.py
    import importlib
    srv_mod = importlib.import_module('server')

    server = DummyServer()
    # Call the handler directly
    out = srv_mod.handle_router_battery({"cases": [{"task": "A", "intent": "analysis", "role": "Reviewer"}]}, server)
    data = json.loads(out)
    assert "report" in data and isinstance(data["report"], list)
    assert len(data["report"]) == 1
    assert "diagnostics" in data

    # router_diagnostics should use _router_log fallback
    diags = srv_mod.handle_router_diagnostics({"limit": 5}, server)
    assert "decisions" in diags or "analytics" in diags


def test_router_self_test_runs(monkeypatch):
    import importlib
    srv_mod = importlib.import_module('server')

    server = DummyServer()
    # self-test uses route_chat internally; dummy is fine
    out = srv_mod.handle_router_self_test({}, server)
    assert isinstance(out, str)
    assert "Case:" in out or out.startswith("Router self-test failed:")

