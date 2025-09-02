import importlib, json


def test_router_battery_metrics_and_gating():
    srv_mod = importlib.import_module('server')

    class DummyServer:
        def __init__(self):
            # Simulate recent router decisions
            self._router_log = [
                {"backend": "lmstudio", "model": "openai/gpt-oss-20b", "latency_ms": 120, "confidence": 0.6},
                {"backend": "anthropic", "model": "claude-4-sonnet", "latency_ms": 450, "confidence": 0.8},
                {"backend": "openai", "model": "gpt5", "latency_ms": 300, "confidence": 0.4},
            ]
        async def route_chat(self, *a, **k):
            return "ok"

    server = DummyServer()
    # Configure thresholds that should PASS
    result = srv_mod.handle_router_battery({
        'cases': [],
        'limit': 0,
        'thresholds': {
            'min_anthropic': 1,
            'max_lmstudio_latency_ms': 500,
            'min_high_conf': 0.2
        }
    }, server)
    data = json.loads(result)
    assert data['passed'] is True
    assert 'metrics' in data and 'backend_counts' in data['metrics']

    # Configure thresholds that should FAIL (too strict)
    bad = srv_mod.handle_router_battery({
        'cases': [],
        'limit': 0,
        'thresholds': {
            'min_anthropic': 2,  # we only have 1
            'max_lmstudio_latency_ms': 50,  # too low
            'min_high_conf': 0.9  # too high
        }
    }, server)
    bdata = json.loads(bad)
    assert bdata['passed'] is False
    assert any(isinstance(r, str) for r in bdata.get('reasons', []))

