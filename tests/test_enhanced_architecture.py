import os
import time
from pathlib import Path

import server as s
from enhanced_agent_architecture import classify_task_complexity, decide_backend, ComplexityLevel


def _call(name: str, arguments: dict):
    payload = {"jsonrpc": "2.0", "id": 777, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return s.handle_tool_call(payload)


def test_classification_and_routing_smoke(monkeypatch):
    # Ensure at least LM Studio appears available to router
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234")

    comp = classify_task_complexity("Perform a security audit with concurrency and cryptography")
    assert comp in {ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT}

    choice = decide_backend("Security Reviewer", "security audit for webapp")
    assert choice.backend in {"anthropic", "openai", "lmstudio"}


def test_envelope_storage_after_tool_call(monkeypatch):
    # Call a cheap tool and then query storage for envelope category
    res = _call("health_check", {"probe_lm": False})
    assert res["result"]["content"][0]["text"]

    # Read from storage using the server singleton
    server = s.get_server_singleton()
    rows = server.storage.retrieve_memory(category="envelope", limit=5)
    assert isinstance(rows, list)
    # At least one envelope logged recently
    assert any(r.get("key", "").startswith("envelope_") for r in rows)


def test_safe_path_enforced_on_write(tmp_path, monkeypatch):
    # Attempt to write outside ALLOWED_BASE_DIR should error
    monkeypatch.setenv("ALLOWED_BASE_DIR", str(tmp_path))
    # Recreate singleton to pick up new base dir
    s._server_singleton = None
    server = s.get_server_singleton()

    outside_path = str(Path(tmp_path).parent / "outside.txt")
    resp = _call("write_file_content", {"file_path": outside_path, "content": "x", "mode": "overwrite"})
    assert "error" in resp
    assert resp["error"]["code"] == -32602


def test_health_check_ok():
    out = _call("health_check", {"probe_lm": False})
    text = out["result"]["content"][0]["text"]
    assert "ok" in text or "storage" in text.lower()

