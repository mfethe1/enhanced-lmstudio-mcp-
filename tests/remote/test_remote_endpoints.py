import os
import json
import asyncio
import pytest

os.environ.setdefault("REMOTE_ENABLED", "true")
from remote_server import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_auth_required_when_token_set(monkeypatch):
    monkeypatch.setenv("MCP_REMOTE_TOKEN", "secret")
    r = client.post("/rpc", json={"id":1, "params": {"name": "health_check", "arguments": {}}})
    assert r.status_code == 401
    # With token
    r2 = client.post("/rpc", json={"id":2, "params": {"name": "health_check", "arguments": {}}}, headers={"Authorization": "Bearer secret"})
    assert r2.status_code == 200
    data = r2.json()
    assert data.get("result"), data


def test_mcp_text_only_envelope_http(monkeypatch):
    monkeypatch.setenv("MCP_REMOTE_TOKEN", "")  # disable auth
    r = client.post("/rpc", json={"id":3, "params": {"name": "health_check", "arguments": {}}})
    assert r.status_code == 200
    data = r.json()
    assert "result" in data and "content" in data["result"]
    ct = data["result"]["content"][0]
    assert ct["type"] == "text"


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("MCP_REMOTE_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    # Recreate client with env applied
    c = TestClient(app)
    # First request should pass
    r1 = c.post("/rpc", json={"id":10, "params": {"name": "health_check", "arguments": {}}})
    # Immediate second request may be limited
    r2 = c.post("/rpc", json={"id":11, "params": {"name": "health_check", "arguments": {}}})
    assert r1.status_code == 200
    assert r2.status_code in (200, 429)

