import os
import json
from pathlib import Path

import server as s


def _call_tool(name: str, arguments: dict):
    payload = {"jsonrpc": "2.0", "id": 999, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return s.handle_tool_call(payload)


def test_agent_team_plan_and_code_apply_changes_fallback(tmp_path, monkeypatch):
    # Force fallback to avoid CrewAI and network usage
    monkeypatch.setenv("AGENT_TEAM_FORCE_FALLBACK", "1")

    # Fabricate a fallback LLM response with a fenced code block and a File header
    proposed = (
        "Plan...\n\n"  # plan text
        "```python\n# File: demo_applied.txt\nhello from agent team\n```\n"
    )

    async def fake_llm(self, prompt: str, temperature: float = 0.2, retries: int = 2, backoff: float = 0.5):
        return proposed

    monkeypatch.setattr(s.EnhancedLMStudioMCPServer, "make_llm_request_with_retry", fake_llm, raising=True)

    # Run tool with apply_changes=True
    res = _call_tool(
        "agent_team_plan_and_code",
        {
            "task": "Create a demo file",
            "target_files": [],
            "constraints": "Small demo",
            "apply_changes": True,
        },
    )
    text = res["result"]["content"][0]["text"]
    assert "Applied changes" in text

    # Verify file was written
    out_path = Path("demo_applied.txt")
    assert out_path.exists()
    try:
        assert out_path.read_text(encoding="utf-8").strip().startswith("hello from agent team")
    finally:
        # Cleanup
        out_path.unlink(missing_ok=True)


def test_agent_team_review_and_test_fallback(monkeypatch):
    monkeypatch.setenv("AGENT_TEAM_FORCE_FALLBACK", "1")

    async def fake_llm(self, prompt: str, temperature: float = 0.2, retries: int = 2, backoff: float = 0.5):
        return "Review: OK\n\n```python\n# File: tests/test_demo.py\nimport pytest\n\n\n def test_demo():\n     assert 1+1==2\n```\n"

    monkeypatch.setattr(s.EnhancedLMStudioMCPServer, "make_llm_request_with_retry", fake_llm, raising=True)

    res = _call_tool(
        "agent_team_review_and_test",
        {"diff": "--- a\n+++ b\n+print('x')", "context": "small change"},
    )
    text = res["result"]["content"][0]["text"]
    assert "Review" in text or "pytest" in text


def test_agent_team_refactor_fallback(monkeypatch, tmp_path):
    # Create a temp file to reference
    module_path = tmp_path / "mod.py"
    module_path.write_text("x=1\n")
    monkeypatch.setenv("AGENT_TEAM_FORCE_FALLBACK", "1")

    async def fake_llm(self, prompt: str, temperature: float = 0.2, retries: int = 2, backoff: float = 0.5):
        return (
            "Rationale: simplify code.\n\n"
            "```python\n# File: tests/scratch_refactor.py\n# refactored content\n```\n"
        )

    monkeypatch.setattr(s.EnhancedLMStudioMCPServer, "make_llm_request_with_retry", fake_llm, raising=True)

    res = _call_tool(
        "agent_team_refactor",
        {"module_path": str(module_path), "goals": "readability"},
    )
    text = res["result"]["content"][0]["text"]
    assert "Rationale" in text or "refactored" in text

