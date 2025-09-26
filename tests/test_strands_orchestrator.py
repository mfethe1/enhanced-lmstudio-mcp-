#!/usr/bin/env python3
import os
os.environ.setdefault("PROACTIVE_RESEARCH_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")

def test_strands_orchestrator_smoke():
    import server
    from strands import TeamOrchestrator, OrchestratorConfig

    # Fake LM Studio to avoid network
    async def _fake_lmstudio_request(prompt: str, temperature: float = 0.2):
        return '{"steps": []}'

    srv = server.EnhancedLMStudioMCPServer()
    srv._lmstudio_request_with_retry = _fake_lmstudio_request

    orch = TeamOrchestrator(srv, OrchestratorConfig(project_id="test", dry_run=True))
    out = orch.orchestrate("Implement a small backend API and document it.", context="Use FastAPI.")

    assert "agents" in out and out["agents"], "Should assemble a team"
    assert "results" in out and out["results"], "Should execute stages"

