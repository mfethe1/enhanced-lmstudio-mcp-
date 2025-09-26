#!/usr/bin/env python3
"""
Example: Dynamically assemble a multi-agent team and run a workflow using the
existing LM Studio MCP server.
"""
import os
os.environ.setdefault("PROACTIVE_RESEARCH_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import server
from strands import TeamOrchestrator, OrchestratorConfig

# Speed up by monkeypatching LM Studio request for demo (optional)
async def _fake_lmstudio_request(prompt: str, temperature: float = 0.2):
    return '{"steps": []}'

srv = server.EnhancedLMStudioMCPServer()
srv._lmstudio_request_with_retry = _fake_lmstudio_request

orchestrator = TeamOrchestrator(srv, OrchestratorConfig(project_id="demo", dry_run=True))
summary = orchestrator.orchestrate(
    task="Design and prototype a small web tool to visualize enzyme kinetics and propose a lab validation plan.",
    context="User wants a simple Next.js app and an experiment outline.")

print("Stages:", [s["stage"] for s in summary["results"]])
print("Agents:", [a["role"] for a in summary["agents"]])

