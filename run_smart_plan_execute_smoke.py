#!/usr/bin/env python3
import os
import asyncio

# Ensure clean environment for test
os.environ.setdefault("PROACTIVE_RESEARCH_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import server

# Monkeypatch LM Studio request to avoid network
async def _fake_lmstudio_request(prompt: str, temperature: float = 0.2):
    # Return a minimal valid JSON for planning stage
    return '{"steps": []}'

srv = server.EnhancedLMStudioMCPServer()
# Inject fake async method
srv._lmstudio_request_with_retry = _fake_lmstudio_request

# Prepare minimal arguments
arguments = {
    "instruction": "Quick smoke plan",
    "context": "",
    "dry_run": True,
    "max_steps": 2
}

# Call the handler
out = server.handle_smart_plan_execute(arguments, srv)
print("smart_plan_execute output type:", type(out))
print("smart_plan_execute output:", str(out)[:200] + "..." if len(str(out)) > 200 else str(out))
if isinstance(out, dict):
    print("smart_plan_execute output keys:", list(out.keys()))
else:
    print("smart_plan_execute returned string (expected for some cases)")
# Verify router log now exists and is a list
print("_router_log exists:", hasattr(srv, "_router_log"))
print("_router_log length:", len(getattr(srv, "_router_log", [])))

