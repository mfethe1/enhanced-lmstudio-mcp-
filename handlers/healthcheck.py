from __future__ import annotations

import asyncio
import json
import os

from typing import Dict


def handle_research_healthcheck(arguments: Dict[str, object], server) -> str:
    """Exercise Firecrawl MCP and CrewAI with a tiny call to verify readiness.
    Returns a JSON report with fields: firecrawl_ok, crewai_ok, route_openai, route_anthropic, route_lmstudio.
    """
    report = {"firecrawl_ok": False, "crewai_ok": False, "route_openai": False, "route_anthropic": False, "route_lmstudio": False}

    # Firecrawl MCP tiny check via deep research on a trivial query (dry run is not used; small time_limit)
    try:
        from functions import firecrawl_deep_research_firecrawl_mcp as firecrawl_deep  # type: ignore
        out = firecrawl_deep({"query": "hello world", "maxDepth": 1, "timeLimit": 30, "maxUrls": 2})
        report["firecrawl_ok"] = bool(out)
    except Exception:
        report["firecrawl_ok"] = False

    # CrewAI presence
    try:
        import crewai  # type: ignore
        report["crewai_ok"] = True
    except Exception:
        report["crewai_ok"] = False

    # Router paths: attempt a quick routed call to each backend if env keys exist
    async def _try_route(backend: str) -> bool:
        try:
            if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
                return False
            if backend == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
                return False
            txt = await server.route_chat("ping", preferred_backend=backend, temperature=0.0)
            return isinstance(txt, str) and len(txt) > 0
        except Exception:
            return False

    try:
        report["route_openai"] = asyncio.get_event_loop().run_until_complete(_try_route("openai"))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            report["route_openai"] = loop.run_until_complete(_try_route("openai"))
        finally:
            loop.close()

    try:
        report["route_anthropic"] = asyncio.get_event_loop().run_until_complete(_try_route("anthropic"))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            report["route_anthropic"] = loop.run_until_complete(_try_route("anthropic"))
        finally:
            loop.close()

    try:
        report["route_lmstudio"] = asyncio.get_event_loop().run_until_complete(_try_route("lmstudio"))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            report["route_lmstudio"] = loop.run_until_complete(_try_route("lmstudio"))
        finally:
            loop.close()

    return json.dumps(report)
