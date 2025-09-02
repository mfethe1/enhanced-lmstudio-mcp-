from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from server import _compact_text, _safe_path, _task_store, _new_task_id, _spawn_thread

# Note: This handler requires the Firecrawl MCP tool to be available in this environment.
# It performs multi-round deep research exclusively via the MCP tool, without HTTP fallbacks.
# Each round's findings are fed to an agent team to propose refined follow-up queries.
# Artifacts and summaries are stored in memory (category="research") so subsequent rounds improve.


async def _firecrawl_round(query: str, max_depth: int, time_limit: int) -> Dict[str, Any]:
    from functions import firecrawl_deep_research_firecrawl_mcp as firecrawl_deep  # type: ignore
    resp = firecrawl_deep({
        "query": query,
        "maxDepth": max(1, min(max_depth, 10)),
        "timeLimit": max(30, min(time_limit, 300)),
        "maxUrls": 50,
    })
    if not isinstance(resp, dict):
        return {"final": None, "raw": resp}
    final = (resp.get("data", {}) or {}).get("finalAnalysis")
    return {"final": final, "raw": resp}


def _mk_research_id(seed: str) -> str:
    return hashlib.md5(f"{seed}{time.time()}".encode()).hexdigest()[:12]


def _store_artifact(server, research_id: str, obj: Any, kind: str) -> None:
    try:
        server.storage.store_memory(
            key=f"research_{research_id}_{kind}",
            value=json.dumps(obj, ensure_ascii=False),
            category="research",
        )
    except Exception:
        pass


def _store_summary(server, research_id: str, summary_text: str) -> None:
    try:
        server.storage.store_memory(
            key=f"research_{research_id}",
            value=summary_text,
            category="research",
        )
    except Exception:
        pass


def _read_code_context(paths: List[str], base: str | None = None, max_chars: int = 20000) -> str:
    """Safely read a set of file paths for context. Does NOT modify any files.
    - Resolves paths under ALLOWED_BASE_DIR via server._safe_path
    - Truncates total content to max_chars for promptability
    """
    out: List[str] = []
    base_dir = Path(base) if base else None
    for p in paths:
        try:
            safe_p = _safe_path(p)
            if base_dir and base_dir not in safe_p.parents and safe_p != base_dir:
                continue
            if safe_p.is_dir():
                # Include directory listing and a few file previews
                out.append(f"\n# Dir: {safe_p}\n")
                previews = 0
                for child in safe_p.rglob("*.py"):
                    if previews >= 5:
                        break
                    try:
                        txt = child.read_text(encoding="utf-8", errors="ignore")
                        out.append(f"\n# File: {child}\n{txt[:1200]}\n")
                        previews += 1
                    except Exception:
                        continue
            else:
                txt = safe_p.read_text(encoding="utf-8", errors="ignore")
                out.append(f"\n# File: {safe_p}\n{txt}\n")
        except Exception:
            continue
        blob = _compact_text("\n".join(out), max_chars)
    return blob

def _agent_followups(server, query: str, prior_facts: str, round_index: int, max_q: int = 3) -> List[str]:
    """Use CrewAI planner agent with router-selected backend; fallback to routed LLM.
    Returns up to max_q follow-up queries as strings.
    """
    # Preferred: CrewAI Agent using per-role backend selection
    try:
        from crewai import Agent, Crew, Task  # type: ignore
        from server import _decide_backend_for_role, _build_llm_for_backend
        backend = _decide_backend_for_role("Planner", "deep_research")
        llm = _build_llm_for_backend(backend)
        base_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            base_kwargs["llm"] = llm
        planner = Agent(role="Planner", goal="Propose refined research queries targeting gaps.", backstory="Diligent planner.", **base_kwargs)
        task = Task(description=(
            f"Base query: {query}\nRound: {round_index}\n\n"
            f"Known findings (truncated):\n{prior_facts}\n\n"
            f"Propose up to {max_q} follow-up web research queries (plain JSON array of strings)."
        ), agent=planner)
        crew = Crew(agents=[planner], tasks=[task], verbose=False)
        out = str(crew.kickoff())
        try:
            arr = json.loads(out.strip())
            return [str(x) for x in arr][:max_q] if isinstance(arr, list) else []
        except Exception:
            lines = [l.strip("- ") for l in out.splitlines() if l.strip()]
            return lines[:max_q]
    except Exception:
        # Fallback to routed LLM
        prompt = (
            "You are a research planner. Given the base query and current findings, "
            "propose up to N refined web research queries as a JSON array of strings.\n\n"
            f"Base query: {query}\nRound: {round_index}\n\nKnown findings (truncated):\n{prior_facts}\n\nN={max_q}"
        )
        try:
            txt = asyncio.get_event_loop().run_until_complete(server.route_chat(prompt, role='Planner', intent='deep_research', temperature=0.2))
        except RuntimeError:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            txt = loop.run_until_complete(server.route_chat(prompt, role='Planner', intent='deep_research', temperature=0.2)); loop.close()
        try:
            arr = json.loads((txt or "").strip())
            return [str(x) for x in arr][:max_q] if isinstance(arr, list) else []
        except Exception:
            lines = [l.strip("- ") for l in (txt or "").splitlines() if l.strip()]
            return lines[:max_q]


def _rank_followups(followups: List[str], accumulated_keywords: set, project_keywords: set) -> List[str]:
    """Rank follow-up queries by novelty and project relevance.
    Returns followups sorted by score (highest first).
    """
    if not followups:
        return followups

    scored = []
    for fup in followups:
        fup_lower = fup.lower()
        fup_words = set(fup_lower.split())

        # Novelty score: how many new words vs accumulated
        novelty = len(fup_words - accumulated_keywords) / max(len(fup_words), 1)

        # Project relevance: mentions key project terms
        relevance = len(fup_words & project_keywords) / max(len(fup_words), 1)

        # Combined score with slight preference for novelty
        score = 0.6 * novelty + 0.4 * relevance
        scored.append((score, fup))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [fup for score, fup in scored]


def _run_deep_research_sync(arguments: Dict[str, Any], server, research_id: str) -> None:
    """Internal: run the full deep_research flow and persist results periodically under task_{id}."""
    try:
        # Mark started
        _task_store(server, research_id, {"id": research_id, "type": "deep_research", "status": "RUNNING", "progress": 0, "eta_seconds": 240, "data": {}})
        # Execute existing synchronous implementation (stores artifacts under research_{id})
        summary = handle_deep_research(arguments | {"research_id": research_id, "__force_sync": True}, server)
        # Mark completed with summary link
        _task_store(server, research_id, {"id": research_id, "type": "deep_research", "status": "DONE", "progress": 100, "eta_seconds": 0, "data": {"summary": summary}})

    except Exception as e:
        _task_store(server, research_id, {"id": research_id, "type": "deep_research", "status": "ERROR", "error": str(e)})

def handle_deep_research(arguments: Dict[str, Any], server) -> str:
    """Multi-round Firecrawl deep research with agent-team refinement and memory persistence.
    - Uses Firecrawl MCP exclusively (no HTTP fallback).
    - Runs multiple rounds: initial query -> findings -> agent proposes follow-ups -> next rounds.
    - Stores artifacts and summaries under category="research".
    - Optionally augments prompts with relevant code context if file_paths provided.
    - Warm-start: optionally pass research_id to continue from prior rounds.
    - verbose_summary: include larger inline text in the response (still stores full artifacts).
    """
    query = (arguments.get("query") or "").strip()
    rounds = max(1, int(arguments.get("rounds", 2)))
    max_depth = int(arguments.get("max_depth", 3))
    time_limit = int(arguments.get("time_limit", 120))
    max_followups = max(1, int(arguments.get("max_followups", 3)))
    file_paths = arguments.get("file_paths") or []
    base_dir = arguments.get("base_dir")
    warm_id = (arguments.get("research_id") or "").strip()
    verbose = bool(arguments.get("verbose_summary", False))

    research_id = warm_id or _mk_research_id(query or str(time.time()))

    # Optional: load code context for more grounded follow-ups/synthesis
    code_ctx = _read_code_context(file_paths, base=base_dir) if file_paths else ""

    # Immediate async mode: if not forced sync and task is heavy, spawn and return ID
    if not bool(arguments.get("__force_sync", False)) and (rounds > 3 or time_limit > 180):
        rid = research_id
        _spawn_thread(_run_deep_research_sync, arguments, server, rid)
        return json.dumps({"research_id": rid, "status": "STARTED", "message": "Deep research running in background. Call get_task_status with task_id=research_id."})

    all_rounds: List[Dict[str, Any]] = []
    accumulated_texts: List[str] = []
    accumulated_keywords: set = set()
    project_keywords = {"schema", "router", "crewai", "anthropic", "openai", "mcp", "firecrawl", "lmstudio", "agent", "backend", "tool"}

    # Warm start: load previous rounds
    if warm_id:
        try:
            rows = server.storage.retrieve_memory(category="research", limit=500) or []
            prev = []
            import re, json as _json
            for r in rows:
                k = r.get("key") or ""
                if k.startswith(f"research_{warm_id}_round"):
                    try:
                        obj = _json.loads(r.get("value", "{}"))
                        prev.append(obj)
                    except Exception:
                        continue
            prev = sorted(prev, key=lambda x: int(x.get("round", 0)))
            all_rounds.extend(prev)
            for p in prev:
                ftxt = p.get("final") or ""
                if ftxt:
                    accumulated_texts.append(ftxt)
                    # Update accumulated keywords
                    accumulated_keywords.update(ftxt.lower().split())
            # If no query given, continue from the last followup if present
            if not query:
                try:
                    foll = [r for r in rows if (r.get("key") or "").startswith(f"research_{warm_id}_followups")]
                    foll = sorted(foll, key=lambda x: int((x.get("key") or "").split("followups")[-1] or 0))
                    if foll:
                        last = _json.loads(foll[-1].get("value", "{}"))
                        fups = last.get("followups") or []
                        if fups:
                            query = fups[0]
                except Exception:
                    pass
        except Exception:
            pass

    if not query:
        raise Exception("'query' is required")

    for r in range(len(all_rounds)+1, len(all_rounds)+rounds+1):
        # 1) Run Firecrawl round
        stage = asyncio.get_event_loop().run_until_complete(_firecrawl_round(query, max_depth, time_limit))
        final = stage.get("final") or ""
        round_art = {"round": r, "query": query, "final": final, "raw": stage.get("raw")}
        _store_artifact(server, research_id, round_art, f"round{r}")

        all_rounds.append(round_art)
        if final:
            accumulated_texts.append(final)
            # Update accumulated keywords
            accumulated_keywords.update(final.lower().split())

        # 2) Ask agent for refined follow-up queries based on current findings + optional code context
        prior_facts = _compact_text("\n".join(accumulated_texts), 1500 if verbose else 1300)
        if code_ctx:
            prior_facts = _compact_text(prior_facts + "\n\n" + _compact_text(code_ctx, 800 if verbose else 600), 1800 if verbose else 1500)
        followups = _agent_followups(server, query, prior_facts, r, max_q=max_followups)

        # Rank follow-ups by novelty and project relevance
        if followups:
            followups = _rank_followups(followups, accumulated_keywords, project_keywords)

        _store_artifact(server, research_id, {"round": r, "followups": followups}, f"followups{r}")

        # 3) If there are follow-ups, pick the most promising for the next round
        if followups:
            query = followups[0]
        else:
            break

    # Final synthesis (include optional code context)
    findings_block = _compact_text("\n\n".join(accumulated_texts), 4000 if verbose else 2600)
    code_block = _compact_text(code_ctx, 2000 if verbose else 1200)
    synthesis_prompt = (
        "You are an expert synthesizer. Produce a concise, actionable summary (5-12 bullets) "
        "followed by a brief conclusion, based on multi-round findings. If code context is provided, "
        "use it to ground recommendations. Do NOT propose edits; provide guidance only.\n\n"
        f"Base research id: {research_id}\n\nFindings:\n{findings_block}\n\nCode context (optional):\n{code_block}"
    )
    try:
        final_text = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        final_text = loop.run_until_complete(server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2)); loop.close()

    final_summary = _compact_text(final_text, 4000 if verbose else 1200)
    _store_artifact(server, research_id, {"rounds": len(all_rounds), "summary": final_summary}, "summary")
    _store_summary(server, research_id, final_summary)

    # Build a per-round overview
    def _round_preview(r):
        q = r.get("query") or ""
        f = r.get("final") or ""
        return f"- Round {r.get('round')}: query=\"{_compact_text(q, 240 if verbose else 160)}\"\n  findings: {_compact_text(f, 400 if verbose else 280)}"
    overview = "\n".join(_round_preview(r) for r in all_rounds)

    # Compose return with richer context
    return (
        f"Deep research {research_id} completed with {len(all_rounds)} rounds.\n\n"
        f"Rounds overview:\n{overview}\n\n"
        f"Key findings (truncated):\n{_compact_text('\n\n'.join(accumulated_texts), 2400 if verbose else 1200)}\n\n"
        f"Summary:\n{final_summary}\n\n"
        "Tip: Call get_research_details with the research_id to see full artifacts (raw Firecrawl outputs, follow-ups, summary)."
    )

    # Stage 1: Firecrawl Deep Research
    try:
        try:
            from functions import firecrawl_deep_research_firecrawl_mcp as firecrawl_deep  # type: ignore
        except Exception:
            firecrawl_deep = None

        if firecrawl_deep is not None:
            fc = firecrawl_deep({
                "query": query,
                "maxDepth": max(1, min(max_depth, 10)),
                "timeLimit": max(30, min(time_limit, 300)),
                "maxUrls": 40,
            })
            stage1 = {
                "final": (fc.get("data", {}) or {}).get("finalAnalysis") if isinstance(fc, dict) else None,
                "raw": fc,
            }
        else:
            # Fallback to HTTP API if configured
            api_key = (os.getenv("FIRECRAWL_API_KEY", "").strip())
            base_url = (os.getenv("FIRECRAWL_BASE_URL", FIRECRAWL_BASE_URL_STATIC).rstrip("/"))
            if not api_key:
                stage1 = {
                    "final": None,
                    "raw": None,
                    "warning": "Firecrawl MCP unavailable and FIRECRAWL_API_KEY not set; skipping stage 1. Set FIRECRAWL_API_KEY or enable MCP."
                }
            else:
                import requests  # local import
                payload = {
                    "query": query,
                    "maxDepth": max(1, min(max_depth, 10)),
                    "timeLimit": max(30, min(time_limit, 300)),
                    "maxUrls": 40,
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                urls_to_try = [f"{base_url}/v1/deep-research", f"{base_url}/deep-research"]
                resp = None
                for url in urls_to_try:
                    try:
                        r = requests.post(url, json=payload, headers=headers, timeout=(30, 240))
                        if r.status_code == 404:
                            resp = r
                            continue
                        resp = r
                        break
                    except Exception:
                        continue
                if resp is not None and 200 <= resp.status_code < 300:
                    fc = resp.json()
                    stage1 = {
                        "final": (fc.get("data", {}) or {}).get("finalAnalysis") if isinstance(fc, dict) else None,
                        "raw": fc,
                    }
                else:
                    msg = f"no response" if resp is None else f"HTTP {resp.status_code}: {resp.text[:200]}"
                    stage1 = {"final": None, "raw": None, "error": f"Firecrawl {msg}"}
    except Exception as e:
        stage1 = {"final": None, "error": f"Firecrawl error: {str(e)}"}

    artifacts["stage1"] = stage1

    # Stage 2: CrewAI multi-agent synthesis (optional; best-effort)
    try:
        from crewai import Agent, Crew, Task  # type: ignore
        # Configure LLM for CrewAI to use LM Studio endpoint/model if available
        llm = None
        try:
            try:
                from crewai import LLM  # type: ignore
            except Exception:
                from crewai.llm import LLM  # type: ignore
            lmstudio_base = os.getenv("LMSTUDIO_API_BASE") or os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
            lmstudio_key = os.getenv("LMSTUDIO_API_KEY") or os.getenv("OPENAI_API_KEY", "sk-noauth")
            hardwired_model = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
            llm = LLM(model=hardwired_model, base_url=lmstudio_base, api_key=lmstudio_key, temperature=0.2)
        except Exception:
            llm = None

        agent_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            agent_kwargs["llm"] = llm

        analyst = Agent(role="Analyst", goal="Categorize and evaluate Firecrawl findings, extract key facts and gaps.", backstory="Senior research analyst skilled at distilling insights from diverse sources.", **agent_kwargs)
        researcher = Agent(role="Researcher", goal="Identify missing angles and perform targeted follow-ups based on gaps.", backstory="Curious investigator who knows where to look for authoritative sources.", **agent_kwargs)
        synthesizer = Agent(role="Synthesizer", goal="Produce concise, actionable recommendations tailored to the query.", backstory="Executive-level writer focusing on clarity and actionability.", **agent_kwargs)

        fc_summary = stage1.get("final") or ""
        fc_raw = stage1.get("raw")
        fc_raw_text = _compact_text(json.dumps(fc_raw, ensure_ascii=False) if isinstance(fc_raw, (dict, list)) else str(fc_raw))

        t1 = Task(description=f"Analyze initial findings for: {query}\n\nFindings:\n{fc_summary or fc_raw_text}", agent=analyst)
        t2 = Task(description=f"Identify missing angles and propose targeted follow-ups for: {query}", agent=researcher)
        t3 = Task(description=f"Synthesize concise, actionable recommendations for: {query}", agent=synthesizer)

        crew = Crew(agents=[analyst, researcher, synthesizer], tasks=[t1, t2, t3], verbose=False)
        crew_out = crew.kickoff()
        stage2 = {"report": str(crew_out)[:4000]}
    except Exception as e:
        # Fallback: synthesize using centralized LLM if CrewAI is unavailable
        try:
            fc_summary = stage1.get("final") or ""
            fc_raw = stage1.get("raw")
            fc_raw_text = _compact_text(json.dumps(fc_raw, ensure_ascii=False) if isinstance(fc_raw, (dict, list)) else str(fc_raw))
            synthesis_prompt = (
                "You are an expert research synthesizer. Based on the following findings, produce a concise, actionable summary with 3-6 bullets, then a short conclusion.\n\n"
                f"Query: {query}\n\nFindings:\n{fc_summary or fc_raw_text}\n\n"
                "Output only text."
            )
            try:
                llm_out = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    llm_out = loop.run_until_complete(server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2))
                finally:
                    loop.close()
            stage2 = {"report": _compact_text(llm_out)}
        except Exception as e2:
            stage2 = {"report": None, "error": f"CrewAI error: {str(e)}; fallback synthesis failed: {str(e2)}"}

    artifacts["stage2"] = stage2

    # Store artifacts and return concise summary
    try:
        server.storage.store_memory(key=f"research_{research_id}", value=json.dumps(artifacts, ensure_ascii=False), category="research")
    except Exception:
        pass

    # Compose concise return message
    stage1_note = stage1.get("final") or stage1.get("warning") or stage1.get("error") or "stage1-ok"
    stage2_note = stage2.get("report") or stage2.get("error") or "stage2-ok"
    summary = (
        f"Deep research {research_id} completed.\n\n"
        f"Stage 1 (Firecrawl): {stage1_note[:300]}\n\n"
        f"Stage 2 (Synthesis): {stage2_note[:600]}\n\n"
        "Use research_id to retrieve details later."
    )

    # Cache the summary to speed up repeated queries
    try:
        _DR_CACHE[key] = summary
    except Exception:
        pass

    return summary


def handle_web_search(arguments: Dict[str, Any], server) -> str:
    query = (arguments.get("query") or "").strip()
    if not query:
        raise Exception("'query' is required")
    time_limit = int(arguments.get("time_limit", 60))
    max_depth = int(arguments.get("max_depth", 1))
    return handle_deep_research({"query": query, "time_limit": time_limit, "max_depth": max_depth}, server)


def handle_get_research_details(arguments: Dict[str, Any], server) -> str:
    research_id = (arguments.get("research_id") or "").strip()
    if not research_id:
        raise Exception("'research_id' is required")
    key = f"research_{research_id}"
    try:
        rows = server.storage.retrieve_memory(key=key)
        if not rows:
            return f"No research found for id: {research_id}"
        row = rows[0]
        value = row.get("value") or ""
        try:
            obj = json.loads(value)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            return _compact_text(pretty, max_chars=4000)
        except Exception:
            return _compact_text(value, max_chars=4000)
    except Exception as e:
        return f"Error retrieving research {research_id}: {e}"


def handle_propose_research(arguments: Dict[str, Any], server) -> str:
    problem = (arguments.get("problem") or "").strip()
    if not problem:
        raise Exception("'problem' is required")
    context = (arguments.get("context") or "").strip()
    max_q = max(1, int(arguments.get("max_queries", 3)))
    prompt = (
        "You are a research planner. Propose up to N targeted research queries with 1-2 sentence reasons each. "
        "Output them as bullet points in plain text.\n\n"
        f"Problem: {problem}\nContext: {context}\nMax queries: {max_q}"
    )
    try:
        try:
            resp = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                resp = loop.run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
            finally:
                loop.close()
        return _compact_text(resp)
    except Exception as e:
        return f"Error: {str(e)}"

