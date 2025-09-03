from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from server import _compact_text, _decide_backend_for_role, _build_llm_for_backend


def _agent_llm_for_role(role: str):
    backend = _decide_backend_for_role(role, "agent_team")
    try:
        from crewai import LLM  # type: ignore
    except Exception:
        return None
    return _build_llm_for_backend(backend)


def handle_agent_collaborate(arguments: Dict[str, Any], server) -> str:
    task = (arguments.get("task") or "").strip()
    if not task:
        raise Exception("'task' is required")
    roles = arguments.get("roles") or ["Researcher", "Developer", "Reviewer", "Security Reviewer"]
    rounds = int(arguments.get("rounds", 2))
    history: list[dict] = []
    for r in range(1, rounds + 1):
        for role in roles:
            context = "\n\n".join([f"{h['role']}: {h['note']}" for h in history][-6:])
            prompt = (
                f"Role: {role}\nRound: {r}/{rounds}\nTask: {task}\n"
                f"Recent context (may be partial):\n{context}\n\n"
                "Contribute succinct bullet points (<=6) with concrete, technical steps and call out any risks or dependencies."
            )
            try:
                note = asyncio.get_event_loop().run_until_complete(server.route_chat(prompt, role=role, intent='agent_collab', temperature=0.2))
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                note = loop.run_until_complete(server.route_chat(prompt, role=role, intent='agent_collab', temperature=0.2)); loop.close()
            note = _compact_text(note, 900)
            history.append({"role": role, "round": r, "note": note})
    # Synthesize
    synth = "\n".join([f"- {h['role']}: {h['note']}" for h in history[-(len(roles)) :]])
    return _compact_text(synth, 4000)


def handle_agent_team_plan_and_code(arguments: Dict[str, Any], server) -> str:
    task_desc = (arguments.get("task") or "").strip()
    if not task_desc:
        raise Exception("'task' is required")
    target_files = arguments.get("target_files") or []
    constraints = (arguments.get("constraints") or "").strip()
    apply_changes = bool(arguments.get("apply_changes", False))
    auto_research_rounds = int(arguments.get("auto_research_rounds", 0))

    # Helper to pull file context: delegate to server
    from server import _read_files_for_context
    file_ctx = _read_files_for_context(target_files)

    try:
        if os.getenv("AGENT_TEAM_FORCE_FALLBACK") == "1":
            raise RuntimeError("forced_fallback")
        from server import _import_crewai_any
        Agent, Crew, Task = _import_crewai_any()
        base_kwargs: dict[str, object] = {"allow_delegation": False, "verbose": False}
        # Per-role LLMs via router-aware helper
        planner_llm = _agent_llm_for_role("Planner")
        coder_llm = _agent_llm_for_role("Coder")
        reviewer_llm = _agent_llm_for_role("Reviewer")
        if planner_llm: base_kwargs_planner = {**base_kwargs, "llm": planner_llm}
        else: base_kwargs_planner = base_kwargs
        if coder_llm: base_kwargs_coder = {**base_kwargs, "llm": coder_llm}
        else: base_kwargs_coder = base_kwargs
        if reviewer_llm: base_kwargs_reviewer = {**base_kwargs, "llm": reviewer_llm}
        else: base_kwargs_reviewer = base_kwargs

        planner = Agent(role="Planner", goal="Break down the task into clear steps and propose code changes.", backstory="Seasoned tech lead.", **base_kwargs_planner)
        coder = Agent(role="Coder", goal="Propose concrete code deltas with fenced diffs.", backstory="Productivity-focused engineer.", **base_kwargs_coder)
        reviewer = Agent(role="Reviewer", goal="Catch defects and suggest fixes.", backstory="Detail-oriented reviewer.", **base_kwargs_reviewer)
        t_plan = Task(description=f"Task: {task_desc}\nConstraints: {constraints}\nContext:\n{file_ctx}", agent=planner)
        t_code = Task(description="Draft code changes as diffs within fenced blocks.", agent=coder)
        t_rev = Task(description="Review the proposed changes, list risks, and suggest refinements.", agent=reviewer)
        crew = Crew(agents=[planner, coder, reviewer], tasks=[t_plan, t_code, t_rev], verbose=False)
        out = str(crew.kickoff())
        resp = _compact_text(out, max_chars=4000)
    except Exception as e:
        # Fallback using routed chat
        try:
            prompt = (
                f"Plan and code for task: {task_desc}. Constraints: {constraints}.\n\n"
                f"Context:\n{file_ctx}\n\n"
                "1) A short plan; 2) Proposed diffs in fenced code; 3) Risks and mitigations."
            )
            # Call class method so tests that monkeypatch it will intercept
            from server import EnhancedLMStudioMCPServer, get_server_singleton
            coro = EnhancedLMStudioMCPServer.make_llm_request_with_retry(get_server_singleton(), prompt, temperature=0.2)
            try:
                resp = asyncio.get_event_loop().run_until_complete(coro)
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(coro); loop.close()
        except Exception as e2:
            resp = f"Error synthesizing plan: {e}; fallback failed: {e2}"
        if apply_changes:
            from server import _apply_proposed_changes
            applied = _apply_proposed_changes(resp, dry_run=False)
            resp += "\n\n[Applied changes]\n" + "\n".join(applied)
    return _compact_text(resp, max_chars=4000)


def handle_agent_team_review_and_test(arguments: Dict[str, Any], server) -> str:
    diff = (arguments.get("diff") or "").strip()
    if not diff:
        raise Exception("'diff' is required")
    context = (arguments.get("context") or "").strip()
    apply_fixes = bool(arguments.get("apply_fixes", False))
    max_loops = int(arguments.get("max_loops", 1))
    test_command = (arguments.get("test_command") or "pytest")

    try:
        from server import _import_crewai_any
        Agent, Crew, Task = _import_crewai_any()
        base_kwargs: dict[str, object] = {"allow_delegation": False, "verbose": False}
        reviewer_llm = _agent_llm_for_role("Reviewer")
        qa_llm = _agent_llm_for_role("QA")
        if reviewer_llm: base_kwargs_reviewer = {**base_kwargs, "llm": reviewer_llm}
        else: base_kwargs_reviewer = base_kwargs
        if qa_llm: base_kwargs_qa = {**base_kwargs, "llm": qa_llm}
        else: base_kwargs_qa = base_kwargs
        reviewer = Agent(role="Reviewer", goal="Review code diff, run tests, propose fixes.", backstory="Pragmatic reviewer.", **base_kwargs_reviewer)
        qa = Agent(role="QA", goal="Surface failing tests and gap coverage.", backstory="QA specialist.", **base_kwargs_qa)
        t_rev = Task(description=f"Review diff and propose fixes. Context:\n{context}\n\nDiff:\n{diff}", agent=reviewer)
        t_qa = Task(description=f"Run tests: {test_command}. Summarize failures.", agent=qa)
        crew = Crew(agents=[reviewer, qa], tasks=[t_rev, t_qa], verbose=False)
        out = str(crew.kickoff())
        resp = _compact_text(out, max_chars=4000)
    except Exception as e:
        try:
            prompt = (
                f"Review the following diff and propose fixes. Then outline test steps for: {test_command}.\n\n"
                f"Context:\n{context}\n\nDiff:\n{diff}"
            )
            from server import EnhancedLMStudioMCPServer, get_server_singleton
            coro = EnhancedLMStudioMCPServer.make_llm_request_with_retry(get_server_singleton(), prompt, temperature=0.2)
            try:
                resp = asyncio.get_event_loop().run_until_complete(coro)
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(coro); loop.close()
        except Exception as e2:
            resp = f"Error synthesizing review: {e}; fallback failed: {e2}"
    return _compact_text(resp, max_chars=4000)


def handle_agent_team_refactor(arguments: Dict[str, Any], server) -> str:
    module_path = (arguments.get("module_path") or "").strip()
    goals = (arguments.get("goals") or "").strip()
    if not module_path:
        raise Exception("'module_path' is required")
    from server import _read_files_for_context
    content = _read_files_for_context([module_path])

    try:
        from crewai import Agent, Crew, Task  # type: ignore
        base_kwargs = {"allow_delegation": False, "verbose": False}
        ref_llm = _agent_llm_for_role("Refactorer")
        qa_llm = _agent_llm_for_role("QA")
        if ref_llm: base_kwargs_ref = {**base_kwargs, "llm": ref_llm}
        else: base_kwargs_ref = base_kwargs
        if qa_llm: base_kwargs_qa = {**base_kwargs, "llm": qa_llm}
        else: base_kwargs_qa = base_kwargs
        refactorer = Agent(role="Refactorer", goal="Propose clearer, modular refactor with docstrings.", backstory="Engineer focused on readability and maintainability.", **base_kwargs_ref)
        qa = Agent(role="QA", goal="Ensure refactor preserves behavior; suggest tests.", backstory="QA who validates behavior.", **base_kwargs_qa)
        t_ref = Task(description=f"Refactor goals: {goals}. Provide a rationale and a refactored version in fenced code.\n\nCurrent content (truncated):\n{content}", agent=refactorer)
        t_qa = Task(description="List behavioral risks, migration steps, and propose tests.", agent=qa)
        crew = Crew(agents=[refactorer, qa], tasks=[t_ref, t_qa], verbose=False)
        out = str(crew.kickoff())
        return _compact_text(out, max_chars=4000)
    except Exception as e:
        try:
            prompt = (
                f"Refactor goals: {goals}. Provide rationale and refactored code.\n\nCurrent content (truncated):\n{content}"
            )
            from server import EnhancedLMStudioMCPServer, get_server_singleton
            coro = EnhancedLMStudioMCPServer.make_llm_request_with_retry(get_server_singleton(), prompt, temperature=0.2)
            try:
                out = asyncio.get_event_loop().run_until_complete(coro)
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                out = loop.run_until_complete(coro); loop.close()
            return _compact_text(out, max_chars=4000)
        except Exception as e2:
            return _compact_text(f"Error: {e}; fallback failed: {e2}", max_chars=4000)

