from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _agent_llm_for_role(role: str):
    # Import here to avoid circular dependency
    from server import _decide_backend_for_role, _build_llm_for_backend

    backend = _decide_backend_for_role(role, "agent_team")
    try:
        from crewai import LLM  # type: ignore
    except Exception:
        return None
    return _build_llm_for_backend(backend)


def handle_agent_collaborate(arguments: Dict[str, Any], server) -> str:
    # Import here to avoid circular dependency
    from server import _compact_text

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
    # Import here to avoid circular dependency
    from server import _compact_text, _read_files_for_context

    task_desc = (arguments.get("task") or "").strip()
    if not task_desc:
        raise Exception("'task' is required")
    target_files = arguments.get("target_files") or []
    constraints = (arguments.get("constraints") or "").strip()
    apply_changes = bool(arguments.get("apply_changes", False))
    auto_research_rounds = int(arguments.get("auto_research_rounds", 0))

    # Helper to pull file context: delegate to server
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
    # Import here to avoid circular dependency
    from server import _compact_text

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
    # Import here to avoid circular dependency
    from server import _compact_text, _read_files_for_context

    module_path = (arguments.get("module_path") or "").strip()
    goals = (arguments.get("goals") or "").strip()
    if not module_path:
        raise Exception("'module_path' is required")
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


def handle_generate_detailed_plan(arguments: Dict[str, Any], server) -> str:
    """
    Generate a detailed, specific implementation plan from a high-level goal.

    This handler creates atomic tasks with:
    - Specific implementation details
    - Clear acceptance criteria
    - File-level granularity
    - Dependency tracking
    - Parallel execution opportunities

    Args:
        arguments: {
            "goal": str - High-level objective (e.g., "Build user authentication system")
            "context": dict - Optional context (tech stack, codebase structure, constraints)
        }

    Returns:
        Detailed task plan in markdown format
    """
    import logging
    logger = logging.getLogger(__name__)

    goal = (arguments.get("goal") or "").strip()
    if not goal:
        raise Exception("'goal' is required")

    context = arguments.get("context") or {}

    try:
        # Import plan generator
        from handlers.plan_generator import generate_specific_plan

        # Generate plan (async function, need to run in event loop)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        plan = loop.run_until_complete(generate_specific_plan(goal, context))

        # Convert to markdown
        output = plan.to_markdown()

        # Add summary at the top
        summary = f"""
# ✅ Detailed Implementation Plan Generated

**Goal**: {goal}
**Plan ID**: {plan.plan_id}
**Total Tasks**: {len(plan.tasks)}
**Estimated Time**: {plan.estimated_total_minutes} minutes ({plan.estimated_total_minutes / 60:.1f} hours)
**Specificity**: High (all tasks validated)

---

{output}

---

## 📊 Plan Statistics

- **Total Tasks**: {len(plan.tasks)}
- **Pending**: {sum(1 for t in plan.tasks if t.status.value == 'pending')}
- **Agent Roles**: {', '.join(sorted(set(t.agent_role.value for t in plan.tasks)))}
- **Parallel Groups**: {len(plan.parallel_opportunities)}

## 🎯 Next Steps

1. Review the plan and adjust priorities if needed
2. Start with tasks marked "Ready to Start"
3. Use `agent_team_plan_and_code` to execute individual tasks
4. Track progress and update task statuses

**Tip**: Tasks are designed to be atomic (<15 minutes each) with clear acceptance criteria.
"""

        return summary

    except Exception as e:
        logger.error(f"Failed to generate detailed plan: {e}", exc_info=True)
        return f"❌ Error generating plan: {str(e)}\n\nPlease ensure:\n- Goal is specific enough\n- Context includes relevant information\n- LLM backend is available"


# ============================================================================
# Ephemeral Agent Integration (Phase 2 Priority 1)
# ============================================================================

def handle_request_ephemeral_agent(arguments: Dict[str, Any], server) -> str:
    """
    Request an ephemeral agent with lifecycle management.

    Args:
        role: Agent role (e.g., "backend", "frontend", "testing")
        task_description: What the agent will do
        priority: Higher = more urgent (default 0)
        timeout_seconds: Max wait time in queue (default 60)

    Returns:
        JSON with agent_id or request_id and status
    """
    import json
    from core.ephemeral_agents import get_ephemeral_agent_manager

    role = (arguments.get("role") or "").strip()
    if not role:
        raise Exception("'role' is required")

    task_description = (arguments.get("task_description") or "").strip()
    if not task_description:
        raise Exception("'task_description' is required")

    priority = int(arguments.get("priority", 0))
    timeout_seconds = int(arguments.get("timeout_seconds", 60))

    manager = get_ephemeral_agent_manager()

    # Ensure manager is started
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not manager._running:
        loop.run_until_complete(manager.start())

    # Request agent
    try:
        agent_id = loop.run_until_complete(
            manager.request_agent(
                role=role,
                task_description=task_description,
                priority=priority,
                timeout_seconds=timeout_seconds
            )
        )

        # Check if created immediately or queued
        if agent_id.startswith("agent-"):
            status = "created"
            message = f"Agent {agent_id} created and ready"
        else:
            status = "queued"
            message = f"Request {agent_id} queued (queue size: {manager.get_stats()['queue_size']})"

        result = {
            "status": status,
            "id": agent_id,
            "role": role,
            "message": message,
            "stats": manager.get_stats()
        }

        return json.dumps(result, indent=2)

    except asyncio.QueueFull:
        return json.dumps({
            "status": "error",
            "error": "Queue full - too many pending requests",
            "stats": manager.get_stats()
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to request ephemeral agent: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


def handle_release_ephemeral_agent(arguments: Dict[str, Any], server) -> str:
    """
    Release an ephemeral agent, triggering cleanup.

    Args:
        agent_id: ID of agent to release

    Returns:
        JSON with status
    """
    import json
    from core.ephemeral_agents import get_ephemeral_agent_manager

    agent_id = (arguments.get("agent_id") or "").strip()
    if not agent_id:
        raise Exception("'agent_id' is required")

    manager = get_ephemeral_agent_manager()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(manager.release_agent(agent_id))

        return json.dumps({
            "status": "success",
            "message": f"Agent {agent_id} released and cleaned up",
            "stats": manager.get_stats()
        }, indent=2)

    except Exception as e:
        logger.error(f"Failed to release agent {agent_id}: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


def handle_get_ephemeral_agent_stats(arguments: Dict[str, Any], server) -> str:
    """
    Get statistics about ephemeral agent system.

    Returns:
        JSON with current stats
    """
    import json
    from core.ephemeral_agents import get_ephemeral_agent_manager

    manager = get_ephemeral_agent_manager()
    stats = manager.get_stats()

    # Format nicely
    summary = f"""# Ephemeral Agent System Stats

## Current Status
- **Active Agents**: {stats['active_agents']} / {stats['max_concurrent']}
- **Queue Size**: {stats['queue_size']} / {stats['max_queue_size']}
- **Avg Creation Time**: {stats['avg_creation_time_ms']:.1f}ms

## Lifetime Stats
- **Total Created**: {stats['total_created']}
- **Total Cleaned**: {stats['total_cleaned']}
- **Total Failed**: {stats['total_failed']}

## Active Agents
"""

    if stats['agents']:
        for agent in stats['agents']:
            summary += f"\n- **{agent['agent_id']}** ({agent['role']}): {agent['state']} (age: {agent['age_seconds']:.1f}s)"
    else:
        summary += "\n*No active agents*"

    return summary

