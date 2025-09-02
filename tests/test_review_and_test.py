import importlib
# flake8: noqa


class DummyArgs(dict):
    pass

class DummyServer:
    def __init__(self, exec_enabled=True):
        self.exec_enabled = exec_enabled
    def route_chat(self, *a, **k):
        # Not used directly here
        raise RuntimeError("should not call in unit test")

# Patch helpers for server module

def test_agent_team_review_and_test_happy(monkeypatch):
    srv_mod = importlib.import_module('server')

    # Patch EXECUTION_ENABLED to true via env
    monkeypatch.setenv('EXECUTION_ENABLED', 'true')

    # Patch crewai usage to avoid importing heavy deps
    class DummyAgent:
        def __init__(self, **kwargs):
            pass
    class DummyTask:
        def __init__(self, description, agent):
            self.description = description
            self.agent = agent
    class DummyCrew:
        def __init__(self, agents, tasks, verbose=False):
            self.agents = agents
            self.tasks = tasks
        def kickoff(self):
            # produce a minimal fenced code block for tests
            return (
                "[Review]\nLooks ok\n\n"
                "```python\n# File: tests/test_sample.py\ndef test_one():\n    assert 1==1\n```\n"
            )

    # Expose dummies in globals so the handler picks them up without importing crewai
    monkeypatch.setitem(srv_mod.__dict__, 'Agent', DummyAgent)
    monkeypatch.setitem(srv_mod.__dict__, 'Task', DummyTask)
    monkeypatch.setitem(srv_mod.__dict__, 'Crew', DummyCrew)

    # Patch test execution to return passing
    def fake_handle_test_execution(arguments):
        return "1 passed"
    monkeypatch.setitem(srv_mod.__dict__, 'handle_test_execution', fake_handle_test_execution)

    # Patch file apply to a dry run (no disk writes)
    def fake_apply(text, dry_run=False):
        return ["Would write tests/test_sample.py (42 chars)"]
    monkeypatch.setitem(srv_mod.__dict__, '_apply_proposed_changes', fake_apply)

    # Call handler
    resp = srv_mod.handle_agent_team_review_and_test({
        'diff': 'diff --git a/x b/x',
        'context': 'ctx',
        'apply_fixes': True,
        'max_loops': 1,
        'test_command': 'pytest -q'
    }, DummyServer())
    assert isinstance(resp, str)
    assert "Applied tests" in resp or "Would write" in resp


def test_agent_team_review_and_test_fix_loop(monkeypatch):
    srv_mod = importlib.import_module('server')
    monkeypatch.setenv('EXECUTION_ENABLED', 'true')

    class DummyAgent:
        def __init__(self, **kwargs):
            pass
    class DummyTask:
        def __init__(self, description, agent):
            self.description = description
            self.agent = agent
    class DummyCrew:
        def __init__(self, agents, tasks, verbose=False):
            self.agents = agents
            self.tasks = tasks
        def kickoff(self):
            return "[Review]\nNeeds fix\n\n```python\n# File: tests/test_needs_fix.py\ndef test_fail():\n    assert 1==2\n```\n"

    # Expose dummies in globals so the handler picks them up without importing crewai
    monkeypatch.setitem(srv_mod.__dict__, 'Agent', DummyAgent)
    monkeypatch.setitem(srv_mod.__dict__, 'Task', DummyTask)
    monkeypatch.setitem(srv_mod.__dict__, 'Crew', DummyCrew)

    # First test run fails, then developer proposes a fix
    calls = {
        'runs': 0
    }
    def fake_handle_test_execution(arguments):
        calls['runs'] += 1
        return "1 failed" if calls['runs'] == 1 else "1 passed"
    monkeypatch.setitem(srv_mod.__dict__, 'handle_test_execution', fake_handle_test_execution)

    def fake_apply(text, dry_run=False):
        return ["Would write tests/test_needs_fix.py (56 chars)"]
    monkeypatch.setitem(srv_mod.__dict__, '_apply_proposed_changes', fake_apply)

    # Developer fix proposal via _run_llm
    def fake_run_llm(server, prompt, temperature=0.2, intent=None, role=None):
        return "```python\n# File: src/module.py\nVALUE=42\n```"
    monkeypatch.setitem(srv_mod.__dict__, '_run_llm', fake_run_llm)

    resp = srv_mod.handle_agent_team_review_and_test({
        'diff': 'diff --git a/x b/x',
        'context': 'ctx',
        'apply_fixes': True,
        'max_loops': 2,
        'test_command': 'pytest -q'
    }, DummyServer())
    assert "Fix Proposal" in resp
    assert "Applied fix" in resp or "Would write" in resp

