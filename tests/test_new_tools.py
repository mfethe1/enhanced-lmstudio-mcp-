import os
from pathlib import Path

import server as s


def _call(name: str, arguments: dict):
    payload = {"jsonrpc": "2.0", "id": 1201, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return s.handle_tool_call(payload)


def test_backend_diagnostics_runs():
    res = _call("backend_diagnostics", {"roles": ["Planner", "Security Reviewer"], "task_desc": "diagnostics"})
    text = res["result"]["content"][0]["text"]
    assert "Backends available:" in text
    assert "Routing decisions:" in text


def test_file_scaffold_dry_run():
    # Use repo-relative path inside allowed base dir; dry_run ensures no write occurs
    target = "tests/__scaffold_demo.py"
    res = _call("file_scaffold", {"path": target, "kind": "module", "description": "demo", "dry_run": True})
    text = res["result"]["content"][0]["text"]
    assert text.startswith("Would write")


def test_code_hotspots_and_import_graph_smoke(tmp_path, monkeypatch):
    # Create a small temp project
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import os\n# TODO\n\n\ndef foo():\n    if True:\n        return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("from pkg import a\n\n\n", encoding="utf-8")

    res1 = _call("code_hotspots", {"directory": str(tmp_path), "limit": 5})
    t1 = res1["result"]["content"][0]["text"]
    assert t1.splitlines()[0] == "loc,todos,funcs,branches,path"

    res2 = _call("import_graph", {"directory": str(tmp_path), "limit": 5})
    t2 = res2["result"]["content"][0]["text"]
    assert t2.splitlines()[0] == "module,in_degree"


def test_web_search_wraps_deep_research(monkeypatch):
    # Monkeypatch deep_research handler to avoid network and speed up
    def fake_deep(args, server):
        return f"Deep research for: {args.get('query')}"
    monkeypatch.setattr(s, "handle_deep_research", fake_deep)

    res = _call("web_search", {"query": "test topic", "time_limit": 10, "max_depth": 1})
    text = res["result"]["content"][0]["text"]
    assert "Deep research for: test topic" in text

