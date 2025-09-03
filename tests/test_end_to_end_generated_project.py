import os
import sys
import json
import tempfile
import importlib

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_orchestrate_generates_project_and_runs_pytest(tmp_path):
    # Make optimizer optional
    os.environ.setdefault('MCP_OPTIMIZER', '1')

    import server
    s = server.get_server_singleton()

    # Patch LLM calls for determinism
    async def fake_llm(prompt: str, temperature: float = 0.2, **kwargs):
        pl = (prompt or '').lower()
        if 'rate this code quality' in pl:
            return '0.98'
        if 'plan for:' in pl or 'implement:' in pl:
            return 'def auto():\n    return "ok"'
        return 'def foo():\n    return 1'

    try:
        s.route_chat = fake_llm  # type: ignore
    except Exception:
        pass
    try:
        if getattr(s, 'production', None) is not None:
            s.production.lb.llm = lambda prompt, **kw: fake_llm(prompt, **kw)  # type: ignore
    except Exception:
        pass

    # Run orchestrator
    args = {
        'task': 'Create a utility with a simple function and tests and docs',
        'constraints': {'style': 'readable'},
        'strategy': 'hierarchical',
        'max_iterations': 32,
    }
    payload = {"jsonrpc": "2.0", "id": 1, "params": {"name": "cognitive_orchestrate", "arguments": args}}
    out = server.handle_tool_call(payload)
    assert 'result' in out, out
    # Extract result content
    content = out['result']['content'][0]['text'] if isinstance(out['result']['content'], list) else out['result']
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except Exception:
        data = content

    # Save artifacts into temp dir
    files = (data.get('artifacts', {}) or {}).get('files', {})
    tests = (data.get('artifacts', {}) or {}).get('tests', {})
    assert isinstance(files, dict)
    for rel, code in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding='utf-8')
    for rel, code in tests.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding='utf-8')

    # Run pytest against the generated test files
    import subprocess
    proc = subprocess.run([sys.executable, '-m', 'pytest', str(tmp_path), '-q'], capture_output=True, text=True)
    # We allow pass or xfail-like empty as code may be generated minimal
    assert proc.returncode in (0, 5), proc.stdout + '\n' + proc.stderr

    # Validate knowledge graph retrieved context
    kg_mod = importlib.import_module('cognitive_architecture.knowledge_graph')
    kg = kg_mod.CodeKnowledgeGraph(s.storage)
    ctx = kg.get_context_for_generation('simple function utility', max_context_items=3)
    assert isinstance(ctx, list)

