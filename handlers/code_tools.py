from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from server import (
    _compact_text,
    _execution_enabled,
    _safe_path,
)

# LLM analysis tools via a centralized prompt builder, using server.make_llm_request_with_retry

def _llm_analysis(tool_name: str, arguments: Dict[str, Any], server) -> str:
    code = arguments.get("code", "")
    compact = bool(arguments.get("compact", False))
    code_only = bool(arguments.get("code_only", False))

    if tool_name == "analyze_code":
        analysis_type = arguments.get("analysis_type", "explanation")
        limit = 8 if compact else 12
        prompt = (
            "You are a concise senior engineer. Provide a short, non-repetitive analysis (<= "
            f"{limit} bullets) focused on {analysis_type}.\n\nCode:\n{code}\n"
        )
    elif tool_name == "explain_code":
        limit = 8 if compact else 15
        prompt = (
            f"Explain how this code works in at most {limit} bullet points. Avoid repetition.\n\n"
            f"Code:\n{code}\n"
        )
    elif tool_name == "suggest_improvements":
        focus = arguments.get("focus", "readability, safety, performance")
        limit = 10 if compact else 15
        prompt = (
            f"Suggest concrete improvements focusing on {focus}. Limit to {limit} bullets, avoid repetition.\n\n"
            f"Code:\n{code}\n"
        )
    elif tool_name == "generate_tests":
        focus = arguments.get("focus", "critical paths and edge cases")
        framework = arguments.get("framework", "pytest")
        prompt = (
            f"Generate {framework} tests covering {focus}. Return only code if code_only=true, else a short summary then code.\n\n"
            f"Code:\n{code}\n"
        )
    else:
        prompt = f"Analyze the following code succinctly.\n\nCode:\n{code}\n"

    try:
        out = server.make_llm_request_with_retry(prompt, temperature=0.2)
        try:
            # Support async or sync server methods
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Best effort in running loop: create new loop
                nloop = asyncio.new_event_loop(); asyncio.set_event_loop(nloop)
                try:
                    out = nloop.run_until_complete(out)
                finally:
                    nloop.close()
            else:
                out = loop.run_until_complete(out)
        except Exception:
            pass
        return _compact_text(str(out))
    except Exception as e:
        return f"Error during LLM analysis: {e}"


def handle_analyze_code(arguments: Dict[str, Any], server) -> str:
    return _llm_analysis("analyze_code", arguments, server)


def handle_suggest_improvements(arguments: Dict[str, Any], server) -> str:
    return _llm_analysis("suggest_improvements", arguments, server)


def handle_generate_tests(arguments: Dict[str, Any], server) -> str:
    return _llm_analysis("generate_tests", arguments, server)


def handle_code_hotspots(arguments: Dict[str, Any], server) -> str:
    root = Path(arguments.get("directory") or ".").resolve()
    limit = int(arguments.get("limit", 10))
    stats = []
    for p in root.rglob("*.py"):
        if any(part in {".git", "venv", ".venv", "node_modules", "dist", "build"} for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        loc = txt.count("\n") + 1
        todos = len(re.findall(r"#\s*TODO", txt, flags=re.IGNORECASE))
        funcs = len(re.findall(r"^def\s+\w+\(", txt, flags=re.IGNORECASE | re.MULTILINE))
        branches = txt.count(" if ") + txt.count(" elif ") + txt.count(" else:")
        stats.append((loc, todos, funcs, branches, str(p.relative_to(root))))
    stats.sort(reverse=True)
    header = "loc,todos,funcs,branches,path"
    rows = [header] + [f"{a},{b},{c},{d},{e}" for a,b,c,d,e in stats[:limit]]
    return "\n".join(rows)


def handle_import_graph(arguments: Dict[str, Any], server) -> str:
    root = Path(arguments.get("directory") or ".").resolve()
    limit = int(arguments.get("limit", 10))
    edges = {}
    for p in root.rglob("*.py"):
        if any(part in {".git", "venv", ".venv", "node_modules", "dist", "build"} for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        mod = str(p.relative_to(root)).replace(os.sep, ".").rstrip(".py").rstrip(".")
        for m in re.findall(r"^from\s+([\w\.]+)\s+import\s+", txt, flags=re.MULTILINE):
            edges.setdefault(m, set()).add(mod)
        for m in re.findall(r"^import\s+([\w\.]+)", txt, flags=re.MULTILINE):
            edges.setdefault(m, set()).add(mod)
    ranked = sorted(((k, len(v)) for k,v in edges.items()), key=lambda x: x[1], reverse=True)[:limit]
    header = "module,in_degree"
    rows = [header] + [f"{m},{d}" for m,d in ranked]
    return "\n".join(rows)


def handle_file_scaffold(arguments: Dict[str, Any], server) -> str:
    path = (arguments.get("path") or "").strip()
    kind = (arguments.get("kind") or "").strip()
    desc = (arguments.get("description") or "").strip()
    dry = bool(arguments.get("dry_run", True))
    if not path or kind not in {"module", "test"}:
        raise Exception("path and kind ('module'|'test') are required")
    sp = _safe_path(path)
    template = ("""# {desc}
""" if kind == "module" else """import pytest


def test_placeholder():
    assert True
""")
    out = f"Would write {sp} ({len(template)} chars)"
    if dry:
        return out
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        f.write(template)
    return f"Wrote {sp}"


def handle_execute_code(arguments: Dict[str, Any]) -> str:
    if not _execution_enabled():
        return "Error: Code execution is disabled. Set EXECUTION_ENABLED=true to allow."
    code = arguments.get("code", "")
    language = arguments.get("language", "python")
    timeout = arguments.get("timeout", 30)
    temp_file = None
    try:
        suffix = ".py" if language == "python" else ".js" if language == "javascript" else ".sh" if language == "bash" else None
        if suffix is None:
            return f"Error: Unsupported language {language}"
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(code)
            temp_file = f.name
        if language == "python":
            cmd = ["python", temp_file]
        elif language == "javascript":
            cmd = ["node", temp_file]
        else:
            cmd = ["bash", temp_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = f"Exit code: {result.returncode}\n\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n\n"
        return output
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def handle_run_tests(arguments: Dict[str, Any]) -> str:
    if not _execution_enabled():
        return "Error: Test execution is disabled. Set EXECUTION_ENABLED=true to allow."
    test_command = arguments.get("test_command", "pytest")
    test_file = arguments.get("test_file", "")
    allowed = {"pytest", "python", "python3", "pipenv", "poetry"}
    try:
        parts = test_command.split()
        if not parts or parts[0] not in allowed:
            return "Error: Test command not allowed."
        cmd = parts
        if test_file:
            cmd.append(test_file)
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = f"Exit code: {result.returncode}\n\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n\n"
        return output
    except Exception as e:
        return f"Error running tests: {e}"


def handle_file_read(arguments: Dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    try:
        rp = _safe_path(file_path)
        with open(rp, 'r', encoding='utf-8') as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                content = ''.join(lines[start_idx:end_idx])
            else:
                content = f.read()
        return f"File: {rp}\n\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


def handle_file_write(arguments: Dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    content = arguments.get("content", "")
    mode = arguments.get("mode", "overwrite")
    try:
        rp = _safe_path(file_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        write_mode = 'w' if mode == "overwrite" else 'a'
        with open(rp, write_mode, encoding='utf-8') as f:
            f.write(content)
        return f"Successfully {'wrote' if mode == 'overwrite' else 'appended'} to {rp}"
    except Exception as e:
        return f"Error writing file: {e}"


def handle_file_search(arguments: Dict[str, Any]) -> str:
    pattern = arguments.get("pattern", "")
    file_pattern = arguments.get("file_pattern", "*")
    directory = arguments.get("directory", ".")
    import glob
    try:
        base = _safe_path(directory)
        results = []
        for path in glob.glob(str(base / "**" / file_pattern), recursive=True):
            try:
                rp = Path(path)
                txt = rp.read_text(encoding="utf-8", errors="ignore")
                if re.search(pattern, txt, flags=re.IGNORECASE):
                    results.append(str(rp))
            except Exception:
                continue
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {e}"




def handle_list_directory(arguments: Dict[str, Any], server) -> str:
    """List files and directories under a path (read-only).
    Args:
      - path: base directory (default ".")
      - depth: how many levels to traverse (default 1)
      - include_hidden: include entries starting with '.' (default False)
    Returns a newline-separated listing: type path
    """
    from pathlib import Path
    path = arguments.get("path", ".")
    depth = int(arguments.get("depth", 1))
    include_hidden = bool(arguments.get("include_hidden", False))
    try:
        base = _safe_path(path)
        base_str = str(base)
        rows = []
        def allowed(p: Path) -> bool:
            name = p.name
            return include_hidden or not name.startswith('.')
        if depth <= 0:
            depth = 1
        # level 0
        for p in sorted(base.iterdir()):
            if not allowed(p):
                continue
            kind = "dir" if p.is_dir() else "file"
            rows.append(f"{kind} {p.relative_to(base_str) if p != base else p.name}")
        # deeper levels
        if depth > 1:
            for root, dirs, files in os.walk(base):
                rel_root = Path(root)
                level = len(rel_root.relative_to(base).parts)
                if level >= depth:
                    # prune deeper traversal
                    dirs[:] = []
                # list children
                for d in sorted(dirs):
                    p = rel_root / d
                    if allowed(p):
                        rows.append(f"dir {p.relative_to(base)}")
                for f in sorted(files):
                    p = rel_root / f
                    if allowed(p):
                        rows.append(f"file {p.relative_to(base)}")
        return "\n".join(rows)
    except Exception as e:
        return f"Error listing directory: {e}"


def handle_read_file_range(arguments: Dict[str, Any], server) -> str:
    """Read a specific line range from a file (1-based inclusive indices)."""
    file_path = arguments.get("file_path", "")
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    try:
        if not file_path:
            return "Error: file_path is required"
        if start_line is None or end_line is None:
            return "Error: start_line and end_line are required"
        start = int(start_line); end = int(end_line)
        if start < 1 or end < start:
            return "Error: invalid range"
        rp = _safe_path(file_path)
        with open(rp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # clamp
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        snippet = ''.join(lines[start_idx:end_idx])
        return f"File: {rp}\nLines: {start}-{end}\n\n{snippet}"
    except Exception as e:
        return f"Error reading range: {e}"
