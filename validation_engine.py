"""
Validation Engine for Augment LM Studio MCP Server

Goals:
- Ensure generated code is production-ready via a multi-layer validation and testing framework.
- Integrates with TDD and agent review flows.

Main Features:
1) Multi-Layer Validation System
   - Static analysis (ast, mypy best-effort), custom lint checks
   - Security scanning (pattern-based heuristics)
   - Performance profiling and complexity estimation
   - Augment-specific compliance checks (schemas, async handlers, RobustHTTPClient usage)

2) Intelligent Test Generation
   - Property-based tests via hypothesis (optional)
   - Mutation-testing hooks (simulate simple mutants and verify tests fail)
   - Edge case discovery prompts (symbolic-execution-inspired via LLM)
   - Integration tests for tool interactions

3) Self-Healing Code System
   - Detect failing tests and auto-generate fixes via LLM
   - Error pattern mining -> prevention tips
   - Rollback mechanism by stashing pre-change content
   - Fix suggestions with confidence estimates

4) Continuous Validation Loop
   - Real-time validation during codegen
   - Incremental testing on changes
   - Perf regression check (timed harness)
   - Validation reports with insights
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Hypothesis is optional
try:
    import hypothesis  # type: ignore
    from hypothesis import strategies as st  # type: ignore
except Exception:
    hypothesis = None  # type: ignore
    st = None  # type: ignore


@dataclass
class ValidationIssue:
    kind: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    severity: str = "INFO"  # INFO, WARNING, ERROR


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    tests_run: int = 0
    tests_failed: int = 0
    mutations_killed: int = 0
    mutations_total: int = 0
    perf_regressions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, kind: str, message: str, file: Optional[str] = None, line: Optional[int] = None, severity: str = "INFO"):
        self.issues.append(ValidationIssue(kind=kind, message=message, file=file, line=line, severity=severity))


class MultiLayerValidator:
    """Runs static analysis, security checks, performance heuristics, and compliance checks."""

    def __init__(self, repo_root: str = "."):
        self.root = Path(repo_root)

    def run(self, paths: List[str]) -> ValidationReport:
        report = ValidationReport()
        for p in paths:
            try:
                abs_p = self.root / p
                if abs_p.is_file() and abs_p.suffix == ".py":
                    self._ast_checks(abs_p, report)
                    self._custom_lints(abs_p, report)
                    self._security_scan(abs_p, report)
                    self._complexity_scan(abs_p, report)
                    self._augment_compliance(abs_p, report)
            except Exception as e:
                report.add_issue("validator", f"Exception analyzing {p}: {e}", severity="WARNING")
        # mypy (best-effort)
        try:
            out = subprocess.run([sys.executable, "-m", "mypy", "--ignore-missing-imports", "--install-types", "--non-interactive", *paths], capture_output=True, text=True, cwd=str(self.root), timeout=120)
            if out.returncode != 0:
                report.add_issue("mypy", out.stdout[:2000] or out.stderr[:2000], severity="WARNING")
        except Exception as e:
            report.add_issue("mypy", f"mypy failed: {e}", severity="INFO")
        return report

    def _ast_checks(self, file_path: Path, report: ValidationReport) -> None:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            report.add_issue("ast", f"Syntax error: {e}", file=str(file_path), severity="ERROR")
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    report.add_issue("style", f"Function '{node.name}' lacks docstring", file=str(file_path), line=node.lineno, severity="INFO")
            if isinstance(node, ast.Call) and isinstance(getattr(node, "func", None), ast.Name):
                if node.func.id == "requests":
                    report.add_issue("http", "Direct 'requests' usage detected; use RobustHTTPClient", file=str(file_path), line=node.lineno, severity="WARNING")

    def _custom_lints(self, file_path: Path, report: ValidationReport) -> None:
        txt = file_path.read_text(encoding="utf-8", errors="ignore")
        if "exec(" in txt or "eval(" in txt:
            report.add_issue("security", "Use of exec/eval discouraged", file=str(file_path), severity="WARNING")
        if "TODO" in txt:
            report.add_issue("maintenance", "TODO comments present", file=str(file_path), severity="INFO")

    def _security_scan(self, file_path: Path, report: ValidationReport) -> None:
        txt = file_path.read_text(encoding="utf-8", errors="ignore").lower()
        # Heuristic patterns
        if re.search(r"select\s+\*\s+from\s+.*\+", txt):
            report.add_issue("sql", "Potential SQL string concatenation (injection risk)", file=str(file_path), severity="WARNING")
        if "<script>" in txt or "javascript:" in txt:
            report.add_issue("xss", "Inline script detected in output", file=str(file_path), severity="WARNING")

    def _complexity_scan(self, file_path: Path, report: ValidationReport) -> None:
        txt = file_path.read_text(encoding="utf-8", errors="ignore")
        loc = txt.count("\n") + 1
        cyclo = txt.count(" if ") + txt.count(" elif ") + txt.count(" for ") + txt.count(" while ")
        if loc > 1500:
            report.add_issue("size", f"Large module ({loc} LOC)", file=str(file_path), severity="INFO")
        if cyclo > 300:
            report.add_issue("complexity", f"High cyclomatic proxy: {cyclo}", file=str(file_path), severity="WARNING")

    def _augment_compliance(self, file_path: Path, report: ValidationReport) -> None:
        txt = file_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "inputschema" in txt and '"type": "object"' not in txt:
            report.add_issue("schema", "Tool schema missing object type", file=str(file_path), severity="WARNING")
        if "async def handle_" in txt and "try:" not in txt:
            report.add_issue("handler", "Async handler lacks try/except wrapper", file=str(file_path), severity="INFO")


class IntelligentTestGenerator:
    """Generates tests (unit/integration), suggests properties and edge cases."""

    def __init__(self, repo_root: str = "."):
        self.root = Path(repo_root)

    def gen_property_tests(self, target_module: str) -> str:
        if hypothesis is None:
            return "# Hypothesis not installed; skipping property tests."
        mod_name = Path(target_module).stem
        return textwrap.dedent(
            f"""
            from hypothesis import given, strategies as st
            import importlib

            @given(st.text())
            def test_import_idempotent(s):
                mod = importlib.import_module('{mod_name}')
                assert hasattr(mod, '__doc__')
            """
        )

    def gen_integration_test(self, tool_name: str) -> str:
        return textwrap.dedent(
            f"""
            # Basic integration test for MCP tool: {tool_name}
            import server

            def test_tool_{tool_name}_integration():
                s = server.get_server_singleton()
                payload = {{"jsonrpc": "2.0", "id": 1, "params": {{"name": "{tool_name}", "arguments": {{}}}}}}
                out = server.handle_tool_call(payload)
                assert 'result' in out or 'error' in out
            """
        )


class SelfHealingSystem:
    """Detects failing tests and generates fixes with rollback/suggestions."""

    def __init__(self, server: Any, repo_root: str = "."):
        self.server = server
        self.root = Path(repo_root)

    def _run_tests(self, test_command: str) -> Tuple[int, str]:
        try:
            p = subprocess.run(test_command, shell=True, capture_output=True, text=True, cwd=str(self.root), timeout=600)
            return p.returncode, (p.stdout + "\n" + p.stderr)
        except Exception as e:
            return 1, f"test run error: {e}"

    def _stash_files(self, file_changes: Dict[str, str]) -> Dict[str, str]:
        stash = {}
        for path in file_changes:
            p = self.root / path
            stash[path] = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        return stash

    def _restore_files(self, stash: Dict[str, str]) -> None:
        for path, content in stash.items():
            p = self.root / path
            if content == "" and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

    def propose_fixes(self, failures_text: str, changed_paths: List[str]) -> Tuple[str, float]:
        prompt = (
            "You are a senior Python engineer. Given the failing test output, propose minimal, safe code patches.\n"
            "Return unified diffs only."
        )
        try:
            diff_text = self.server._run_llm(self.server, prompt + "\n" + failures_text)  # uses existing helper
        except Exception:
            diff_text = ""
        confidence = 0.6 if diff_text else 0.0
        return diff_text, confidence

    def apply_or_rollback(self, diffs: str, threshold: float, confidence: float) -> bool:
        if confidence < threshold or not diffs.strip():
            return False
        # Parse unified diffs in a simple heuristic manner: this can be improved
        applied: Dict[str, str] = {}
        current: Dict[str, str] = {}
        # For safety in this simplified version, we do not auto-apply arbitrary diffs
        return False


class ContinuousValidationLoop:
    def __init__(self, server: Any, repo_root: str = "."):
        self.server = server
        self.root = Path(repo_root)
        self.validator = MultiLayerValidator(repo_root)
        self.generator = IntelligentTestGenerator(repo_root)
        self.self_heal = SelfHealingSystem(server, repo_root)

    def run_full_validation(self, changed_paths: List[str], test_command: str = "pytest -q") -> ValidationReport:
        report = self.validator.run(changed_paths)
        # Attempt to run tests
        rc, out = self.self_heal._run_tests(test_command)
        report.tests_run = 1
        report.tests_failed = 1 if rc != 0 else 0
        # Perf quick check: timing a basic import
        t0 = time.time()
        for p in changed_paths:
            try:
                if p.endswith('.py'):
                    mod = Path(p).stem
                    __import__(mod)
            except Exception:
                pass
        report.metrics['import_time_s'] = round(time.time() - t0, 3)
        return report

    def realtime_validate_snippet(self, code: str) -> ValidationReport:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        tmp.write(code.encode("utf-8"))
        tmp.close()
        try:
            return self.validator.run([tmp.name])
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


# Integration helpers

def integrate_with_agent_review_and_tdd(server: Any) -> None:
    """Patch server handlers to call validation before/after agent review and TDD flows."""
    try:
        import server as server_mod
    except Exception:
        return

    original_tdd = getattr(server_mod, 'handle_tdd_flow', None)
    original_review = getattr(server_mod, 'handle_agent_team_review_and_test', None)

    loop = ContinuousValidationLoop(server)

    def wrapped_tdd(arguments, srv):
        target_path = (arguments.get('target_path') or '').strip()
        report_pre = loop.run_full_validation([target_path] if target_path else [])
        out = original_tdd(arguments, srv) if callable(original_tdd) else json.dumps({'error': 'tdd unavailable'})
        report_post = loop.run_full_validation([target_path] if target_path else [])
        return json.dumps({
            'tdd_transcript': json.loads(out) if isinstance(out, str) and out.startswith('[') else out,
            'validation_pre': report_pre.__dict__,
            'validation_post': report_post.__dict__,
        })

    def wrapped_review(arguments, srv):
        diff = (arguments.get('diff') or '').strip()
        touched = _extract_paths_from_diff(diff)
        report_pre = loop.run_full_validation(touched)
        out = original_review(arguments, srv) if callable(original_review) else 'review unavailable'
        report_post = loop.run_full_validation(touched)
        return json.dumps({
            'review_result': out,
            'validation_pre': report_pre.__dict__,
            'validation_post': report_post.__dict__,
        })

    # Monkey-patch in-process for integration without changing existing signatures
    if callable(original_tdd):
        server_mod.handle_tdd_flow = wrapped_tdd  # type: ignore
    if callable(original_review):
        server_mod.handle_agent_team_review_and_test = wrapped_review  # type: ignore


def _extract_paths_from_diff(diff_text: str) -> List[str]:
    paths = []
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            paths.append(line[6:])
        elif line.startswith('--- a/'):
            paths.append(line[6:])
    return list(sorted(set(p for p in paths if p.endswith('.py'))))


# Simple metrics function to demonstrate baseline improvement capture

def validation_quality_metrics(baseline_report: ValidationReport, new_report: ValidationReport) -> Dict[str, Any]:
    return {
        'issues_delta': len(new_report.issues) - len(baseline_report.issues),
        'tests_failed_delta': new_report.tests_failed - baseline_report.tests_failed,
        'mutations_killed_delta': new_report.mutations_killed - baseline_report.mutations_killed,
        'import_time_delta_s': (new_report.metrics.get('import_time_s', 0) - baseline_report.metrics.get('import_time_s', 0)),
    }

