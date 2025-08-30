import asyncio
# Optional robust router integration
try:
    from improvements.router import RouterAgent as RobustRouterAgent, Backend as RouterBackend, TaskProfile as RouterTaskProfile
except Exception:  # noqa: E722
    RobustRouterAgent = None
    RouterBackend = None
    RouterTaskProfile = None

import json
import sys
import os
import subprocess
import tempfile
import traceback
import hashlib
import time
from typing import Any, Dict, List, Optional
import aiohttp
import logging
from pathlib import Path
import ast
import re
from uuid import uuid4

# Enhanced modular components
from enhanced_agent_teams import decide_backend_for_role as _enh_decide_backend_for_role  # delegate to enhanced module
from enhanced_mcp_tools import merged_tools as _merged_tools
from audit_logger import ImmutableAuditLogger, AuditLevel, ActionType, ComplianceRule, AttorneyStyleReviewer
from workflow_composer import WorkflowComposer

# Import enhanced storage facade (selects SQLite/Postgres per env)
from enhanced_mcp_storage import EnhancedMCPStorage as _LegacyEnhanced
from enhanced_mcp_storage_v2 import StorageSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Singleton server instance (P1)
_server_singleton: Optional["EnhancedLMStudioMCPServer"] = None


# Globals for audit/workflow subsystems
_audit_logger: Optional[ImmutableAuditLogger] = None
_workflow: Optional[WorkflowComposer] = None

def get_server_singleton():
    global _server_singleton, _audit_logger, _workflow
    if _server_singleton is None:
        _server_singleton = EnhancedLMStudioMCPServer()
        # Initialize audit logger and workflow composer lazily with the server's storage
        try:
            _audit_logger = ImmutableAuditLogger(_server_singleton.storage)
        except Exception:
            _audit_logger = None
        try:
            _workflow = WorkflowComposer(_server_singleton, _server_singleton.storage)
        except Exception:
            _workflow = None
    return _server_singleton


# --- Response formatting helpers (text-only, MCP compliant) ---

def _compact_text(text: str, max_chars: int = 4000) -> str:
    """Compact overly verbose LLM output and cap size.
    - Trim to max_chars
    - Remove immediate duplicate consecutive lines
    - Collapse excessive blank lines
    """
    if not isinstance(text, str):
        text = str(text)
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    compact = []
    last = None
    blank_streak = 0
    for ln in lines:
        if ln == last:
            # skip exact consecutive duplicates
            continue
        if ln.strip() == "":
            blank_streak += 1
            # Allow at most one blank line in a row
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        compact.append(ln)
        last = ln
    out = "\n".join(compact)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out


def _to_text_content(result) -> str:
    """Always produce a text string suitable for MCP content.
    Dicts are JSON-stringified, other types are cast to str, then compacted.
    """
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False)
        except Exception:
            text = str(result)
    return _compact_text(text)

def _sanitize_llm_output(s: str) -> str:
    """Strip LM Studio debug wrappers and extract message content if a chat.completion JSON leaks into text.
    If JSON-like, try to parse and return choices[0].message.content. Otherwise return original.
    """
    if not isinstance(s, str):
        return _to_text_content(s)
    txt = s.strip()
    # Remove common verbose prefixes
    if txt.startswith("[openai/") or txt.startswith("[deepseek/"):
        # Drop the first line prefix
        nl = txt.find("\n")
        if nl != -1:
            txt = txt[nl+1:].lstrip()
    # Attempt to parse JSON and extract content
    if (txt.startswith("{") and '"choices"' in txt and '"message"' in txt) or '"chat.completion"' in txt:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and obj.get("choices"):
                msg = obj["choices"][0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        except Exception:
            pass
    return txt


def _mcp_text_response(result) -> Dict[str, Any]:
    """Build a valid MCP result payload with text content only."""
    text = _to_text_content(result)
    return {"content": [{"type": "text", "text": text}]}

def _extract_code_from_text(text: str) -> str:
    """Extract fenced code blocks from text; fallback to heuristic lines.
    Returns concatenated code suitable for a single file output.
    """
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_lang = None
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = stripped[3:].strip() or None
                continue
            else:
                in_fence = False
                fence_lang = None
                continue
        if in_fence:
            out.append(ln)
    if out:
        return "\n".join(out).strip()
    # Fallback: keep only lines that look like python tests
    keep = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from ") or s.startswith("def test_") or s.startswith("assert "):
            keep.append(ln)
    return "\n".join(keep).strip()

# --- Secrets loader (local .secrets/.env.local) ---
_DEF_SECRETS_PATH = Path('.secrets/.env.local')

def _load_local_secrets():
    try:
        if _DEF_SECRETS_PATH.exists():
            logger.info(f"Loading local secrets from {_DEF_SECRETS_PATH}")
            for line in _DEF_SECRETS_PATH.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                # Do not overwrite existing env
                if k and (k not in os.environ or not os.environ[k]):
                    os.environ[k] = v
    except Exception as e:
        logger.warning(f"Failed to load local secrets: {e}")

_load_local_secrets()

def _build_sequential_thinking_prompt(
    session_id: str,
    server,
    current_thought: str,
    thought_number: int,
    total_thoughts: int,
    is_revision: bool,
    problem: str | None = None,
    max_history: int = 10,
) -> str:
    """Compose a concise prompt for the LLM to suggest the next thought.

    Includes recent session thoughts, current problem (if provided), and constraints.
    """
    try:
        history = server.storage.get_thinking_session(session_id) or []
    except Exception:
        history = []

    # Use only the latest max_history entries
    recent = history[-max_history:]
    history_lines = []
    for step in recent:
        try:
            tn = step.get("thought_number")
            th = step.get("thought", "").strip()
            rev = step.get("is_revision")
            history_lines.append(f"- T{tn}{' (rev)' if rev else ''}: {th}")
        except Exception:
            continue
    history_str = "\n".join(history_lines) if history_lines else "(no prior thoughts)"

    pb = problem.strip() if isinstance(problem, str) else ""
    current = (current_thought or "").strip()

    prompt = (
        "You are an expert planner facilitating a sequential reasoning session.\n"
        "Given the problem, session context, and current thought, suggest the BEST next thought(s).\n"
        "Constraints:\n"
        "- Be concrete and actionable\n"
        "- Avoid repetition; build on context\n"
        "- 1-3 bullets max; each <= 140 chars\n"
        "- If this is a revision, refine the current thought\n\n"
        f"Problem: {pb if pb else 'N/A'}\n"
        f"Session: T{thought_number}/{total_thoughts} (revision={is_revision})\n\n"
        f"Recent thoughts:\n{history_str}\n\n"
        f"Current thought:\n{current}\n\n"
        "Output ONLY the suggested next thought(s) as bullets."
    )
    return prompt


# TDD helpers
_DEF_TEST_CMD = os.getenv("TEST_COMMAND", "pytest -q")


# Safety and validation helpers (P0)
class ValidationError(Exception):
    """Raised for invalid user inputs or disallowed operations."""
    pass

# Allowed base directory for all filesystem operations
_BASE_DIR = Path(os.getenv("ALLOWED_BASE_DIR", os.getcwd())).resolve()

# Firecrawl configuration: do not embed secrets; require FIRECRAWL_API_KEY via environment
FIRECRAWL_BASE_URL_STATIC = "https://api.firecrawl.dev"


def _safe_path(p: str) -> Path:
    """Return a resolved path if and only if it is inside the allowed base dir."""
    if not p or not isinstance(p, str):
        raise ValidationError("file_path must be a non-empty string")
    rp = Path(p).resolve()
    try:
        rp.relative_to(_BASE_DIR)
    except Exception:
        raise ValidationError("Path outside allowed base directory")
    return rp


def _safe_directory(p: str) -> Path:
    """Validate directory path is within base dir and exists."""
    rp = _safe_path(p)
    if not rp.exists() or not rp.is_dir():
        raise ValidationError("Directory does not exist or is not a directory")
    return rp


def _execution_enabled() -> bool:
    v = os.getenv("EXECUTION_ENABLED", "false").strip().lower()
    return v in {"1", "true", "yes", "on"}

class EnhancedLMStudioMCPServer:
    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
        self.working_directory = os.getcwd()

        # Initialize persistent storage: legacy facade wrapped by V2 selector when enabled
        self.storage = StorageSelector(_LegacyEnhanced())
        # Initialize robust router agent if available (after storage is ready)
        try:
            self._router_agent = RobustRouterAgent(self.storage, self.base_url) if RobustRouterAgent else None
        except Exception:
            self._router_agent = None
        # Performance monitoring settings
        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.2"))  # seconds

        # Router logs (last ~200 decisions)
        self._router_log = []


        # Router timing controls
        self._router_last_call = {"lmstudio": 0.0, "openai": 0.0, "anthropic": 0.0}
        self._router_min_interval = 1.0 / float(os.getenv("ROUTER_RATE_LIMIT_TPS", "5"))

    async def _lmstudio_request_with_retry(self, prompt: str, temperature: float = 0.35, retries: int = 2, backoff: float = 0.5) -> str:
        """Direct LM Studio request with simple retry/backoff (no routing)"""
        import requests
        attempt = 0
        last_err = None
        while attempt <= retries:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 4000
                    },
                    timeout=(60, 240),
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"].get("content", "")
                        return _sanitize_llm_output(content)
                    return "Error: No response from model"
                last_err = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_err = str(e)
            attempt += 1
            await asyncio.sleep(backoff * attempt)
        return f"Error: LLM request failed after {retries+1} attempts: {last_err}"

    async def make_llm_request_with_retry(self, prompt: str, temperature: float = 0.35, retries: int = 2, backoff: float = 0.5) -> str:
        """Centralized LLM request; optionally routes across providers if enabled."""
        if os.getenv("ROUTER_WRAP_MLR", "1").lower() in {"1","true","yes","on"}:
            return await self.route_chat(prompt, temperature=temperature)
        return await self._lmstudio_request_with_retry(prompt, temperature=temperature, retries=retries, backoff=backoff)


    def _router_wait(self, backend: str):
        now = time.time()
        last = self._router_last_call.get(backend, 0.0)
        gap = now - last
        if gap < self._router_min_interval:
            time.sleep(self._router_min_interval - gap)
        self._router_last_call[backend] = time.time()


    async def decide_backend_via_router(self, prompt: str, intent: str | None = None, role: str | None = None) -> dict:
        """Ask a lightweight local router agent (LM Studio) to choose a backend.
        Returns a dict like {backend, reason, confidence, preferred_model}.
        Backends: anthropic | openai | lmstudio.
        """
        schema_hint = (
            "Respond ONLY in JSON with keys: backend(one of 'anthropic','openai','lmstudio'), "
            "reason, confidence (0..1), preferred_model (string)."
        )
        router_prompt = (
            f"You are a router agent for coding tasks. Decide best backend.\n"
            f"Intent: {intent or 'general'} | Role: {role or 'none'}\n"
            f"Task preview (truncated):\n{_compact_text(prompt, 700)}\n\n{schema_hint}"
        )
        try:
            raw = await self.make_llm_request_with_retry(router_prompt, temperature=0.0, retries=1, backoff=0.2)
            try:
                data = json.loads(raw.strip())
                if isinstance(data, dict) and data.get('backend') in {'anthropic','openai','lmstudio'}:
                    return data
            except Exception:
                pass
        except Exception:
            pass
        return {"backend": None, "reason": "router_failed"}

    async def route_chat(self, prompt: str, *, intent: str | None = None, complexity: str | None = "auto", role: str | None = None, preferred_backend: str | None = None, temperature: float = 0.2) -> str:
        """Intelligent multi-model routing with fallback chain and optional router agent.
        Preference order: preferred_backend -> router_agent -> heuristic -> lmstudio. Fallbacks: lmstudio -> openai -> anthropic.
        """
        decision = {"intent": intent, "role": role, "temperature": temperature}
        model_override = None
        # 1) Preferred backend overrides
        backend = (preferred_backend or "").lower()
        source = "preferred" if backend else None
        # 2) Robust RouterAgent (from improvements/) if available
        if not backend and self._router_agent is not None:
            try:
                task = await self._router_agent.analyze_task(prompt, intent=intent, role=role)
                if complexity and complexity != "auto":
                    task.complexity = complexity
                dec = await self._router_agent.decide_routing(task)
                backend = dec.backend.value if hasattr(dec, 'backend') else None
                model_override = getattr(dec, 'model', None)
                source = "router_agent"
                decision.update({"agent": {"backend": backend, "model": model_override, "confidence": getattr(dec, 'confidence', None)}})
            except Exception as e:
                logger.warning("RouterAgent decision failed: %s", e)
        # 3) Lightweight JSON router agent (LM Studio) if enabled and still unknown
        if not backend and os.getenv("ROUTER_USE_AGENT", "1").lower() in {"1","true","yes","on"}:
            agent_dec = await self.decide_backend_via_router(prompt, intent=intent, role=role)
            b = (agent_dec.get("backend") or "").lower()
            if b in {"anthropic","openai","lmstudio"}:
                backend, source = b, "router_agent"
                model_override = agent_dec.get("preferred_model") or model_override
                decision.update({"agent_json": agent_dec})
        # 4) Heuristic fallback if still unknown
        if not backend:
            text = (prompt or "").lower()
            complex_markers = ["architecture", "design", "vulnerability", "formal", "proof", "theorem", "deep analysis", "multi-step"]
            is_complex = any(w in text for w in complex_markers) or (complexity == "high") or (intent in {"analysis","architecture","validation"})
            if is_complex and os.getenv("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            elif is_complex and os.getenv("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "lmstudio"
            source = source or "heuristic"
        decision["backend"] = backend
        decision["source"] = source

        def _is_complex_text(prompt_text: str, intent_str: str | None, complexity_hint: str | None) -> bool:
            text = (prompt_text or "").lower()
            complex_markers = [
                "architecture", "design", "vulnerability", "formal", "proof",
                "theorem", "deep analysis", "multi-step", "refactor", "optimize",
            ]
            return (
                any(w in text for w in complex_markers)
                or (complexity_hint == "high")
                or (intent_str in {"analysis", "architecture", "validation"})
            )

        def _select_anthropic_model(role_str: str | None, intent_str: str | None, complexity_hint: str | None, prompt_text: str, agent_conf: float | None, default_model: str) -> str:
            """Smart switch between Sonnet/Opus.
            Env flags:
              - ANTHROPIC_SMART_SWITCH (default: on)
              - OVERSEER_USE_OPUS (default: on) + ANTHROPIC_MODEL_OVERSEER (default: claude-4-opus)
              - ANTHROPIC_MODEL_COMPLEX (default: claude-4-opus)
              - OPUS_FOR_ANALYSIS (default: on)
              - OPUS_FOR_LOW_CONF (default: on) + LOW_CONF_THRESHOLD (default: 0.5)
            """
            try:
                if os.getenv("ANTHROPIC_SMART_SWITCH", "1").strip().lower() not in {"1","true","yes","on"}:
                    return default_model
                role_l = (role_str or "").strip().lower()
                intent_l = (intent_str or "").strip().lower()
                # 1) Overseer/Reviewer override
                if os.getenv("OVERSEER_USE_OPUS", "1").strip().lower() in {"1","true","yes","on"} and role_l in {"overseer","reviewer"}:
                    return os.getenv("ANTHROPIC_MODEL_OVERSEER", "claude-4-opus")
                # 2) Complexity-based
                if _is_complex_text(prompt_text, intent_l, complexity_hint):
                    return os.getenv("ANTHROPIC_MODEL_COMPLEX", "claude-4-opus")
                # 3) Intent-based analysis
                if intent_l in {"analysis","architecture","validation"} and os.getenv("OPUS_FOR_ANALYSIS", "1").strip().lower() in {"1","true","yes","on"}:
                    return os.getenv("ANTHROPIC_MODEL_COMPLEX", "claude-4-opus")
                # 4) Low-confidence from router agent
                if os.getenv("OPUS_FOR_LOW_CONF", "1").strip().lower() in {"1","true","yes","on"}:
                    try:
                        thr = float(os.getenv("LOW_CONF_THRESHOLD", "0.5"))
                        if agent_conf is not None and float(agent_conf) < thr:
                            return os.getenv("ANTHROPIC_MODEL_COMPLEX", "claude-4-opus")
                    except Exception:
                        pass
            except Exception:
                pass
            return default_model

        decision["model_override"] = model_override
        logger.info("Router selected backend=%s via %s intent=%s role=%s", backend, source, intent, role)

        # Per-backend params
        max_tokens = int(os.getenv("ROUTER_MAX_TOKENS", "4000"))
        t0 = time.time()
        try:
            if backend == "openai" and os.getenv("OPENAI_API_KEY"):
                self._router_wait("openai")
                import requests
                base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                model = (model_override or os.getenv("OPENAI_MODEL", "gpt5"))
                decision["model"] = model
                r = requests.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
                    timeout=(60, 240)
                )
                if r.status_code == 200:
                    data = r.json()
                    out = _sanitize_llm_output(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                    decision.update({"latency_ms": int((time.time()-t0)*1000), "success": True})
                    self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
                    return out
                logger.warning("OpenAI error %s: %s", r.status_code, r.text[:200])
                decision.update({"latency_ms": int((time.time()-t0)*1000), "success": False, "error": f"http {r.status_code}"})
                self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
            if backend == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
                self._router_wait("anthropic")
                import requests
                base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
                default_model = (model_override or os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet"))
                agent_conf = None
                try:
                    agent_conf = decision.get("agent", {}).get("confidence") or decision.get("agent_json", {}).get("confidence")
                except Exception:
                    pass
                model = select_anthropic_model(role, intent, complexity, prompt, agent_conf, default_model) if not model_override else default_model
                decision["model"] = model
                r = requests.post(
                    f"{base}/messages",
                    headers={
                        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                        "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                        "content-type": "application/json",
                    },
                    json={"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]},
                    timeout=(60, 240)
                )
                if r.status_code == 200:
                    data = r.json()
                    parts = data.get("content", [])
                    txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                    out = _sanitize_llm_output(txt)
                    decision.update({"latency_ms": int((time.time()-t0)*1000), "success": True})
                    self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
                    return out
                logger.warning("Anthropic error %s: %s", r.status_code, r.text[:200])
                decision.update({"latency_ms": int((time.time()-t0)*1000), "success": False, "error": f"http {r.status_code}"})
                self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
            # Default LM Studio
            self._router_wait("lmstudio")
            out = await self._lmstudio_request_with_retry(prompt, temperature=temperature)
            decision.update({"model": self.model_name, "latency_ms": int((time.time()-t0)*1000), "success": True})
            self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
            return out
        except Exception as e:
            logger.warning("Router primary failed (%s). Falling back: %s", backend, e)
            decision.update({"latency_ms": int((time.time()-t0)*1000), "success": False, "error": str(e), "fallback": True})
            self._router_log.append(decision); self._router_log[:] = self._router_log[-200:]
            # Fallback chain
            try:
                self._router_wait("lmstudio")
                out = await self._lmstudio_request_with_retry(prompt, temperature=temperature)
                return out
            except Exception as e2:
                try:
                    if os.getenv("OPENAI_API_KEY"):
                        return await self.route_chat(prompt, preferred_backend="openai", temperature=temperature)
                except Exception:
                    pass
                try:
                    if os.getenv("ANTHROPIC_API_KEY"):
                        return await self.route_chat(prompt, preferred_backend="anthropic", temperature=temperature)
                except Exception:
                    pass
                return f"Error: routing failed: {e2}"




    def monitor_performance(self, func_name: str):
        """Decorator for monitoring tool performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error_message = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    # Log error to storage
                    self.storage.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        tool_name=func_name,
                        stack_trace=traceback.format_exc()
                    )
                    raise
                finally:
                    execution_time = time.time() - start_time

                    # Log performance metrics
                    self.storage.log_performance(
                        tool_name=func_name,
                        execution_time=execution_time,
                        success=success,
                        error_message=error_message
                    )

                    # Alert on performance issues
                    if execution_time > self.performance_threshold:
                        logger.warning(f"Performance alert: {func_name} took {execution_time:.3f}s (threshold: {self.performance_threshold}s)")

            return wrapper
        return decorator

    async def make_llm_request(self, prompt: str, temperature: float = 0.35) -> str:
        """Make request to LM Studio with enhanced error handling"""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 4000
                },
                timeout=(60, 240),
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "Error: No response from model"
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Error making LLM request: {e}")
            return f"Error: {str(e)}"

def handle_message(message):
    """Handle incoming MCP messages"""
    try:
        method = message.get("method")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True}
                    },
                    "serverInfo": {
                        "name": "enhanced-lmstudio-assistant",
                        "version": "2.1.0"
                    }
                }
            }
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": get_all_tools()
            }
        elif method == "tools/call":
            return handle_tool_call(message)

        else:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {}
            }
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32603,
                "message": "Internal error"
            }
        }

def get_all_tools():
    """Return all available tools with enhanced capabilities"""
    base = {
        "tools": [
            # Research & Planning
            {
                "name": "deep_research",
                "description": "Hybrid deep research: Firecrawl for sources, CrewAI agents for analysis and synthesis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Research question or topic"},
                        "max_depth": {"type": "integer", "description": "Max recursion depth", "default": 8},
                        "time_limit": {"type": "integer", "description": "Time limit (seconds)", "default": 300}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_research_details",
                "description": "Retrieve stored artifacts for a previous deep research run by research_id",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "research_id": {"type": "string", "description": "The research run identifier"}
                    },
                    "required": ["research_id"]
                }
            },

            # Agentic Teams
            {
                "name": "agent_team_plan_and_code",
                "description": "Planner, Coder, Reviewer propose a plan and code changes (returns patches and tests; optional apply).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "What to build or change"},
                        "target_files": {"type": "array", "items": {"type": "string"}, "description": "Files to read for context (optional)"},
                        "constraints": {"type": "string", "description": "Constraints/acceptance criteria (optional)"},
                        "apply_changes": {"type": "boolean", "description": "If true, write proposed contents to disk", "default": False},
                        "auto_research_rounds": {"type": "integer", "description": "If >0, run a short refine pass after research", "default": 0}
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "agent_team_review_and_test",
                "description": "Reviewer and Test Author analyze changes, propose tests, and outline a test plan.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "diff": {"type": "string", "description": "Proposed diff or code snippet to review"},
                        "context": {"type": "string", "description": "Additional context (optional)"},
                        "apply_fixes": {"type": "boolean", "description": "If true, apply proposed tests/fixes", "default": False},
                        "max_loops": {"type": "integer", "description": "Test/fix iterations", "default": 1},
                        "test_command": {"type": "string", "description": "Command to run tests (e.g., 'pytest -q')"}
                    },
                    "required": ["diff"]
                }
            },
            {
                "name": "agent_team_refactor",
                "description": "Refactorer and QA agents suggest modular refactors with docstrings and tests.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "module_path": {"type": "string", "description": "Module to refactor"},
                        "goals": {"type": "string", "description": "Refactor goals (readability, modularity, TDD)"}
                    },
                    "required": ["module_path", "goals"]
                }
            },

            # Code Understanding & Improvement
            {"name": "analyze_code", "description": "Analyze code for bugs/improvements/explanations", "inputSchema": {"type":"object","properties":{"code":{"type":"string"},"analysis_type":{"type":"string","enum":["bugs","optimization","explanation","refactor"]}}, "required":["code","analysis_type"]}},
            {"name": "suggest_improvements", "description": "Suggestions to improve code", "inputSchema": {"type":"object","properties":{"code":{"type":"string"}} , "required":["code"]}},
            {"name": "generate_tests", "description": "Generate unit tests for code", "inputSchema": {"type":"object","properties":{"code":{"type":"string"},"framework":{"type":"string","default":"pytest"}}, "required":["code"]}},

            # Execution & Testing
            {"name": "execute_code", "description": "Execute code in a temp sandbox", "inputSchema": {"type":"object","properties":{"code":{"type":"string"},"language":{"type":"string","enum":["python","javascript","bash"],"default":"python"},"timeout":{"type":"integer","default":30}}, "required":["code"]}},
            {"name": "run_tests", "description": "Run project tests", "inputSchema": {"type":"object","properties":{"test_command":{"type":"string"},"test_file":{"type":"string"}}, "required":["test_command"]}},

            # Memory
            {"name": "store_memory", "description": "Store info in persistent memory", "inputSchema": {"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"},"category":{"type":"string","default":"general"}}, "required":["key","value"]}},
            {"name": "retrieve_memory", "description": "Retrieve info from persistent memory", "inputSchema": {"type":"object","properties":{"key":{"type":"string"},"category":{"type":"string"},"search_term":{"type":"string"}}}},

            # System
            {"name": "health_check", "description": "Check server health and connectivity", "inputSchema": {"type":"object","properties":{}}},
            {"name": "get_version", "description": "Get server version and system information", "inputSchema": {"type":"object","properties":{}}},

            # Research helper
            {"name": "propose_research", "description": "Propose research queries (and reasons) for a given problem/context.", "inputSchema": {"type":"object","properties":{"problem":{"type":"string"},"context":{"type":"string"},"max_queries":{"type":"integer","default":3}}, "required":["problem"]}}
,

            # Web research (shallow)
            {"name": "web_search", "description": "Quick research using Firecrawl-backed deep_research (shallow).", "inputSchema": {"type":"object","properties":{"query":{"type":"string"},"time_limit":{"type":"integer","default":60},"max_depth":{"type":"integer","default":1}}, "required":["query"]}},

            # Diagnostics & analysis
            {"name": "backend_diagnostics", "description": "Report available LLM backends and per-role routing for a sample task.", "inputSchema": {"type":"object","properties":{"roles":{"type":"array","items":{"type":"string"}},"task_desc":{"type":"string"}}}},
            {"name": "code_hotspots", "description": "Analyze repository for hotspots (LOC, functions, TODOs, branches).", "inputSchema": {"type":"object","properties":{"directory":{"type":"string","default":"."},"limit":{"type":"integer","default":10}}}},
            # Audit and Workflow (Rec 6 & 7)
            {
                "name": "audit_search",
                "description": "Search immutable audit trail with simple filters",
            },

            # Smart task router (function router)
            {
                "name": "smart_task",
                "description": "Smart router: send a natural instruction and it will select and invoke the right tool(s).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "context": {"type": "string"},
                        "dry_run": {"type": "boolean", "default": False}
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "smart_plan_execute",
                "description": "Plan a multi-step tool workflow and (optionally) execute it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "context": {"type": "string"},
                        "dry_run": {"type": "boolean", "default": False},
                        "max_steps": {"type": "integer", "default": 4}
                    },
                    "required": ["instruction"]
                }
            },


            {
                "name": "audit_verify_integrity",
                "description": "Verify the audit chain integrity",
                "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 200}}, "required": []}
            },
            {
                "name": "audit_add_rule",
                "description": "Add a compliance rule (simple substring pattern)",
                "inputSchema": {"type": "object", "properties": {"rule_id": {"type": "string"}, "name": {"type": "string"}, "pattern": {"type": "string"}, "severity": {"type": "string", "default": "info"}, "action_required": {"type": "string", "default": "log"}, "retention_years": {"type": "integer", "default": 7}}, "required": ["rule_id","name","pattern"]}
            },
            {
                "name": "audit_compliance_report",
                "description": "Generate a basic compliance report",
                "inputSchema": {"type": "object", "properties": {"tenant_id": {"type": "string"}}, "required": []}
            },
            {
                "name": "audit_review_action",
                "description": "Attorney-style compliance review of an action",
                "inputSchema": {"type": "object", "properties": {"action_desc": {"type": "string"}, "context": {"type": "object"}, "domain": {"type": "string", "default": "general"}}, "required": ["action_desc"]}
            },
            {
                "name": "workflow_create",
                "description": "Create a new workflow",
                "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}}, "required": ["name"]}
            },
            {
                "name": "workflow_add_node",
                "description": "Add a node to a workflow",
                "inputSchema": {"type": "object", "properties": {"workflow_id": {"type": "string"}, "node_type": {"type": "string"}, "name": {"type": "string"}, "config": {"type": "object"}}, "required": ["workflow_id","node_type","name"]}
            },
            {
                "name": "workflow_connect_nodes",
                "description": "Connect two nodes in a workflow",
                "inputSchema": {"type": "object", "properties": {"workflow_id": {"type": "string"}, "source_node_id": {"type": "string"}, "target_node_id": {"type": "string"}}, "required": ["workflow_id","source_node_id","target_node_id"]}
            },
            {
                "name": "workflow_explain",
                "description": "Explain a workflow using LLM",
                "inputSchema": {"type": "object", "properties": {"workflow_id": {"type": "string"}}, "required": ["workflow_id"]}
            },
            {
                "name": "workflow_execute",
                "description": "Execute a workflow (mock execution)",
                "inputSchema": {"type": "object", "properties": {"workflow_id": {"type": "string"}, "inputs": {"type": "object"}}, "required": ["workflow_id"]}
            },
            {
                "name": "agent_collaborate",
                "description": "Run a multi-round, multi-role collaboration session and synthesize a plan.",
                "inputSchema": {"type":"object","properties":{
                  "task":{"type":"string"},
                  "roles":{"type":"array","items":{"type":"string"}},
                  "specialized":{"type":"boolean","default":True},
                  "rounds":{"type":"integer","default":2},
                  "session_id":{"type":"string"}
                }, "required":["task"]}
            },
            {
                "name": "reflect",
                "description": "Critique and improve content using reflection loops.",
                "inputSchema": {"type":"object","properties":{"content":{"type":"string"},"criteria":{"type":"string"},"rounds":{"type":"integer","default":1},"session_id":{"type":"string"}}, "required":["content"]}
            },
            {
                "name": "tool_match",
                "description": "Suggest relevant MCP tools for a task via semantic keyword matching.",
                "inputSchema": {"type":"object","properties":{"task":{"type":"string"}}, "required":["task"]}
            },
            {
                "name": "router_diagnostics",
                "description": "Show recent router decisions and latencies.",
                "inputSchema": {"type":"object","properties":{"limit":{"type":"integer","default":20}}}
            },
            {
                "name": "router_test",
                "description": "Test router decision for a given task without executing.",
                "inputSchema": {"type":"object","properties":{"task":{"type":"string"},"role":{"type":"string"},"intent":{"type":"string"},"complexity":{"type":"string","enum":["low","medium","high","auto"],"default":"auto"}}, "required":["task"]}
            },
            {
                "name": "router_update_profile",
                "description": "Update backend profile (performance/capabilities)",
                "inputSchema": {"type":"object","properties":{"backend":{"type":"string","enum":["lmstudio","openai","anthropic"]},"updates":{"type":"object"}}, "required":["backend","updates"]}
            },
            {
                "name": "router_battery",
                "description": "Run a structured router test suite and return JSON report with metrics and gating.",
                "inputSchema": {"type":"object","properties":{
                    "cases":{"type":"array","items":{"type":"object"}},
                    "limit":{"type":"integer","default":3},
                    "thresholds":{"type":"object","properties":{
                        "min_anthropic":{"type":"number","default":0},
                        "max_lmstudio_latency_ms":{"type":"number","default":1000000000},
                        "min_high_conf":{"type":"number","default":0}
                    }}
                }}
            },
            {
                "name": "router_self_test",
                "description": "Run a quick router test battery and return a report.",
                "inputSchema": {"type":"object","properties":{}}
            },

            {
                "name": "memory_consolidate",
                "description": "Summarize recent memories of a category into semantic_memory.",
                "inputSchema": {"type":"object","properties":{"category":{"type":"string","default":"general"},"limit":{"type":"integer","default":50}}, "required":[]}
            },
            {
                "name": "memory_retrieve_semantic",
                "description": "Retrieve top semantic memories matching a query.",
                "inputSchema": {"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","default":5}}, "required":["query"]}
            },

            {"name": "import_graph", "description": "Build a simple Python import graph and report top nodes.", "inputSchema": {"type":"object","properties":{"directory":{"type":"string","default":"."},"limit":{"type":"integer","default":10}}}},
            {"name": "session_create", "description": "Create a collaboration session (V2).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"},"participants":{"type":"array","items":{"type":"string"}},"initial_task":{"type":"string"}}, "required":["initial_task"]}},

            {"name": "context_envelope_create", "description": "Create a V2 context envelope (ENHANCED_STORAGE=1).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"},"context_type":{"type":"string"},"content":{"type":"object"},"metadata":{"type":"object","default":{}},"parent_envelope_id":{"type":"string"},"ttl_seconds":{"type":"integer"}}, "required":["session_id","context_type","content"]}},
            {"name": "context_envelopes_list", "description": "List context envelopes for a session (V2).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"},"limit":{"type":"integer","default":50}}, "required":["session_id"]}},
            {"name": "artifact_create", "description": "Create a collaborative artifact (V2).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"},"artifact_type":{"type":"string"},"title":{"type":"string"},"content":{"type":"string"},"contributors":{"type":"array","items":{"type":"string"}},"tags":{"type":"array","items":{"type":"string"}},"parent_artifact_id":{"type":"string"}}, "required":["session_id","artifact_type","title","content","contributors"]}},
            {"name": "artifacts_list", "description": "List artifacts for a session (V2).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"},"limit":{"type":"integer","default":50}}, "required":["session_id"]}},
            {"name": "session_analytics", "description": "Return analytics for a collaboration session (V2).", "inputSchema": {"type":"object","properties":{"session_id":{"type":"string"}}, "required":["session_id"]}},

            {"name": "file_scaffold", "description": "Create a new module or test skeleton.", "inputSchema": {"type":"object","properties":{"path":{"type":"string"},"kind":{"type":"string","enum":["module","test"]},"description":{"type":"string"},"dry_run":{"type":"boolean","default":True}}, "required":["path","kind"]}}
        ]
    }
    # Allow enhanced module to merge/augment tools while preserving shape
    return _merged_tools(base)


def _sanitize_url(url: str) -> str:
    try:
        # Basic URL sanitation: strip whitespace and guard against data: or javascript:
        u = (url or "").strip()
        if u.startswith("data:") or u.startswith("javascript:"):
            return ""
        return u
    except Exception:
        return ""

# --- New Tools: web_search, backend_diagnostics, code_hotspots, import_graph, file_scaffold ---

def handle_web_search(arguments, server):
    query = (arguments.get("query") or "").strip()
    if not query:
        raise ValidationError("'query' is required")
    time_limit = int(arguments.get("time_limit", 60))
    max_depth = int(arguments.get("max_depth", 1))
    return handle_deep_research({"query": query, "time_limit": time_limit, "max_depth": max_depth}, server)


def handle_backend_diagnostics(arguments, server):
    roles = arguments.get("roles") or [
        "Planner", "Coder", "Reviewer", "Security Reviewer", "Performance Analyzer",
    ]
    task_desc = (arguments.get("task_desc") or "diagnostics").strip()

    # Backend presence
    available = {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "lmstudio": bool(os.getenv("LMSTUDIO_API_BASE") or os.getenv("OPENAI_API_BASE")),
    }
    lines = ["Backends available:"] + [f"- {k}: {v}" for k, v in available.items()]

    # Per-role routing
    lines.append("\nRouting decisions:")
    for r in roles:
        b = _decide_backend_for_role(r, task_desc)
        llm = _build_llm_for_backend(b)
        model = getattr(llm, "model", "n/a") if llm else "n/a"
        lines.append(f"- {r}: {b} ({model})")
    return "\n".join(lines)


def _walk_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in {".git", "venv", ".venv", "node_modules", "dist", "build"} for part in p.parts):
            continue
        yield p


def handle_code_hotspots(arguments, server):
    root = Path(arguments.get("directory") or ".").resolve()
    limit = int(arguments.get("limit", 10))
    stats = []
    for p in _walk_py_files(root):
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


def handle_import_graph(arguments, server):
    root = Path(arguments.get("directory") or ".").resolve()
    limit = int(arguments.get("limit", 10))
    edges = {}
    for p in _walk_py_files(root):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        mod = str(p.relative_to(root)).replace(os.sep, ".").rstrip(".py").rstrip(".")
        for m in re.findall(r"^from\s+([\w\.]+)\s+import\s+", txt, flags=re.MULTILINE):
            edges.setdefault(m, set()).add(mod)
        for m in re.findall(r"^import\s+([\w\.]+)", txt, flags=re.MULTILINE):
            edges.setdefault(m, set()).add(mod)
    # Rank by in-degree
    ranked = sorted(((k, len(v)) for k,v in edges.items()), key=lambda x: x[1], reverse=True)[:limit]
    header = "module,in_degree"
    rows = [header] + [f"{m},{d}" for m,d in ranked]
    return "\n".join(rows)


def handle_file_scaffold(arguments, server):
    path = (arguments.get("path") or "").strip()
    kind = (arguments.get("kind") or "").strip()
    desc = (arguments.get("description") or "").strip()
    dry = bool(arguments.get("dry_run", True))
    if not path or kind not in {"module", "test"}:
        raise ValidationError("path and kind ('module'|'test') are required")
    sp = _safe_path(path)
    template = """# {desc}
""" if kind == "module" else """import pytest


def test_placeholder():
    assert True
"""
    out = f"Would write {sp} ({len(template)} chars)"
    if dry:
        return out
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(template, encoding="utf-8")
    return out.replace("Would ", "")



def handle_deep_research(arguments, server):
    """Hybrid deep research using Firecrawl (stage 1) then CrewAI agents (stage 2).
    Returns a concise summary and stores full artifacts under research_{id}.
    """
    query = arguments.get("query", "").strip()
    if not query:
        raise ValidationError("'query' is required")
    max_depth = int(arguments.get("max_depth", 8))
    time_limit = int(arguments.get("time_limit", 300))

    research_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:12]
    artifacts = {"query": query, "stage1": {}, "stage2": {}}

    # Stage 1: Firecrawl Deep Research
    # Prefer Firecrawl MCP proxy if available (Augment environment), otherwise call Firecrawl HTTP API directly
    try:
        from functions import firecrawl_deep_research_firecrawl_mcp as firecrawl_deep
    except Exception:
        firecrawl_deep = None

    try:
        if firecrawl_deep is not None:
            # Use MCP tool proxy (provided by the hosting environment)
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
            # Fallback: direct Firecrawl API call via HTTP if API key is configured
            api_key = (os.getenv("FIRECRAWL_API_KEY", "").strip())
            base_url = (os.getenv("FIRECRAWL_BASE_URL", FIRECRAWL_BASE_URL_STATIC).rstrip("/"))
            if not api_key:
                stage1 = {
                    "final": None,
                    "raw": None,
                    "warning": "Firecrawl MCP unavailable and FIRECRAWL_API_KEY not set; skipping stage 1. Set FIRECRAWL_API_KEY or enable MCP."
                }
            else:
                import requests
                payload = {
                    "query": query,
                    "maxDepth": max(1, min(max_depth, 10)),
                    "timeLimit": max(30, min(time_limit, 300)),
                    "maxUrls": 40,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                # Try versioned endpoint first, then unversioned for backward compatibility
                urls_to_try = [f"{base_url}/v1/deep-research", f"{base_url}/deep-research"]
                resp = None
                for url in urls_to_try:
                    try:
                        r = requests.post(url, json=payload, headers=headers, timeout=(30, 240))
                        # Prefer first success; if 404, try next URL
                        if r.status_code == 404:
                            resp = r
                            continue
                        resp = r
                        break
                    except Exception as _:
                        # Try next URL on transport errors
                        continue
                if resp is not None and 200 <= resp.status_code < 300:
                    fc = resp.json()
                    stage1 = {
                        "final": (fc.get("data", {}) or {}).get("finalAnalysis") if isinstance(fc, dict) else None,
                        "raw": fc,
                    }
                else:
                    msg = f"no response" if resp is None else f"HTTP {resp.status_code}: {resp.text[:200]}"
                    stage1 = {
                        "final": None,
                        "raw": None,
                        "error": f"Firecrawl {msg}"
                    }
    except Exception as e:
        stage1 = {"final": None, "error": f"Firecrawl error: {str(e)}"}

    artifacts["stage1"] = stage1

    # Stage 2: CrewAI multi-agent synthesis (optional; best-effort)
    try:
        from crewai import Agent, Crew, Task
        # Try to hardwire CrewAI to LM Studio's OpenAI-compatible endpoint and model
        llm = None
        try:
            try:
                # CrewAI v0.50+ style
                from crewai import LLM  # type: ignore
            except Exception:
                # Older style
                from crewai.llm import LLM  # type: ignore
            # Configure LLM to use LM Studio with the requested model
            # Prefer LMSTUDIO_* variables if set; fall back to OPENAI_* for backward compat
            lmstudio_base = os.getenv("LMSTUDIO_API_BASE") or os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
            lmstudio_key = os.getenv("LMSTUDIO_API_KEY") or os.getenv("OPENAI_API_KEY", "sk-noauth")
            hardwired_model = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
            llm = LLM(model=hardwired_model, base_url=lmstudio_base, api_key=lmstudio_key, temperature=0.2)
        except Exception:
            llm = None  # If LLM config fails, proceed; fallback synthesis will cover errors

        # Create agents with focused roles (inject llm if available)
        agent_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            agent_kwargs["llm"] = llm

        analyst = Agent(
            role="Analyst",
            goal="Categorize and evaluate Firecrawl findings, extract key facts and gaps.",
            backstory="Senior research analyst skilled at distilling insights from diverse sources.",
            **agent_kwargs,
        )
        researcher = Agent(
            role="Researcher",
            goal="Identify missing angles and perform targeted follow-ups based on gaps.",
            backstory="Curious investigator who knows where to look for authoritative sources.",
            **agent_kwargs,
        )
        synthesizer = Agent(
            role="Synthesizer",
            goal="Produce concise, actionable recommendations tailored to the query.",
            backstory="Executive-level writer focusing on clarity and actionability.",
            **agent_kwargs,
        )
        # Prepare context from Stage 1
        fc_summary = stage1.get("final") or ""
        fc_raw = stage1.get("raw")
        # Size-limit raw content
        fc_raw_text = _compact_text(json.dumps(fc_raw, ensure_ascii=False) if isinstance(fc_raw, (dict, list)) else str(fc_raw))

        t1 = Task(description=f"Analyze initial findings for: {query}\n\nFindings:\n{fc_summary or fc_raw_text}", agent=analyst)
        t2 = Task(description=f"Identify missing angles and propose targeted follow-ups for: {query}", agent=researcher)
        t3 = Task(description=f"Synthesize concise, actionable recommendations for: {query}", agent=synthesizer)

        crew = Crew(agents=[analyst, researcher, synthesizer], tasks=[t1, t2, t3], verbose=False)
        crew_out = crew.kickoff()
        stage2 = {"report": str(crew_out)[:4000]}
    except Exception as e:
        # Fallback: synthesize using the local LLM if CrewAI isn't installed or fails
        try:
            fc_summary = stage1.get("final") or ""
            fc_raw = stage1.get("raw")
            fc_raw_text = _compact_text(json.dumps(fc_raw, ensure_ascii=False) if isinstance(fc_raw, (dict, list)) else str(fc_raw))
            synthesis_prompt = (
                "You are an expert research synthesizer. Based on the following findings, produce a concise, actionable summary with 3-6 bullets, then a short conclusion.\n\n"
                f"Query: {query}\n\nFindings:\n{fc_summary or fc_raw_text}\n\n"
                "Output only text."
            )
            # Use the centralized LLM helper with retry
            try:
                llm_out = asyncio.get_event_loop().run_until_complete(
                    server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    llm_out = loop.run_until_complete(
                        server.make_llm_request_with_retry(synthesis_prompt, temperature=0.2)
                    )
                finally:
                    loop.close()
            stage2 = {"report": _compact_text(llm_out)}
        except Exception as e2:
            stage2 = {"report": None, "error": f"CrewAI error: {str(e)}; fallback synthesis failed: {str(e2)}"}

    artifacts["stage2"] = stage2

    # Store artifacts
    try:
        server.storage.store_memory(key=f"research_{research_id}", value=json.dumps(artifacts)[:4000], category="research")
    except Exception:
        # Best effort; even if storage fails, return text
        pass

    # Compose concise return message (MCP text-only)
    stage1_note = stage1.get("final") or stage1.get("warning") or stage1.get("error") or "stage1-ok"
    stage2_note = stage2.get("report") or stage2.get("error") or "stage2-ok"
    summary = (
        f"Deep research {research_id} completed.\n\n"
        f"Stage 1 (Firecrawl): {stage1_note[:300]}\n\n"
        f"Stage 2 (Synthesis): {stage2_note[:600]}\n\n"
        "Use research_id to retrieve details later."
    )
    return summary




def handle_tool_call(message):
    """Handle tool execution requests with enhanced capabilities"""
    try:
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        server = get_server_singleton()

        # Dispatch table mapping tool names to (handler, needs_server)
        registry = {
            # Core
            "deep_research": (handle_deep_research, True),
            "get_research_details": (handle_get_research_details, True),
            "propose_research": (handle_propose_research, True),

            # Agentic teams
            "agent_team_plan_and_code": (handle_agent_team_plan_and_code, True),
            "agent_team_review_and_test": (handle_agent_team_review_and_test, True),
            "agent_team_refactor": (handle_agent_team_refactor, True),

            # Code understanding
            "analyze_code": (lambda args, srv: handle_llm_analysis_tool("analyze_code", args, srv), True),
            "explain_code": (lambda args, srv: handle_llm_analysis_tool("explain_code", args, srv), True),
            "suggest_improvements": (lambda args, srv: handle_llm_analysis_tool("suggest_improvements", args, srv), True),
            "generate_tests": (lambda args, srv: handle_llm_analysis_tool("generate_tests", args, srv), True),

            # Execution & Project
            "execute_code": (handle_code_execution, False),
            "run_tests": (handle_test_execution, False),
            "read_file_content": (handle_file_read, False),
            "write_file_content": (handle_file_write, False),
            "list_directory": (handle_directory_list, False),
            "search_files": (handle_file_search, False),

            # Memory & System
            "store_memory": (handle_memory_store, True),
            "retrieve_memory": (handle_memory_retrieve, True),
            "health_check": (handle_health_check, True),
            "get_version": (handle_get_version, True),

            # Web research
            "web_search": (handle_web_search, True),

            # Collaboration/Reflection/Memory tools
            "agent_collaborate": (handle_agent_collaborate, True),
            "reflect": (handle_reflect, True),
            "smart_plan_execute": (handle_smart_plan_execute, True),
            "tool_match": (handle_tool_match, True),
            "memory_consolidate": (handle_memory_consolidate, True),
            "memory_retrieve_semantic": (handle_memory_retrieve_semantic, True),

            # Router tools
            "router_diagnostics": (handle_router_diagnostics, True),
            "router_test": (handle_router_test, True),
            # TDD
            "tdd_flow": (handle_tdd_flow, False),
            "router_update_profile": (handle_router_update_profile, True),
            "router_battery": (handle_router_battery, True),
            # Smart router
            "smart_task": (handle_smart_task, True),
            "router_self_test": (handle_router_self_test, True),

            # Diagnostics & analysis
            "backend_diagnostics": (handle_backend_diagnostics, True),
            "code_hotspots": (handle_code_hotspots, True),
            "import_graph": (handle_import_graph, True),
            "file_scaffold": (handle_file_scaffold, True),

            # Optional debugging and thinking tools (available but not advertised in minimal surface)
            "sequential_thinking": (handle_sequential_thinking, True),
            "get_thinking_session": (handle_get_thinking_session, True),
            "summarize_thinking_session": (handle_summarize_thinking_session, True),
            "get_performance_stats": (handle_performance_stats, True),
            "get_error_patterns": (handle_error_patterns, True),
            "debug_analyze": (handle_debug_analysis, True),
            "trace_execution": (handle_execution_trace, True),
            # Audit & Workflow tools
            "audit_search": (handle_audit_search, True),
            "audit_verify_integrity": (handle_audit_verify_integrity, True),
            "audit_add_rule": (handle_audit_add_rule, True),
            "audit_compliance_report": (handle_audit_compliance_report, True),
            "audit_review_action": (handle_audit_review_action, True),
            "workflow_create": (handle_workflow_create, True),
            "workflow_add_node": (handle_workflow_add_node, True),
            "workflow_connect_nodes": (handle_workflow_connect_nodes, True),
            "workflow_explain": (handle_workflow_explain, True),
            "workflow_execute": (handle_workflow_execute, True),

            # Session management (V2)
            "session_create": (handle_session_create, True),

            # V2 context and artifact tools
            "context_envelope_create": (handle_context_envelope_create, True),
            "context_envelopes_list": (handle_context_envelopes_list, True),
            "artifact_create": (handle_artifact_create, True),
            "artifacts_list": (handle_artifacts_list, True),
            "session_analytics": (handle_session_analytics, True),
        }


        if tool_name not in registry:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        handler, needs_server = registry[tool_name]
        # Tool execution auditing + timing
        start_ts = time.time()
        session_id = arguments.get("session_id") or f"sess_{uuid4().hex[:8]}"
        agent_role = arguments.get("agent_role") or "system"
        error_details = None
        result_type = "success"
        try:
            monitored = server.monitor_performance(tool_name)
            if needs_server:
                result = monitored(handler)(arguments, server)
            else:
                result = monitored(handler)(arguments)
        except Exception as exc:
            error_details = str(exc)
            result_type = "error"
            # Log usage on error as well
            try:
                server.storage.log_tool_usage(
                    tool_name=tool_name,
                    agent_role=agent_role,
                    session_id=session_id,
                    parameters={k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in (arguments or {}).items()},
                    result_type=result_type,
                    execution_time_ms=int((time.time() - start_ts) * 1000),
                    success=False,
                    error_details=error_details,
                )
            except Exception:
                pass
            raise

        # After handler executes successfully, audit the tool call
        try:
            if _audit_logger is not None:
                _audit_logger.log(
                    action_type=ActionType.AGENT_DECISION,
                    level=AuditLevel.INFO,
                    user_id=None,
                    tenant_id=None,
                    details={"tool_name": tool_name, "success": True},
                    context={"args": arguments, "ts": time.time()},
                    compliance_tags=[tool_name, "tool_execution"],
                )
        except Exception:
            pass

        # Log tool usage success
        try:
            server.storage.log_tool_usage(
                tool_name=tool_name,
                agent_role=agent_role,
                session_id=session_id,
                parameters={k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in (arguments or {}).items()},
                result_type=result_type,
                execution_time_ms=int((time.time() - start_ts) * 1000),
                success=True,
                error_details=None,
            )
        except Exception:
            pass

        # Always return text content per MCP schema
        payload_text = _to_text_content(result)

        # Best-effort: log a compact conversation envelope for traceability
        try:
            env_key = f"envelope_{tool_name}_{int(time.time())}"
            env_val = json.dumps({
                "tool": tool_name,
                "args": arguments,
                "result_preview": payload_text[:512],
            }, ensure_ascii=False)
            server.storage.store_memory(key=env_key, value=env_val, category="envelope")
        except Exception:
            pass

        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "content": [
                    {"type": "text", "text": payload_text}
                ]
            }
        }
    except ValidationError as e:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32602,
                "message": f"Invalid params: {str(e)}"
            }
        }
    except Exception as e:
        logger.error(f"Error in tool call: {e}")
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32603,
                "message": f"Tool execution error: {str(e)}"
            }
        }

# Tool Handler Functions (Updated for Persistent Storage)

def handle_sequential_thinking(arguments, server):
    """Handle sequential thinking with persistent storage and sessions (P2)"""
    thought = arguments.get("thought", "")
    thought_number = arguments.get("thought_number", 1)
    total_thoughts = arguments.get("total_thoughts", 1)
    next_thought_needed = arguments.get("next_thought_needed", False)
    is_revision = arguments.get("is_revision", False)
    revises_thought = arguments.get("revises_thought")
    branch_id = arguments.get("branch_id")
    session_id = arguments.get("session_id") or f"thinking_{uuid4().hex[:12]}"
    # Optional context to aid LLM suggestion
    problem = arguments.get("problem")
    auto_next = bool(arguments.get("auto_next", False))



    # Store thinking step in persistent storage (always record user-provided step)
    server.storage.store_thinking_step(
        session_id=session_id,
        thought_number=thought_number,
        thought=thought,
        is_revision=is_revision,
        revises_thought=revises_thought,
        branch_id=branch_id
    )

    llm_suggestion = None
    # If caller requests auto_next and the session expects next thoughts, query LLM
    if next_thought_needed and auto_next:
        prompt = _build_sequential_thinking_prompt(
            session_id=session_id,
            server=server,
            current_thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            is_revision=is_revision,
            problem=problem,
        )
        try:
            # Use centralized LLM with retry; keep low temperature for focused suggestions
            try:
                llm_suggestion = asyncio.get_event_loop().run_until_complete(
                    server.make_llm_request_with_retry(prompt, temperature=0.2, retries=2, backoff=0.5)
                )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    llm_suggestion = loop.run_until_complete(
                        server.make_llm_request_with_retry(prompt, temperature=0.2, retries=2, backoff=0.5)
                    )
                finally:
                    loop.close()
            # Compact the suggestion and store it as an auto-generated thought step
            compacted = _compact_text(llm_suggestion)
            try:

                history = server.storage.get_thinking_session(session_id) or []
                next_idx = max((s.get("thought_number", 0) for s in history), default=0) + 1
            except Exception:
                next_idx = thought_number + 1
            server.storage.store_thinking_step(
                session_id=session_id,
                thought_number=next_idx,
                thought=compacted,
                is_revision=False,
                revises_thought=None,
                branch_id=branch_id,
            )
        except Exception as e:
            # Non-fatal; proceed without LLM suggestion
            llm_suggestion = f"Error generating suggestion: {str(e)}"

    # Build response
    response = {
        "session_id": session_id,
        "thought_number": thought_number,
        "total_thoughts": total_thoughts,
        "is_revision": is_revision,
        "next_thought_needed": next_thought_needed,
        "message": "More analysis needed..." if next_thought_needed else "Analysis complete",
    }
    if llm_suggestion is not None:
        response["llm_suggestion"] = _sanitize_llm_output(llm_suggestion)

    # On completion, store session summary
    if not next_thought_needed:
        session_thoughts = server.storage.get_thinking_session(session_id)
        summary = f"Sequential thinking session with {len(session_thoughts)} thoughts"
        server.storage.store_memory(
            key=f"thinking_session_{session_id}",
            value=summary,
            category="reasoning"
        )

    return response

def handle_llm_analysis_tool(tool_name, arguments, server):
    """Handle LLM-based analysis tools via centralized request with retry"""
    code = arguments.get("code", "")
    compact = bool(arguments.get("compact", False))

    code_only = bool(arguments.get("code_only", False))  # for generate_tests

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
        limit = 8 if compact else 12
        prompt = (
            f"List concrete improvements (<= {limit} bullets). Include complexity/perf/security notes where relevant.\n\n"
            f"Code:\n{code}\n"
        )
    elif tool_name == "generate_tests":
        framework = arguments.get("framework", "pytest")
        tests_target = 4 if compact else 6
        prompt = (
            f"Generate {framework} unit tests for the code. Output ONLY the test code without prose, "
            f"using functions named test_*. Aim for {tests_target}-{tests_target+2} focused tests.\n\n"
            f"Code:\n{code}\n"
        )
    else:
        prompt = ""

    # Call centralized LLM with retry (sync wrapper over async for current architecture)
    try:
        raw = asyncio.get_event_loop().run_until_complete(
            server.make_llm_request_with_retry(prompt, temperature=0.1, retries=2, backoff=0.5)
        )
    except RuntimeError:
        # If no running loop, create one
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            raw = loop.run_until_complete(
                server.make_llm_request_with_retry(prompt, temperature=0.1, retries=2, backoff=0.5)
            )
        finally:
            loop.close()
    except Exception as e:
        return f"Error: {str(e)}"

    if tool_name == "generate_tests" and code_only:
        return _extract_code_from_text(raw)
    return raw


def handle_get_research_details(arguments, server):
    """Retrieve stored deep research artifacts by research_id."""
    research_id = (arguments.get("research_id") or "").strip()
    if not research_id:
        raise ValidationError("'research_id' is required")
    key = f"research_{research_id}"
    try:
        rows = server.storage.retrieve_memory(key=key)
        if not rows:
            return f"No research found for id: {research_id}"
        row = rows[0]
        value = row.get("value") or ""
        # If value is JSON, pretty-print; else return as-is
        try:
            obj = json.loads(value)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            return _compact_text(pretty, max_chars=4000)
        except Exception:
            return _compact_text(value, max_chars=4000)
    except Exception as e:
        return f"Error retrieving research {research_id}: {str(e)}"

def handle_code_execution(arguments):
    """Handle safe code execution (disabled by default, robust cleanup)"""
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
        else:  # bash
            cmd = ["bash", temp_file]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )

        output = f"Exit Code: {result.returncode}\n\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n\n"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass

def handle_test_execution(arguments):
    """Handle test execution (disabled by default, allow-listed commands)"""
    if not _execution_enabled():
        return "Error: Test execution is disabled. Set EXECUTION_ENABLED=true to allow."

    test_command = arguments.get("test_command", "pytest")
    test_file = arguments.get("test_file", "")

    # Allow-list base executables
    allowed = {"pytest", "python", "python3", "pipenv", "poetry"}

    try:
        parts = test_command.split()
        if not parts or parts[0] not in allowed:
            return "Error: Test command not allowed."

        cmd = parts
        if test_file:
            cmd.append(test_file)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd()
        )

        output = f"Test Command: {' '.join(cmd)}\n"
        output += f"Exit Code: {result.returncode}\n\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n\n"
        return output
    except Exception as e:
        return f"Error running tests: {str(e)}"

def handle_file_read(arguments):
    """Handle file reading (restricted to ALLOWED_BASE_DIR)"""
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
    except ValidationError as e:
        raise
    except Exception as e:
        return f"Error reading file: {str(e)}"
def handle_get_thinking_session(arguments, server):
    session_id = arguments.get("session_id")
    if not session_id:
        raise ValidationError("session_id is required")
    steps = server.storage.get_thinking_session(session_id)
    return {"session_id": session_id, "steps": steps}


def handle_summarize_thinking_session(arguments, server):
    session_id = arguments.get("session_id")
    if not session_id:
        raise ValidationError("session_id is required")
    steps = server.storage.get_thinking_session(session_id) or []
    summary = f"Session {session_id}: {len(steps)} steps"
    return {"session_id": session_id, "summary": summary}


def handle_file_write(arguments):
    """Handle file writing (restricted to ALLOWED_BASE_DIR)"""
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
    except ValidationError as e:
        raise
    except Exception as e:
        return f"Error writing file: {str(e)}"

def handle_directory_list(arguments):
    """Handle directory listing (restricted to ALLOWED_BASE_DIR)"""
    directory_path = arguments.get("directory_path", ".")
    include_hidden = arguments.get("include_hidden", False)

    try:
        rp = _safe_directory(directory_path)
        items = []
        for item in os.listdir(rp):
            if not include_hidden and item.startswith('.'):
                continue
            item_path = rp / item
            item_type = "DIR" if item_path.is_dir() else "FILE"
            items.append(f"{item_type:4} {item}")
        return f"Directory: {rp}\n\n" + "\n".join(sorted(items))
    except ValidationError as e:
        raise
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def handle_file_search(arguments):
    """Handle file searching (restricted to ALLOWED_BASE_DIR)"""
    pattern = arguments.get("pattern", "")
    file_pattern = arguments.get("file_pattern", "*")
    directory = arguments.get("directory", ".")

    import glob
    import re

# --- Audit MCP Tool Handlers (Rec 6) ---

def _require_audit():
    if _audit_logger is None:
        raise ValidationError("audit subsystem unavailable")
    return _audit_logger


def handle_audit_search(arguments, server):
    audit = _require_audit()
    filters = {k: arguments.get(k) for k in ("action_type", "level", "tool_name") if arguments.get(k)}
    limit = int(arguments.get("limit", 100))
    return audit.search(filters, limit=limit)


def handle_audit_verify_integrity(arguments, server):
    audit = _require_audit()
    limit = int(arguments.get("limit", 200))
    return audit.verify_chain(limit=limit)


def handle_audit_add_rule(arguments, server):
    audit = _require_audit()
    rid = (arguments.get("rule_id") or "").strip()
    name = (arguments.get("name") or "").strip()
    pattern = (arguments.get("pattern") or "").strip()
    if not rid or not name or not pattern:
        raise ValidationError("rule_id, name, and pattern are required")
    severity_str = (arguments.get("severity") or "info").lower()
    # Normalize severity to AuditLevel
    try:
        sev = AuditLevel(severity_str)
    except Exception:
        sev = AuditLevel.INFO
    action_required = (arguments.get("action_required") or "log").lower()
    retention_years = int(arguments.get("retention_years", 7))
    rule = ComplianceRule(
        rule_id=rid,
        name=name,
        description=arguments.get("description", ""),
        pattern=pattern,
        severity=sev,
        action_required=action_required,
        retention_years=retention_years,
    )
    audit.add_rule(rule)
    return {"ok": True}


def handle_audit_compliance_report(arguments, server):
    audit = _require_audit()
    return audit.generate_compliance_report(tenant_id=arguments.get("tenant_id"))


def handle_audit_review_action(arguments, server):
    audit = _require_audit()
    action_desc = (arguments.get("action_desc") or "").strip()
    if not action_desc:
        raise ValidationError("action_desc is required")
    context = arguments.get("context") or {}
    domain = (arguments.get("domain") or "general").strip()
    reviewer = AttorneyStyleReviewer(audit, server)
    try:
        return asyncio.get_event_loop().run_until_complete(reviewer.review(action_desc, context, domain))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(reviewer.review(action_desc, context, domain))
        finally:
            loop.close()


# --- Workflow MCP Tool Handlers (Rec 7) ---

def _require_workflow():
    if _workflow is None:
        raise ValidationError("workflow subsystem unavailable")
    return _workflow


def handle_workflow_create(arguments, server):
    wf = _require_workflow()
    name = (arguments.get("name") or "").strip()
    if not name:
        raise ValidationError("name is required")
    description = (arguments.get("description") or "").strip()
    wid = wf.create_workflow(name=name, description=description)
    return {"workflow_id": wid}


def handle_workflow_add_node(arguments, server):
    wf = _require_workflow()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    node_type = (arguments.get("node_type") or "").strip()
    name = (arguments.get("name") or "").strip()
    config = arguments.get("config") or {}
    if not (workflow_id and node_type and name):
        raise ValidationError("workflow_id, node_type, and name are required")
    nid = wf.add_node(workflow_id, node_type, name, config)
    return {"node_id": nid}


def handle_workflow_connect_nodes(arguments, server):
    wf = _require_workflow()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    src = (arguments.get("source_node_id") or "").strip()
    tgt = (arguments.get("target_node_id") or "").strip()
    if not (workflow_id and src and tgt):
        raise ValidationError("workflow_id, source_node_id, target_node_id are required")
    ok = wf.connect_nodes(workflow_id, src, tgt)
    return {"ok": bool(ok)}


def handle_workflow_explain(arguments, server):
    wf = _require_workflow()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    if not workflow_id:
        raise ValidationError("workflow_id is required")
    try:
        text = asyncio.get_event_loop().run_until_complete(wf.explain_workflow(workflow_id))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            text = loop.run_until_complete(wf.explain_workflow(workflow_id))
        finally:
            loop.close()
    return text

# --- Collaboration, Reflection, Tool Match, Memory (follow-up recs) ---

def _run_llm(server, prompt: str, temperature: float = 0.2, intent: str | None = None, role: str | None = None) -> str:
    try:
        return asyncio.get_event_loop().run_until_complete(
            server.route_chat(prompt, temperature=temperature, intent=intent, role=role)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(server.route_chat(prompt, temperature=temperature, intent=intent, role=role))
        finally:
            loop.close()
    except Exception as e:
        return f"LLM error: {e}"


def handle_agent_collaborate(arguments, server):
    task = (arguments.get("task") or "").strip()
    if not task:
        raise ValidationError("'task' is required")
    roles = arguments.get("roles") or ["Researcher", "Developer", "Reviewer", "Security Reviewer"]
    rounds = int(arguments.get("rounds", 2))
    history: list[dict] = []
    for r in range(1, rounds + 1):
        round_notes = []
        for role in roles:
            context = "\n\n".join([f"{h['role']}: {h['note']}" for h in history][-6:])
            prompt = (
                f"Role: {role}\nRound: {r}/{rounds}\nTask: {task}\n"
                f"Recent context (may be partial):\n{context}\n\n"
                "Contribute succinct bullet points (<=6) with concrete, technical steps and call out any risks or dependencies."
            )
            note = _run_llm(server, prompt, intent="collaboration", role=role)
            note = _compact_text(note, 900)
            history.append({"role": role, "round": r, "note": note})
            round_notes.append({"role": role, "note": note})
    # Synthesize
    transcript = "\n\n".join([f"[{h['role']} r{h['round']}]\n{h['note']}" for h in history])
    synth_prompt = (
        "Synthesize the multi-agent discussion into a clear, actionable plan.\n"
        "Respond in JSON with keys: goals (list), plan (list), risks (list), open_questions (list).\n\n"
        f"Transcript:\n{_compact_text(transcript, 3500)}\n"
    )
    plan_txt = _run_llm(server, synth_prompt, temperature=0.1, intent="synthesis", role="Planner")
    # V2: use provided session or create one, then persist artifact
    session_id = (arguments.get("session_id") or f"sess_{int(time.time())}")
    try:
        server.storage.create_collaboration_session(
            session_id=session_id,
            session_type="agent_collaboration",
            participants=roles,
            initial_task=task,
            complexity_analysis={"rounds": rounds}
        )
    except Exception:
        pass
    try:
        server.storage.create_artifact(
            session_id=session_id,
            artifact_type="plan",
            title=f"Synthesis for: {task[:60]}",
            content=plan_txt,
            contributors=roles,
            tags=["collaboration", "synthesis"]
        )
    except Exception:
        pass
    return {"rounds": rounds, "roles": roles, "synthesis": plan_txt, "session_id": session_id}


def handle_reflect(arguments, server):
    content = (arguments.get("content") or "").strip()
    if not content:
        raise ValidationError("'content' is required")
    criteria = (arguments.get("criteria") or "Quality, correctness, clarity").strip()
    rounds = int(arguments.get("rounds", 1))
    current = content
    last_critique = None
    combined_critiques = []
    for i in range(rounds):
        critique_prompt = (
            f"Critique the following content against these criteria: {criteria}.\n"
            "List concrete issues as bullets and suggest precise fixes.\n\nCONTENT:\n" + _compact_text(current, 2500)
        )


        last_critique = _run_llm(server, critique_prompt, temperature=0.2)
        combined_critiques.append(last_critique)
        improve_prompt = (
            f"Improve the content by applying the critique (keep original intent).\nCriteria: {criteria}.\n"
            "Return only the improved content, no preface.\n\nCRITIQUE:\n" + _compact_text(last_critique, 1200) + "\n\nCONTENT:\n" + _compact_text(current, 2500)
        )
        current = _run_llm(server, improve_prompt, temperature=0.2)
    # V2: persist critique and improved content as an artifact (no-op on legacy)
    try:
        sid = (arguments.get("session_id") or f"sess_reflect_{int(time.time())}")
        server.storage.create_collaboration_session(sid, 'reflection', ['reflector'], f"Reflection: {current[:40]}")
        server.storage.create_artifact(sid, 'analysis', 'Reflection Critique', "\n\n".join(combined_critiques), ['reflector'], tags=['reflection','critique'])
        server.storage.create_artifact(sid, 'analysis', 'Improved Content', current, ['reflector'], tags=['reflection','improved'])
    except Exception:
        sid = arguments.get("session_id") or None
    return {"improved": current, "last_critique": last_critique, "session_id": sid}


def handle_tool_match(arguments, server):
    task = (arguments.get("task") or "").lower()
    if not task:
        raise ValidationError("'task' is required")
    tools = get_all_tools()["tools"]
    def score(t):
        text = (t.get("name", "") + " " + t.get("description", "")).lower()
        # simple token overlap
        st = set(re.findall(r"[a-z0-9_]+", text))
        sq = set(re.findall(r"[a-z0-9_]+", task))
        return len(st & sq)
    ranked = sorted(tools, key=score, reverse=True)
    top = [{"name": t["name"], "description": t.get("description", "")} for t in ranked[:5]]
    return {"matches": top}


def handle_memory_consolidate(arguments, server):
    category = (arguments.get("category") or "general").strip()
    limit = int(arguments.get("limit", 50))
    rows = server.storage.retrieve_memory(category=category, limit=limit) or []
    texts = []
    for r in rows:
        v = r.get("value")
        if isinstance(v, str):
            texts.append(v)
        else:
            try:
                texts.append(json.dumps(v, ensure_ascii=False))
            except Exception:
                continue
    blob = _compact_text("\n\n".join(texts), 3500)
    prompt = (
        f"Summarize key facts and insights from {len(texts)} short notes into a compact knowledge memo.\n"
        "Structure as bullets by theme, include actionable takeaways. Keep under 300 words.\n\n" + blob
    )
    summary = _run_llm(server, prompt, temperature=0.2)
    key = f"semantic_summary_{category}_{int(time.time())}"
    server.storage.store_memory(key=key, value=json.dumps({"summary": summary, "source_count": len(texts)}), category="semantic_memory")
    return {"key": key, "summary": summary}

# --- V2 MCP wrappers: context and artifacts ---
def handle_session_create(arguments, server):
    session_id = (arguments.get('session_id') or f"sess_{uuid4().hex[:8]}")
    participants = arguments.get('participants') or []
    task = (arguments.get('initial_task') or '').strip()
    if not task:
        raise ValidationError('initial_task is required')
    try:
        server.storage.create_collaboration_session(session_id=session_id, session_type='manual', participants=participants, initial_task=task)
    except Exception:
        server.storage.store_memory(key=f'session_{session_id}', value=json.dumps({'participants':participants,'task':task}), category='session')
    return {'session_id': session_id}


def handle_context_envelope_create(arguments, server):
    sid = (arguments.get('session_id') or '').strip()
    ctype = (arguments.get('context_type') or '').strip()
    content = arguments.get('content') or {}
    metadata = arguments.get('metadata') or {}
    parent = arguments.get('parent_envelope_id')
    ttl = arguments.get('ttl_seconds')
    if not (sid and ctype and isinstance(content, dict)):
        raise ValidationError('session_id, context_type, content required')
    try:
        eid = server.storage.create_context_envelope(session_id=sid, context_type=ctype, content=content, metadata=metadata, parent_envelope_id=parent, ttl_seconds=ttl)
    except AttributeError:
        # Legacy storage: emulate by storing memory and returning a pseudo id
        eid = f"env_legacy_{int(time.time())}"
        server.storage.store_memory(key=eid, value=json.dumps({'session_id':sid,'context_type':ctype,'content':content,'metadata':metadata}), category='context_envelope')
    return {'envelope_id': eid}


def handle_context_envelopes_list(arguments, server):
    sid = (arguments.get('session_id') or '').strip()
    limit = int(arguments.get('limit', 50))
    if not sid:
        raise ValidationError('session_id required')
    try:
        items = server.storage.list_context_envelopes(session_id=sid, limit=limit)
    except AttributeError:
        items = []
    return {'envelopes': items}


def handle_artifact_create(arguments, server):
    sid = (arguments.get('session_id') or '').strip()
    atype = (arguments.get('artifact_type') or '').strip()
    title = (arguments.get('title') or '').strip()
    content = (arguments.get('content') or '')
    contributors = arguments.get('contributors') or []
    tags = arguments.get('tags') or []
    parent = arguments.get('parent_artifact_id')
    if not (sid and atype and title and contributors is not None):
        raise ValidationError('session_id, artifact_type, title, content, contributors required')
    try:
        aid = server.storage.create_artifact(session_id=sid, artifact_type=atype, title=title, content=content, contributors=contributors, tags=tags, parent_artifact_id=parent)
    except AttributeError:
        aid = f"art_legacy_{int(time.time())}"
        server.storage.store_memory(key=aid, value=json.dumps({'session_id':sid,'artifact_type':atype,'title':title,'content':content,'contributors':contributors,'tags':tags}), category='artifact')
    return {'artifact_id': aid}

# Router self-test tool
async def _router_self_test(server) -> str:
    cases = [
        {"task": "Draft architecture and threat model for a microservices system", "intent": "analysis", "role": "Reviewer"},
        {"task": "Refactor a Python module to improve readability and add tests", "intent": "implementation", "role": "Developer"},
        {"task": "Research the best practices for MCP server routing", "intent": "research", "role": "Researcher"},
    ]
    lines = []
    for c in cases:
        try:
            dec = await server.route_chat(f"Router decision only for: {c['task']}", intent=c['intent'], role=c['role'])
            lines.append(f"Case: {c['role']}/{c['intent']} -> decided (see router_diagnostics for details)")
        except Exception as e:
            lines.append(f"Case: {c['role']}/{c['intent']} -> error {e}")
    return "\n".join(lines)

def handle_router_self_test(arguments, server):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop(); asyncio.set_event_loop(new_loop)
            try:
                out = new_loop.run_until_complete(_router_self_test(server))
            finally:
                new_loop.close(); asyncio.set_event_loop(loop)
            return out
        else:
            return loop.run_until_complete(_router_self_test(server))
    except Exception as e:
        return f"Router self-test failed: {e}"

# Router battery tool (structured)

def handle_router_battery(arguments, server):
    cases = arguments.get('cases') or [
        {"task": "Architect scalable pipeline", "intent": "analysis", "role": "Reviewer"},
        {"task": "Refactor Python module", "intent": "implementation", "role": "Developer"},
        {"task": "Research MCP routing best practices", "intent": "research", "role": "Researcher"},
    ]
    limit = int(arguments.get('limit', 3))
    report = []
    async def decide(task, intent, role):
        # Invoke route_chat but we return only the backend/model via diagnostics side effect
        try:
            text = await server.route_chat(f"Router decision only for: {task}", intent=intent, role=role)
            return True, text[:40]
        except Exception as e:
            return False, str(e)
    for c in cases[:limit]:
        ok = True; note = ''
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                new_loop = asyncio.new_event_loop(); asyncio.set_event_loop(new_loop)
                try:
                    ok, note = new_loop.run_until_complete(decide(c.get('task',''), c.get('intent'), c.get('role')))
                finally:
                    new_loop.close(); asyncio.set_event_loop(loop)
            else:
                ok, note = loop.run_until_complete(decide(c.get('task',''), c.get('intent'), c.get('role')))
        except Exception as e:
            ok, note = False, str(e)
        report.append({"case": c, "ok": ok, "note": note})
    # add latest diagnostics
    try:
        diags = handle_router_diagnostics({"limit": 10}, server)
    except Exception:
        diags = {}
    # Enhanced metrics
    metrics = {"backend_counts": {}, "model_counts": {}, "avg_latency_ms": {}, "confidence": {"low":0,"med":0,"high":0}}
    # Extract decisions from diagnostics
    try:
        if isinstance(diags, dict) and 'analytics' in diags and isinstance(diags['analytics'], dict):
            items = diags['analytics'].get('decisions') or diags['analytics'].get('history') or []
        elif isinstance(diags, dict) and 'decisions' in diags:
            items = diags['decisions']
        else:
            items = []
    except Exception:
        items = []
    lat_sum = {}
    lat_cnt = {}
    for d in items:
        try:
            b = (d.get('backend') or '').lower(); m = d.get('model') or 'auto'
            metrics['backend_counts'][b] = metrics['backend_counts'].get(b,0)+1
            metrics['model_counts'][m] = metrics['model_counts'].get(m,0)+1
            lat = d.get('latency_ms');
            if isinstance(lat, (int,float)):
                lat_sum[b] = lat_sum.get(b,0)+float(lat); lat_cnt[b] = lat_cnt.get(b,0)+1
            conf = d.get('confidence');
            if isinstance(conf, (int,float)):
                if conf < 0.34: metrics['confidence']['low'] += 1
                elif conf < 0.67: metrics['confidence']['med'] += 1
                else: metrics['confidence']['high'] += 1
        except Exception:
            continue
    for b in lat_sum:
        metrics['avg_latency_ms'][b] = round(lat_sum[b]/max(1,lat_cnt[b]), 2)

    # CI gating thresholds
    thr = arguments.get('thresholds') or {}
    res_ok = True
    reasons = []
    # Example gates: require some non-zero Anthropic/OpenAI usage, and cap LM Studio avg latency
    min_anthropic = float(thr.get('min_anthropic', 0))
    if metrics['backend_counts'].get('anthropic',0) < min_anthropic:
        res_ok = False; reasons.append(f"anthropic_count<{min_anthropic}")
    max_lmstudio_latency = float(thr.get('max_lmstudio_latency_ms', 1e9))
    if metrics['avg_latency_ms'].get('lmstudio',0) > max_lmstudio_latency:
        res_ok = False; reasons.append("lmstudio_latency")
    min_high_conf = float(thr.get('min_high_conf', 0))
    total_conf = sum(metrics['confidence'].values()) or 1
    if (metrics['confidence']['high']/total_conf) < min_high_conf:
        res_ok = False; reasons.append("low_high_conf_fraction")

    return json.dumps({"report": report, "diagnostics": diags, "metrics": metrics, "passed": res_ok, "reasons": reasons})

# Smart function router: maps a natural instruction to a specific tool call

SMART_RULES = [
    {"match": ["plan", "code", "implement"], "tool": "agent_team_plan_and_code"},
    {"match": ["review", "test", "pytest"], "tool": "agent_team_review_and_test"},
    {"match": ["research", "investigate", "sources"], "tool": "deep_research"},
    {"match": ["refactor", "cleanup", "modular"], "tool": "agent_team_refactor"},
]


# --- Schema helpers for smart routing ---

def _get_tool_schemas():
    tools = get_all_tools().get("tools", [])
    schemas = {}
    for t in tools:
        name = t.get("name")
        schema = t.get("inputSchema") or {}
        if name:
            schemas[name] = schema
    return schemas


def _coerce_value_to_type(value, typ):
    try:
        if typ == "string":
            return str(value)
        if typ == "integer":
            if isinstance(value, bool):
                return int(value)
            return int(str(value).strip())
        if typ == "boolean":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            return s in {"1", "true", "yes", "on"}
        if typ == "array":
            if isinstance(value, list):
                return value
            if value is None or value == "":
                return []
            return [v.strip() for v in str(value).split(",") if v.strip()]
        if typ == "object":
            if isinstance(value, dict):
                return value
            try:
                return json.loads(value)
            except Exception:
                return {}
    except Exception:
        return value
    return value


def _autocorrect_args(args, schema, context, instruction, prevent_fill_keys: set | None = None):
    props = (schema or {}).get("properties", {})
    required = (schema or {}).get("required", [])
    corrected = dict(args or {})
    corrections = {}
    block = set(prevent_fill_keys or [])

    # Coerce provided values to expected types
    for key, meta in props.items():
        if key in corrected and isinstance(meta, dict) and "type" in meta:
            before = corrected[key]
            after = _coerce_value_to_type(corrected[key], meta.get("type"))
            if before is not after:
                corrections[key] = {"from": before, "to": after}
                corrected[key] = after

    # Fill required fields
    for key in required:
        if key not in corrected or corrected.get(key) in (None, ""):
            meta = props.get(key, {})
            if "default" in meta:
                corrected[key] = meta["default"]
                corrections[key] = {"filled_default": True}
            elif meta.get("type") == "string" and context:
                corrected[key] = context
                corrections[key] = {"filled_from": "context"}
            elif meta.get("type") == "string" and instruction:
                corrected[key] = instruction
                corrections[key] = {"filled_from": "instruction"}
            elif meta.get("type") == "integer":
                corrected[key] = 1
                corrections[key] = {"filled_default_int": True}
            elif meta.get("type") == "boolean":
                corrected[key] = False
                corrections[key] = {"filled_default_bool": True}
            elif meta.get("type") == "array":
                corrected[key] = []
                corrections[key] = {"filled_default_array": True}
            elif meta.get("type") == "object":
                corrected[key] = {}
                corrections[key] = {"filled_default_object": True}


    if 'query' in props and 'query' not in corrected and instruction:
        corrected['query'] = instruction
        corrections['query'] = {"filled_from": "instruction"}

    return corrected, corrections




def _validate_args_against_schema(args, schema):
    props = (schema or {}).get("properties", {})
    required = (schema or {}).get("required", [])
    errors = []
    for key in required:
        if key not in args or args.get(key) in (None, ""):
            errors.append(f"Missing required: {key}")
    for key, meta in props.items():
        if key in args and isinstance(meta, dict) and "type" in meta:
            t = meta.get("type")
            val = args[key]
            if t == "string" and not isinstance(val, str):
                errors.append(f"{key} expected string")
            if t == "integer" and not isinstance(val, int):
                errors.append(f"{key} expected integer")
            if t == "boolean" and not isinstance(val, bool):
                errors.append(f"{key} expected boolean")
            if t == "array" and not isinstance(val, list):
                errors.append(f"{key} expected array")
            if t == "object" and not isinstance(val, dict):
                errors.append(f"{key} expected object")
            enum = meta.get("enum")
            if enum is not None and val not in enum:
                errors.append(f"{key} must be one of {enum}")
    return (len(errors) == 0), errors


def _smart_match_rule(instruction: str) -> str | None:
    text = (instruction or "").lower()
    for rule in SMART_RULES:
        if any(k in text for k in rule.get("match", [])):
            return rule.get("tool")
    return None


def handle_smart_task(arguments, server):
    instruction = (arguments.get('instruction') or '').strip()
    context = (arguments.get('context') or '').strip()
    dry_run = bool(arguments.get('dry_run', False))
    max_refines = int(arguments.get('max_refines', os.getenv('SMART_TASK_MAX_REFINES', '2')))
    if not instruction:
        raise ValidationError('instruction is required')

    # 1) Static rule match first
    tool = _smart_match_rule(instruction)

    # 2) If ambiguous, ask the router via LM Studio to select tool + args + confidence
    schemas = _get_tool_schemas()
    proposed_args = {}
    confidence = 0.0
    if not tool:
        schema_hint = "Respond ONLY with JSON: {\"tool\": <tool_name>, \"arguments\": {...}, \"confidence\": 0..1}"
        prompt = (
            f"You are a function router. Choose the best tool for the user's instruction and propose arguments.\n{schema_hint}\n"
            f"Instruction: {instruction}\nContext: {context}\n"
            f"Known tools: {', '.join(schemas.keys())}"
        )
        try:
            raw = asyncio.get_event_loop().run_until_complete(server.route_chat(prompt, intent='routing', role='Router'))
        except RuntimeError:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            raw = loop.run_until_complete(server.route_chat(prompt, intent='routing', role='Router')); loop.close()
        try:
            data = json.loads((raw or '').strip())
            if isinstance(data, dict):
                tool = data.get('tool') or tool
                proposed_args = data.get('arguments') or {}
                try:
                    confidence = float(data.get('confidence') or 0.0)
                except Exception:
                    confidence = 0.0
        except Exception:
            pass

    # 3) If still unknown, return options
    if not tool:
        return json.dumps({"selected": None, "options": [r['tool'] for r in SMART_RULES], "note": "ambiguous"})

    # 4) Schema-aware argument inference & validation + refinement loop
    target_schema = schemas.get(tool) or {}
    args_current = dict(proposed_args) if isinstance(proposed_args, dict) else {}
    refine_attempts = []
    for i in range(max(1, max_refines+1)):
        corrected, corrections = _autocorrect_args(args_current, target_schema, context, instruction)
        ok, errs = _validate_args_against_schema(corrected, target_schema)
        attempt_meta = {"iteration": i, "arguments": corrected, "corrections": corrections, "valid": ok, "errors": errs}
        refine_attempts.append(attempt_meta)
        if ok or max_refines == 0:
            break
        # ask router to refine arguments
        if server is None:
            break
        try:
            err_text = "\n".join(errs)
            minimal_schema = {
                "required": (target_schema or {}).get("required", []),
                "properties": {k: {"type": v.get("type"), **({"enum": v.get("enum")} if v.get("enum") else {})} for k, v in (target_schema or {}).get("properties", {}).items()}
            }
            refine_prompt = (
                "Your proposed arguments did not validate against the tool schema.\n"
                "Return ONLY JSON for the corrected arguments object, no prose.\n"
                f"Tool: {tool}\n"
                f"Schema: {json.dumps(minimal_schema)}\n"
                f"Current arguments: {json.dumps(corrected)}\n"
                f"Validation errors: {err_text}\n"
            )
            try:
                raw2 = asyncio.get_event_loop().run_until_complete(server.route_chat(refine_prompt, intent='routing', role='Router'))
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                raw2 = loop.run_until_complete(server.route_chat(refine_prompt, intent='routing', role='Router')); loop.close()
            try:
                new_args = json.loads((raw2 or '').strip())
                if isinstance(new_args, dict):
                    args_current = new_args
                else:
                    break
            except Exception:
                break
        except Exception:
            break

    final_corrected, corrections = _autocorrect_args(args_current, target_schema, context, instruction)
    ok, errs = _validate_args_against_schema(final_corrected, target_schema)

    summary = {"selected": tool, "confidence": confidence, "valid": ok, "errors": errs, "refine_attempts": refine_attempts}
    if dry_run:
        return json.dumps({**summary, "invoked": False, "arguments": final_corrected})

    # 5) Dispatch via existing registry
    message = {"params": {"name": tool, "arguments": final_corrected}}
    return handle_tool_call(message)


# --- Multi-tool planning and execution ---

def handle_smart_plan_execute(arguments, server):
    instruction = (arguments.get('instruction') or '').strip()
    context = (arguments.get('context') or '').strip()
    dry_run = bool(arguments.get('dry_run', False))
    max_steps = int(arguments.get('max_steps', 4))
    if not instruction:
        raise ValidationError('instruction is required')

    # Step 1: ask Router role for a plan [{tool, arguments, note}]
    schemas = _get_tool_schemas()
    plan_prompt = (
        "Return ONLY JSON: {\"steps\": [{\"tool\": <name>, \"arguments\": {...}, \"note\": <string>}]}.\n"
        f"Instruction: {instruction}\nContext: {context}\nKnown tools: {', '.join(schemas.keys())}\n"
        f"Max steps: {max_steps}. Ensure arguments satisfy required fields if possible."
    )
    try:
        raw = asyncio.get_event_loop().run_until_complete(server.route_chat(plan_prompt, intent='routing', role='Router'))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        raw = loop.run_until_complete(server.route_chat(plan_prompt, intent='routing', role='Router')); loop.close()
    steps = []
    try:
        data = json.loads((raw or '').strip())
        steps = data.get('steps') or []
    except Exception:
        steps = []

    # Step 2: validate and auto-correct step args
    state = {"artifacts": [], "logs": []}
    validated = []
    for i, s in enumerate(steps[:max_steps]):
        tool = s.get('tool'); args = s.get('arguments') or {}
        sch = schemas.get(tool) or {}
        corrected, corr = _autocorrect_args(args, sch, context, instruction)
        ok, errs = _validate_args_against_schema(corrected, sch)
        validated.append({"tool": tool, "arguments": corrected, "valid": ok, "errors": errs, "corrections": corr, "note": s.get('note')})

    if dry_run:
        return json.dumps({"plan": validated})

    # Step 3: execute steps sequentially with basic rollback/error handling
    results = []
    applied_writes = []
    for i, st in enumerate(validated):
        tool = st.get('tool'); args = st.get('arguments')
        if not tool:
            state['logs'].append({"step": i, "error": "missing tool"}); break
        message = {"params": {"name": tool, "arguments": args}}
        res = handle_tool_call(message)
        results.append({"step": i, "tool": tool, "result": res})
        # Track write operations from known tools to allow rollback
        if tool in {"write_file_content", "file_scaffold"}:
            try:
                # Best-effort: parse result preview to guess path
                txt = _to_text_content(res)
                if "Would write" in txt:
                    # dry-run; no rollback needed
                    pass
                else:
                    applied_writes.append(txt)
            except Exception:
                pass
        # Save intermediate artifact
        try:
            preview = _to_text_content(res)[:512]
            state['artifacts'].append({"step": i, "tool": tool, "preview": preview})
        except Exception:
            pass

    return json.dumps({"plan": validated, "results": results, "state": state})


    # 4) Schema-aware argument inference & validation
    target_schema = schemas.get(tool) or {}
    args0 = dict(proposed_args) if isinstance(proposed_args, dict) else {}
    corrected, corrections = _autocorrect_args(args0, target_schema, context, instruction)
    ok, errs = _validate_args_against_schema(corrected, target_schema)

    summary = {"selected": tool, "confidence": confidence, "corrections": corrections, "valid": ok, "errors": errs}
    if dry_run:
        return json.dumps({**summary, "invoked": False, "arguments": corrected})

    # 5) Dispatch via existing registry
    message = {"params": {"name": tool, "arguments": corrected}}
    return handle_tool_call(message)


def handle_artifacts_list(arguments, server):
    sid = (arguments.get('session_id') or '').strip()
    limit = int(arguments.get('limit', 50))
    if not sid:
        raise ValidationError('session_id required')
    try:
        items = server.storage.list_artifacts(session_id=sid, limit=limit)
    except AttributeError:
        items = []
    return {'artifacts': items}


def handle_session_analytics(arguments, server):
    # Placeholder to satisfy interface; return minimal data if storage lacks analytics
    sid = (arguments.get('session_id') or '').strip()
    limit = int(arguments.get('limit', 50))
    if not sid:
        raise ValidationError('session_id required')
    try:
        # If storage provides analytics, return them; otherwise return recent artifacts as a proxy
        if hasattr(server.storage, 'get_session_analytics'):
            return server.storage.get_session_analytics(session_id=sid, limit=limit)
        if hasattr(server.storage, 'list_artifacts'):
            arts = server.storage.list_artifacts(session_id=sid, limit=limit)
            return {'artifacts': arts}
    except Exception:
        pass
    return {'message': 'analytics not available'}

# TDD orchestration tool: generate tests -> run -> fix loop

def handle_tdd_flow(arguments, server):
    target_path = (arguments.get('target_path') or '').strip()
    goal = (arguments.get('goal') or '').strip()
    max_loops = int(arguments.get('max_loops', 2))
    apply_fixes = bool(arguments.get('apply_fixes', False))
    test_command = (arguments.get('test_command') or _DEF_TEST_CMD)
    if not target_path:
        raise ValidationError('target_path is required')

    transcript = []

    # 1) Generate initial tests
    gen_prompt = (
        f"Write focused pytest tests for {target_path}. Goal: {goal or 'improve reliability and correctness.'} "
        "Return one or more fenced code blocks where the FIRST non-empty line is '# File: <relative/test_path>.py'."
    )
    tests_text = _run_llm(server, gen_prompt, intent='tdd', role='Test Author')
    transcript.append({'stage': 'generate_tests', 'content': _compact_text(tests_text, 2000)})

    # Optionally apply proposed tests
    if apply_fixes:
        applied = _apply_proposed_changes(tests_text, dry_run=False)
        transcript.append({'stage': 'apply_tests', 'applied': applied})

    # 2) Loop: run tests -> fix -> run again
    for i in range(max_loops):
        run_out = handle_test_execution({'test_command': test_command})
        transcript.append({'stage': f'test_run_{i+1}', 'result': _compact_text(run_out, 2000)})
        if 'failed' not in run_out.lower() and 'error' not in run_out.lower():
            break
        # Ask Developer to propose minimal fixes given failures
        fix_prompt = (
            f"Tests failed for {target_path}. Test output (truncated):\n{_compact_text(run_out, 2000)}\n\n"
            "Propose minimal code changes to fix failures. Use fenced blocks; each block's first non-empty line must be '# File: <path>'."
        )
        fix_text = _run_llm(server, fix_prompt, intent='implementation', role='Developer')
        transcript.append({'stage': f'fix_proposal_{i+1}', 'content': _compact_text(fix_text, 2000)})
        if apply_fixes:
            applied = _apply_proposed_changes(fix_text, dry_run=False)
            transcript.append({'stage': f'apply_fix_{i+1}', 'applied': applied})

    return json.dumps(transcript)





def handle_memory_retrieve_semantic(arguments, server):
    query = (arguments.get("query") or "").lower()
    if not query:
        raise ValidationError("'query' is required")
    limit = int(arguments.get("limit", 5))
    rows = server.storage.retrieve_memory(category="semantic_memory", limit=200) or []
    items = []
    for r in rows:
        try:
            d = json.loads(r.get("value", "{}"))
            txt = (d.get("summary") or "").lower()
            tokens_q = set(re.findall(r"[a-z0-9_]+", query))
            tokens_t = set(re.findall(r"[a-z0-9_]+", txt))
            score = len(tokens_q & tokens_t)
            items.append({"key": r.get("key"), "summary": d.get("summary"), "score": score})
        except Exception:
            continue
    items.sort(key=lambda x: x["score"], reverse=True)
    return {"results": items[:limit]}


# Router tools
_router_history: list[dict] = []

def handle_router_diagnostics(arguments, server):
    lim = int(arguments.get('limit', 20))
    # Prefer robust router analytics if available
    try:
        if getattr(server, '_router_agent', None) is not None:
            try:
                # call async analytics synchronously
                try:
                    data = asyncio.get_event_loop().run_until_complete(server._router_agent.get_router_analytics(hours=24))
                except RuntimeError:
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                    data = loop.run_until_complete(server._router_agent.get_router_analytics(hours=24)); loop.close()
                # include recent decisions as well
                recent = server.storage.retrieve_memory(category='router_decisions', limit=lim) or []
                return {"analytics": data, "recent": [json.loads(m.get('value','{}')) for m in recent if isinstance(m.get('value'), str)]}
            except Exception:
                pass
    except Exception:
        pass
    # Fallback to in-memory log
    try:
        hist = list(server._router_log)[-lim:]
    except Exception:
        hist = []
    return {'decisions': hist}


def handle_router_test(arguments, server):
    task = (arguments.get('task') or '').strip()
    if not task:
        raise ValidationError('task is required')
    role = arguments.get('role')
    intent = arguments.get('intent')
    complexity = arguments.get('complexity', 'auto')
    # Prefer robust router
    backend = None; model = None; confidence = None; reasoning = None
    try:
        if getattr(server, '_router_agent', None) is not None:
            try:
                try:
                    task_prof = asyncio.get_event_loop().run_until_complete(server._router_agent.analyze_task(task, intent=intent, role=role))
                except RuntimeError:
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                    task_prof = loop.run_until_complete(server._router_agent.analyze_task(task, intent=intent, role=role)); loop.close()
                if complexity != 'auto':
                    task_prof.complexity = complexity
                try:
                    dec = asyncio.get_event_loop().run_until_complete(server._router_agent.decide_routing(task_prof))
                except RuntimeError:
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                    dec = loop.run_until_complete(server._router_agent.decide_routing(task_prof)); loop.close()
                backend = dec.backend.value if hasattr(dec, 'backend') else None
                model = getattr(dec, 'model', None)
                confidence = getattr(dec, 'confidence', None)
                reasoning = getattr(dec, 'reasoning', None)
            except Exception:
                pass
    except Exception:
        pass
    # Fallback to lightweight JSON router if still unknown
    if not backend:
        try:
            try:
                agent_dec = asyncio.get_event_loop().run_until_complete(server.decide_backend_via_router(task, intent=intent, role=role))
            except RuntimeError:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                agent_dec = loop.run_until_complete(server.decide_backend_via_router(task, intent=intent, role=role)); loop.close()
            backend = (agent_dec.get('backend') or '').lower()
            model = agent_dec.get('preferred_model') or model
            confidence = agent_dec.get('confidence', confidence)
            reasoning = agent_dec.get('reason', reasoning)
        except Exception:
            pass
    if not backend:
        backend = 'lmstudio'
        reasoning = reasoning or 'heuristic default'
    return f"Backend: {backend}\nModel: {model or 'auto'}\nConfidence: {confidence if confidence is not None else 'n/a'}\nComplexity: {complexity}\nReason: {reasoning or 'n/a'}"


def handle_router_update_profile(arguments, server):
    if getattr(server, '_router_agent', None) is None:
        return 'Router agent not available'
    backend = (arguments.get('backend') or '').lower()
    updates = arguments.get('updates') or {}
    if not backend or not updates:
        raise ValidationError('backend and updates are required')
    # Map string to enum if present
    try:
        from improvements.router import Backend as B
        bmap = {'lmstudio': B.LMSTUDIO, 'openai': B.OPENAI, 'anthropic': B.ANTHROPIC}
        be = bmap.get(backend)
        prof = server._router_agent.backend_profiles.get(be)
        if not prof:
            return 'Unknown backend'
        for k, v in updates.items():
            if hasattr(prof, k):
                setattr(prof, k, v)
        return 'Profile updated'
    except Exception as e:
        return f'Error updating profile: {e}'



def handle_workflow_execute(arguments, server):
    wf = _require_workflow()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    if not workflow_id:
        raise ValidationError("workflow_id is required")
    inputs = arguments.get("inputs") or {}
    try:
        out = asyncio.get_event_loop().run_until_complete(wf.execute_workflow(workflow_id, inputs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            out = loop.run_until_complete(wf.execute_workflow(workflow_id, inputs))
        finally:
            loop.close()
    return out


# --- Agentic Team Handlers ---


def _extract_fenced_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    fence = "```"
    i = 0
    while True:
        start = text.find(fence, i)
        if start == -1:
            break
        lang_end = text.find("\n", start + 3)
        if lang_end == -1:
            break
        lang = text[start + 3:lang_end].strip().lower()  # may be 'python', 'diff', etc.
        end = text.find(fence, lang_end + 1)
        if end == -1:
            break
        content = text[lang_end + 1:end]
        blocks.append((lang, content))
        i = end + 3
    return blocks


def _apply_proposed_changes(proposal_text: str, dry_run: bool = False) -> list[str]:
    """Apply proposed file changes from fenced code blocks.

    Supported format:
    ```python
    # File: relative/path.ext
    <file content>
    ```
    or
    ```
    # path: relative/path.ext
    <file content>
    ```

    When dry_run=True, do not write files; return messages describing what would be written.
    Unified diffs are not applied for safety.
    """
    applied: list[str] = []
    blocks = _extract_fenced_code_blocks(proposal_text)
    for lang, content in blocks:
        # Parse first non-empty line for file path marker
        lines = [ln for ln in content.splitlines()]
        idx = 0
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx >= len(lines):
            continue
        header = lines[idx].strip()
        path_val = None
        if header.lower().startswith("# file:"):
            path_val = header.split(":", 1)[1].strip()
            idx += 1
        elif header.lower().startswith("# path:"):
            path_val = header.split(":", 1)[1].strip()
            idx += 1
        if not path_val:
            continue
        # Remaining content is file body
        body = "\n".join(lines[idx:])
        try:
            sp = _safe_path(path_val)
            if dry_run:
                applied.append(f"Would write {path_val} ({len(body)} chars)")
            else:
                sp.parent.mkdir(parents=True, exist_ok=True)
                with open(sp, "w", encoding="utf-8") as f:
                    f.write(body)
                applied.append(f"Wrote {path_val} ({len(body)} chars)")
        except Exception as e:
            applied.append(f"Failed {path_val}: {e}")
    return applied

# Expert selection based on task description
_DEF_EXPERTS = [
    ("security", "Security Reviewer", "Identify vulnerabilities and suggest remediations."),
    ("perf", "Performance Analyzer", "Identify performance bottlenecks and optimizations."),
    ("performance", "Performance Analyzer", "Identify performance bottlenecks and optimizations."),
    ("frontend", "FrontEnd Expert", "Implement UI logic and ensure accessibility."),
    ("backend", "Backend Expert", "Implement server logic, APIs, and data flows."),
    ("deep", "DeepLearning Expert", "Design and optimize ML/DL pipelines."),
    ("ml", "DeepLearning Expert", "Design and optimize ML/DL pipelines."),
    ("biology", "Biology Expert", "Domain guidance for bio-related problems."),
    ("physics", "Physics Expert", "Domain guidance for physics problems."),
    ("math", "Mathematics Expert", "Formal reasoning and proofs."),
    ("pharma", "Pharmaceutical Expert", "Drug development and regulations."),
    ("business", "Business Expert", "Product and market strategy tradeoffs."),
]


_RESEARCH_DIR_PATTERN = re.compile(r"<<RESEARCH:\s*(.+?)>>", re.IGNORECASE)


def _detect_research_directives(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [m.strip() for m in _RESEARCH_DIR_PATTERN.findall(text) if m.strip()]


def _perform_research_queries(queries: list[str], server) -> str:
    """Run quick deep_research for each query and return a compact combined summary."""
    chunks: list[str] = []
    for q in queries[:3]:  # safety cap
        try:
            summary = handle_deep_research({"query": q, "time_limit": 90, "max_depth": 2}, server)
            chunks.append(f"Query: {q}\n{_compact_text(summary, max_chars=1200)}")
        except Exception as e:
            chunks.append(f"Query: {q}\nError running research: {e}")
    return "\n\n".join(chunks)

# --- Multi-backend LLM selection for CrewAI ---



# Utility: extract up to N queries from bullet-point text
def _extract_queries_from_bullets(text: str, max_n: int = 3) -> list[str]:
    queries: list[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # remove leading bullet markers
        for prefix in ("- ", "* ", "• ", "1. ", "2. ", "3. "):
            if ln.lower().startswith(prefix.strip()):
                ln = ln[len(prefix):].strip()
                break
        # drop trailing reasons after ' - ' or ' — '
        for sep in (" - ", " — "):
            if sep in ln:
                ln = ln.split(sep, 1)[0].strip()
        if ln:
            queries.append(ln)
        if len(queries) >= max_n:
            break
    return queries

# --- Multi-backend LLM selection for CrewAI ---

def _build_llm_for_backend(backend: str):
    try:
        try:
            from crewai import LLM  # type: ignore
        except Exception:
            from crewai.llm import LLM  # type: ignore
        backend = (backend or "").lower()
        if backend == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt5")
            if not key:
                return None
            return LLM(model=model, api_key=key, base_url=base, temperature=0.2)
        if backend == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet")
            if not key:
                return None
            return LLM(model=model, api_key=key, base_url=base, temperature=0.2)
        # default lmstudio
        base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        key = os.getenv("OPENAI_API_KEY", "sk-noauth")
        model = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
        return LLM(model=model, api_key=key, base_url=base, temperature=0.2)
    except Exception:
        return None


def _decide_backend_for_role(role: str, task_desc: str) -> str:
    # Delegate to enhanced module first; fall back to built-in heuristics
    try:
        return _enh_decide_backend_for_role(role, task_desc)
    except Exception:
        pass
    """Heuristic backend selection per role and task description.
    - Complex reasoning (security, math, physics, deep learning): prefer anthropic, then openai, else lmstudio
    - General coding (planner, scheduler, backend/frontend, coder, reviewer): lmstudio by default
    - Allow override via AGENT_BACKEND_OVERRIDE (e.g., 'all=openai', 'security=anthropic')
    """
    override = os.getenv("AGENT_BACKEND_OVERRIDE", "").strip().lower()
    if override:
        # simple format: "all=openai" or "security=anthropic,math=openai"
        try:
            entries = [x.strip() for x in override.split(",") if x.strip()]
            mapping = {}
            for e in entries:
                k, v = [p.strip() for p in e.split("=", 1)]
                mapping[k] = v
            key = role.lower().split()[0]
            if key in mapping:
                return mapping[key]
            if "all" in mapping:
                return mapping["all"]
        except Exception:
            pass

    desc = (task_desc or "").lower()
    role_l = (role or "").lower()
    prefer_complex = any(w in role_l for w in ["security", "math", "physics", "deep", "pharma"]) or \
        any(w in desc for w in ["formal proof", "vulnerability", "theorem", "derivation", "attack", "model eval"])

    if prefer_complex:
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "lmstudio"
    # default
    return "lmstudio"


def handle_propose_research(arguments, server):
    problem = (arguments.get("problem") or "").strip()
    if not problem:
        raise ValidationError("'problem' is required")
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
            resp = loop.run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
            loop.close()
    except Exception as e:
        resp = f"Error proposing research: {e}"
    return _compact_text(resp, max_chars=4000)


def _create_agent(role: str, goal: str, backstory: str, task_desc: str):
    try:
        from crewai import Agent  # type: ignore
        backend = _decide_backend_for_role(role, task_desc)
        llm = _build_llm_for_backend(backend)
        kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            kwargs["llm"] = llm
        logger.info("Creating agent '%s' -> backend=%s model=%s", role, backend, getattr(llm, 'model', 'n/a'))
        return Agent(role=role, goal=goal, backstory=backstory, **kwargs)
    except Exception as e:
        logger.warning("Failed to create agent '%s': %s", role, e)
        return None



def _select_experts(task_desc: str):
    try:
        from crewai import Agent  # type: ignore
        llm = _make_crewai_llm()
        base_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            base_kwargs["llm"] = llm
        desc = (task_desc or "").lower()
        agents = []
        for key, role, goal in _DEF_EXPERTS:
            if key in desc:
                agents.append(Agent(role=role, goal=goal, backstory=f"Specialist in {role} concerns.", **base_kwargs))
        # Provide generic FE/BE for code tasks if none matched
        if not agents and any(w in desc for w in ["code", "refactor", "api", "ui", "frontend", "backend"]):
            agents.extend([
                Agent(role="Backend Expert", goal="Implement server logic.", backstory="Backend-focused engineer.", **base_kwargs),
                Agent(role="FrontEnd Expert", goal="Implement UI logic.", backstory="Frontend-focused engineer.", **base_kwargs),
            ])
        return agents
    except Exception:
        return []


def _make_crewai_llm():
    try:
        try:
            from crewai import LLM  # type: ignore
        except Exception:
            from crewai.llm import LLM  # type: ignore
        # Prefer LMSTUDIO_* if set, else fallback to OPENAI_* for compatibility
        lmstudio_base = os.getenv("LMSTUDIO_API_BASE") or os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        lmstudio_key = os.getenv("LMSTUDIO_API_KEY") or os.getenv("OPENAI_API_KEY", "sk-noauth")
        hardwired_model = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
        return LLM(model=hardwired_model, base_url=lmstudio_base, api_key=lmstudio_key, temperature=0.2)
    except Exception:
        return None

def _read_files_for_context(paths: list[str]) -> str:
    parts = []
    for p in paths or []:
        try:
            sp = _safe_path(p)
            if sp.exists() and sp.is_file():
                with open(sp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                parts.append(f"# File: {p}\n{content}\n")
        except Exception:
            continue
    return _compact_text("\n\n".join(parts), max_chars=4000)

def handle_agent_team_plan_and_code(arguments, server):
    task_desc = (arguments.get("task") or "").strip()
    if not task_desc:
        raise ValidationError("'task' is required")
    target_files = arguments.get("target_files") or []
    constraints = (arguments.get("constraints") or "").strip()
    apply_changes = bool(arguments.get("apply_changes", False))
    auto_research_rounds = int(arguments.get("auto_research_rounds", 0))

    file_ctx = _read_files_for_context(target_files)

    try:
        # Allow tests or offline mode to force fallback
        if os.getenv("AGENT_TEAM_FORCE_FALLBACK") == "1":
            raise RuntimeError("forced_fallback")
        from crewai import Agent, Crew, Task
        llm = _make_crewai_llm()
        base_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            base_kwargs["llm"] = llm

        planner = Agent(role="Planner", goal="Create a practical, step-by-step plan to complete the task.", backstory="Senior tech lead who scopes and sequences work effectively.", **base_kwargs)
        coder = Agent(role="Coder", goal="Draft clean, minimal code changes with clear diffs and tests.", backstory="Pragmatic engineer who favors readability and tests.", **base_kwargs)
        reviewer = Agent(role="Reviewer", goal="Review the patch for correctness, safety, and tests.", backstory="Staff engineer who catches risks and improves tests.", **base_kwargs)

        t_plan = Task(description=f"Task: {task_desc}\nConstraints: {constraints}\nFiles Context (truncated):\n{file_ctx}", agent=planner)
        t_code = Task(description=(
            f"Produce proposed code changes for: {target_files}. Prefer fenced code blocks. "
            "For each code block, the FIRST non-empty line MUST be '# File: <relative/path>'. "
            "If you propose multiple files, provide multiple fenced blocks, one per file. Include brief rationale and list of files changed."
        ), agent=coder)
        t_review = Task(description="Review the proposed changes and list concrete fixes or approvals. Ensure tests are present.", agent=reviewer)

        # Optional orchestration & scheduling
        experts = _select_experts(task_desc)
        orchestrator = Agent(role="Orchestrator", goal="Choose relevant experts and coordinate their inputs.", backstory="Director who routes work to domain experts.", **base_kwargs)
        scheduler = Agent(role="Scheduler", goal="Propose an execution order and parallelization plan.", backstory="PM who sequences work efficiently.", **base_kwargs)
        # Orchestrator can decide whether research is needed; also parse inline directives like <<RESEARCH: topic>>
        research_directives = _detect_research_directives(task_desc + "\n" + constraints)
        research_summary = None
        if research_directives:
            logger.info("Research directives detected: %s", research_directives)
            research_summary = _perform_research_queries(research_directives, server)

        t_orch = Task(description=(
            f"Given the task, select relevant experts and assign sub-goals. If more research is needed, say so explicitly and specify queries.\n\n"
            f"Task: {task_desc}\nConstraints: {constraints}\n\nPrior Research (if any):\n{research_summary or 'n/a'}"
        ), agent=orchestrator)
        t_sched = Task(description="Propose an ordered list of steps for the team.", agent=scheduler)

        crew = Crew(agents=[planner, coder, reviewer, orchestrator, scheduler] + experts, tasks=[t_plan, t_orch, t_sched, t_code, t_review], verbose=False)
        out = str(crew.kickoff())
        # Allow agents to request more research using inline directives in the first pass
        post_directives = _detect_research_directives(out)
        if post_directives:
            logger.info("Post-run research directives detected: %s", post_directives)
            post_research = _perform_research_queries(post_directives, server)
            out += f"\n\n[Research Results]\n{post_research}"
            if auto_research_rounds > 0:
                auto_research_rounds = max(0, auto_research_rounds - 1)
                # Minimal second pass prompt that includes research results
                try:
                    from crewai import Task
                    t_refine = Task(description=(
                        "Refine plan and code suggestions using the new research results. "
                        "If patches changed, output updated fenced code blocks.\n\n"
                        f"Research Results:\n{post_research}"
                    ), agent=planner)
                    crew2 = Crew(agents=[planner, coder, reviewer, orchestrator, scheduler] + experts, tasks=[t_refine], verbose=False)
                    out2 = str(crew2.kickoff())
                    out += "\n\n[Refine Pass]\n" + out2
                except Exception as e2:
                    out += f"\n\n[Refine Pass Error] {e2}"
        if apply_changes:
            dry = os.getenv("APPLY_DRY_RUN", "0").strip().lower() in {"1","true","yes","on"}
            applied = _apply_proposed_changes(out, dry_run=dry)
            header = "[Dry Run] Would apply changes" if dry else "[Applied changes]"
            out += f"\n\n{header}\n" + "\n".join(applied)
            logger.info("agent_team_plan_and_code apply_changes=%s dry_run=%s; applied: %s", apply_changes, dry, len(applied))
        return _compact_text(out, max_chars=4000)
    except Exception as e:
        # Fallback: single-pass synthesis via LM Studio
        prompt = (
            "You are a planning and coding team. Given a task, optional constraints, and file context, "
            "produce: (1) a concise plan, (2) proposed changes (diff or fenced code), (3) test suggestions.\n\n"
            f"Task: {task_desc}\nConstraints: {constraints}\n\nFiles Context (truncated):\n{file_ctx}\n\nOutput steps 1-3."
        )
        try:
            try:
                resp = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
                loop.close()
        except Exception as e2:
            resp = f"Error synthesizing plan: {e}; fallback failed: {e2}"
        # Apply changes if requested and response includes fenced code blocks
        if apply_changes:
            applied = _apply_proposed_changes(resp)
            resp += "\n\n[Applied changes]\n" + "\n".join(applied)
        return _compact_text(resp, max_chars=4000)

def handle_agent_team_review_and_test(arguments, server):
    diff = (arguments.get("diff") or "").strip()
    if not diff:
        raise ValidationError("'diff' is required")
    context = (arguments.get("context") or "").strip()
    apply_fixes = bool(arguments.get("apply_fixes", False))
    max_loops = int(arguments.get("max_loops", 1))
    test_command = (arguments.get("test_command") or _DEF_TEST_CMD)

    try:
        if os.getenv("AGENT_TEAM_FORCE_FALLBACK") == "1":
            raise RuntimeError("forced_fallback")
        Agent = globals().get("Agent")
        Crew = globals().get("Crew")
        Task = globals().get("Task")
        if not (Agent and Crew and Task):
            from crewai import Agent as _CAAgent, Crew as _CACrew, Task as _CATask
            Agent, Crew, Task = _CAAgent, _CACrew, _CATask
        llm = _make_crewai_llm()
        base_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            base_kwargs["llm"] = llm
        reviewer = Agent(role="Reviewer", goal="Assess diff for correctness/risk and request fixes.", backstory="Thorough code reviewer.", **base_kwargs)
        test_author = Agent(role="Test Author", goal="Propose focused tests (pytest).", backstory="Engineer who writes tests first.", **base_kwargs)
        t_rev = Task(description=f"Review this diff and list issues, risks, and fixes. Context: {context}\n\nDiff:\n{diff}", agent=reviewer)
        t_tests = Task(description=f"Propose pytest tests that validate the changes above. Provide fenced code blocks.", agent=test_author)
        crew = Crew(agents=[reviewer, test_author], tasks=[t_rev, t_tests], verbose=False)
        out = str(crew.kickoff())

        # Apply tests if provided
        applied_tests = []
        if apply_fixes:
            applied_tests = _apply_proposed_changes(out, dry_run=False)
            out += "\n\n[Applied tests]\n" + "\n".join(applied_tests)

        # Run tests and loop minimal fixes if requested
        transcript = []
        for i in range(max_loops):
            run_out = handle_test_execution({"test_command": test_command})
            transcript.append({"stage": f"test_run_{i+1}", "result": _compact_text(run_out, 1500)})
            if 'failed' not in run_out.lower() and 'error' not in run_out.lower():
                break
            # Route developer for minimal fix proposals
            fix_prompt = (
                f"Review found issues; tests failing. Provide minimal fixes in fenced blocks. Context: {context}\n\nDiff:\n{diff}\n\n"
                f"Test output (truncated):\n{_compact_text(run_out, 1500)}"
            )
            fix_text = _run_llm(server, fix_prompt, intent='implementation', role='Developer')
            out += "\n\n[Fix Proposal]\n" + _compact_text(fix_text, 2000)
            if apply_fixes:
                applied = _apply_proposed_changes(fix_text, dry_run=False)
                out += "\n\n[Applied fix]\n" + "\n".join(applied)
        # Append transcript summary
        if transcript:
            out += "\n\n[Test Transcript]\n" + "\n".join([json.dumps(t) for t in transcript])
        return _compact_text(out, max_chars=4000)
    except Exception as e:
        # Fallback synthesis
        prompt = (
            "Review the following diff and produce: (1) review notes and risks, (2) pytest tests in fenced code blocks.\n\n"
            f"Context: {context}\n\nDiff:\n{diff}"
        )
        try:
            try:
                resp = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
                loop.close()
        except Exception as e2:
            resp = f"Error synthesizing review: {e}; fallback failed: {e2}"
        return _compact_text(resp, max_chars=4000)

def handle_agent_team_refactor(arguments, server):
    module_path = (arguments.get("module_path") or "").strip()
    goals = (arguments.get("goals") or "").strip()
    if not module_path:
        raise ValidationError("'module_path' is required")
    content = _read_files_for_context([module_path])

    try:
        from crewai import Agent, Crew, Task
        llm = _make_crewai_llm()
        base_kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            base_kwargs["llm"] = llm
        refactorer = Agent(role="Refactorer", goal="Propose clearer, modular refactor with docstrings.", backstory="Engineer focused on readability and maintainability.", **base_kwargs)
        qa = Agent(role="QA", goal="Ensure refactor preserves behavior; suggest tests.", backstory="QA who validates behavior.", **base_kwargs)
        t_ref = Task(description=f"Refactor goals: {goals}. Provide a rationale and a refactored version in fenced code.\n\nCurrent content (truncated):\n{content}", agent=refactorer)
        t_qa = Task(description="List behavioral risks, migration steps, and propose tests.", agent=qa)
        crew = Crew(agents=[refactorer, qa], tasks=[t_ref, t_qa], verbose=False)
        out = str(crew.kickoff())
        return _compact_text(out, max_chars=4000)
    except Exception as e:
        prompt = (
            "Given the current module content and refactor goals, propose: (1) rationale, (2) refactored code in fenced blocks, (3) tests.\n\n"
            f"Goals: {goals}\n\nCurrent content (truncated):\n{content}"
        )
        try:
            try:
                resp = asyncio.get_event_loop().run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(server.make_llm_request_with_retry(prompt, temperature=0.2))
                loop.close()
        except Exception as e2:
            resp = f"Error synthesizing refactor: {e}; fallback failed: {e2}"
        return _compact_text(resp, max_chars=4000)


        # Find matching files under safe directory
        search_path = str(rp / "**" / file_pattern)
        matching_files = glob.glob(search_path, recursive=True)

        for file_path in matching_files:
            fpath = Path(file_path)
            try:
                fpath.relative_to(_BASE_DIR)
            except Exception:
                continue  # ensure stays inside base
            # skip ignored dirs
            if any(part in ignore_dirs for part in fpath.parts):
                continue
            if fpath.is_file():
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if search_regex.search(line):
                                results.append(f"{fpath}:{line_num}: {line.strip()}")
                except Exception:
                    continue  # Skip unreadable files

        if results:
            return f"Search results for '{pattern}':\n\n" + "\n".join(results[:200])  # higher cap but bounded
        else:
            return f"No matches found for pattern '{pattern}'"
    except ValidationError as e:
        raise
    except Exception as e:
        return f"Error searching files: {str(e)}"

def handle_memory_store(arguments, server):
    """Handle memory storage with persistent backend"""
    key = arguments.get("key", "")
    value = arguments.get("value", "")
    category = arguments.get("category", "general")

    success = server.storage.store_memory(key, value, category)

    if success:
        return f"✅ Stored memory with key '{key}' in category '{category}'"
    else:
        return f"❌ Failed to store memory with key '{key}'"

def handle_memory_retrieve(arguments, server):
    """Handle memory retrieval with enhanced search"""
    key = arguments.get("key")
    category = arguments.get("category")
    search_term = arguments.get("search_term")

    memories = server.storage.retrieve_memory(
        key=key,
        category=category,
        search_term=search_term,
        limit=50
    )

    if not memories:
        return "No matching memories found"

    # Format results
    results = []
    for memory in memories:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(memory['timestamp']))
        results.append(f"**{memory['key']}** ({memory['category']}) - {timestamp}\n{memory['value']}")

    return "📋 **Memory Search Results:**\n\n" + "\n\n".join(results)

def handle_performance_stats(arguments, server):
    """Handle performance statistics retrieval"""
    tool_name = arguments.get("tool_name")
    hours = arguments.get("hours", 24)

    stats = server.storage.get_performance_stats(tool_name=tool_name, hours=hours)

    if not stats.get('stats'):
        return f"No performance data found for the last {hours} hours"

    # Format performance report
    report = f"📊 **Performance Report (Last {hours} hours)**\n\n"

    for stat in stats['stats']:
        success_rate = f"{stat['success_rate']:.1%}"
        avg_time = f"{stat['avg_time']:.3f}s"
        max_time = f"{stat['max_time']:.3f}s"

        tool_report = f"""**{stat['tool_name']}**
- Total Calls: {stat['total_calls']}
- Success Rate: {success_rate}
- Avg Response Time: {avg_time}
- Max Response Time: {max_time}
- Errors: {stat['error_count']}

"""
        report += tool_report

    return report

def handle_error_patterns(arguments, server):
    """Handle error pattern analysis"""
    hours = arguments.get("hours", 24)

    patterns = server.storage.get_error_patterns(hours=hours)

    if not patterns:
        return f"No error patterns found in the last {hours} hours ✅"

def handle_health_check(arguments, server):
    """Health check with optional LM Studio readiness probe.
    Pass probe_lm=true to attempt a short model readiness check.
    """
    probe = arguments.get("probe_lm", False)
    lm_ready = None
    lm_error = None

    if probe:
        try:
            # Lightweight prompt with small timeout
            prompt = "ping"
            # Use the centralized LLM request with short backoff
            lm_resp = asyncio.get_event_loop().run_until_complete(
                server.make_llm_request_with_retry(prompt, temperature=0.0, retries=0, backoff=0.1)
            )
            # If the call returns a string without error prefix, consider ready
            lm_ready = not lm_resp.startswith("Error:")
            if not lm_ready:
                lm_error = lm_resp
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                lm_resp = loop.run_until_complete(
                    server.make_llm_request_with_retry("ping", temperature=0.0, retries=0, backoff=0.1)
                )
                lm_ready = not lm_resp.startswith("Error:")
                if not lm_ready:
                    lm_error = lm_resp
            finally:
                loop.close()
        except Exception as e:
            lm_ready = False
            lm_error = str(e)

    return {
        "ok": True,
        "storage": bool(server.storage is not None),
        "base_dir": str(_BASE_DIR),
        "execution_enabled": _execution_enabled(),
        "lm": {
            "url": server.base_url,
            "model": server.model_name,
            "ready": lm_ready,
            "error": lm_error,
        } if probe else None,
    }


def handle_get_version(arguments, server):
    return {
        "name": "enhanced-lmstudio-assistant",
        "version": "2.1.0",
        "lm_studio_url": server.base_url,
        "model": server.model_name,
    }

    # Format error report
    report = f"🚨 **Error Patterns (Last {hours} hours)**\n\n"

    for pattern in patterns:
        last_occurrence = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pattern['last_occurrence']))
        tool_info = f" in {pattern['tool_name']}" if pattern['tool_name'] else ""

        pattern_report = f"""**{pattern['error_type']}**{tool_info}
- Occurrences: {pattern['occurrence_count']}
- Last Seen: {last_occurrence}

"""
        report += pattern_report

    return report

def handle_debug_analysis(arguments, server):
    """Handle debugging analysis"""
    code = arguments.get("code", "")
    error_message = arguments.get("error_message", "")
    context = arguments.get("context", "")

    prompt = f"""Debug Analysis Request:

Code to analyze:
{code}

Error message (if any):
{error_message}

Additional context:
{context}

Please provide a comprehensive debugging analysis including:
1. Potential issues in the code
2. Likely causes of the error
3. Step-by-step debugging approach
4. Suggested fixes
5. Prevention strategies
"""

    try:
        analysis = asyncio.get_event_loop().run_until_complete(
            server.make_llm_request_with_retry(prompt, temperature=0.1, retries=2, backoff=0.5)
        )
        debug_key = f"debug_{hashlib.md5(code.encode()).hexdigest()[:8]}"
        server.storage.store_memory(
            key=debug_key,
            value=f"Code: {code[:100]}... | Error: {error_message} | Analysis: {analysis[:200]}...",
            category="debug"
        )
        return analysis
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            analysis = loop.run_until_complete(
                server.make_llm_request_with_retry(prompt, temperature=0.1, retries=2, backoff=0.5)
            )
            debug_key = f"debug_{hashlib.md5(code.encode()).hexdigest()[:8]}"
            server.storage.store_memory(
                key=debug_key,
                value=f"Code: {code[:100]}... | Error: {error_message} | Analysis: {analysis[:200]}...",
                category="debug"
            )
            return analysis
        finally:
            loop.close()
    except Exception as e:
        return f"Error in debug analysis: {str(e)}"

def handle_execution_trace(arguments, server):
    """Handle execution tracing"""
    code = arguments.get("code", "")
    inputs = arguments.get("inputs", "")

    # Create a traced version of the code
    traced_code = f"""
import sys
import traceback

def trace_execution():
    try:
        # Original code with tracing
        print("=== EXECUTION TRACE ===")
        {code}
        print("=== TRACE COMPLETE ===")
    except Exception as e:
        print(f"=== EXCEPTION OCCURRED ===")
        print(f"Exception: {{e}}")
        print(f"Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    trace_execution()
"""

    # Execute the traced code
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(traced_code)
            temp_file = f.name

        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )

        output = f"Execution Trace Results:\n\n"
        if result.stdout:
            output += f"{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        # Clean up
        os.unlink(temp_file)

        return output

    except Exception as e:
        return f"Error tracing execution: {str(e)}"

def main():
    """Main entry point for enhanced MCP server"""
    try:
        server = EnhancedLMStudioMCPServer()
        logger.info(f"Enhanced LM Studio MCP Server v2.1 starting...")
        logger.info(f"Connecting to: {server.base_url}")
        logger.info(f"Using model: {server.model_name}")
        logger.info(f"Working directory: {server.working_directory}")
        logger.info(f"Persistent storage initialized: {server.storage.db_path}")

        # MCP protocol communication via stdin/stdout
        for line in sys.stdin:
            try:
                message = json.loads(line.strip())
                response = handle_message(message)
                if response:
                    print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except Exception as e:
        logger.error(f"Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
