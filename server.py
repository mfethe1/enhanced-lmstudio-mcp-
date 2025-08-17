import asyncio
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

# Import our new storage layer
from storage import MCPStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Singleton server instance (P1)
_server_singleton: Optional["EnhancedLMStudioMCPServer"] = None


def get_server_singleton():
    global _server_singleton
    if _server_singleton is None:
        _server_singleton = EnhancedLMStudioMCPServer()
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


# Safety and validation helpers (P0)
class ValidationError(Exception):
    """Raised for invalid user inputs or disallowed operations."""
    pass

# Allowed base directory for all filesystem operations
_BASE_DIR = Path(os.getenv("ALLOWED_BASE_DIR", os.getcwd())).resolve()


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

        # Initialize persistent storage
        self.storage = MCPStorage()
        # Performance monitoring settings
        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.2"))  # seconds


        # Performance monitoring settings
    async def make_llm_request_with_retry(self, prompt: str, temperature: float = 0.1, retries: int = 2, backoff: float = 0.5) -> str:
        """Centralized LLM request with simple retry/backoff (P1)"""
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
                        "max_tokens": 2000
                    },
                    timeout=(30, 120),
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

        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.2"))  # 200ms

        logger.info("Enhanced MCP Server initialized with persistent storage")

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

    async def make_llm_request(self, prompt: str, temperature: float = 0.1) -> str:
        """Make request to LM Studio with enhanced error handling"""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2000
                },
                timeout=(30, 120),
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
    return {
        "tools": [
            # Sequential Thinking & Problem Solving Tools
            {
                "name": "sequential_thinking",
                "description": "A detailed tool for dynamic and reflective problem-solving through thoughts. This tool helps analyze problems through a flexible thinking process that can adapt and evolve.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your current thinking step"
                        },
                        "next_thought_needed": {
                            "type": "boolean",
                            "description": "Whether another thought step is needed"
                        },
                        "thought_number": {
                            "type": "integer",
                            "description": "Current thought number",
                            "minimum": 1
                        },
                        "total_thoughts": {
                            "type": "integer",
                            "description": "Estimated total thoughts needed",
                            "minimum": 1
                        },
                        "is_revision": {
                            "type": "boolean",
                            "description": "Whether this revises previous thinking"
                        },
                        "revises_thought": {
                            "type": "integer",
                            "description": "Which thought is being reconsidered",
                            "minimum": 1
                        },
                        "branch_from_thought": {
                            "type": "integer",
                            "description": "Branching point thought number",
                            "minimum": 1
                        },
                        "branch_id": {
                            "type": "string",
                            "description": "Branch identifier"
                        },
                        "needs_more_thoughts": {
                            "type": "boolean",
                            "description": "If more thoughts are needed"
                        }
                    },
                    "required": ["thought", "next_thought_needed", "thought_number", "total_thoughts"]
                }
            },
            {
                "name": "get_thinking_session",
                "description": "Retrieve all stored thoughts for a sequential thinking session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "summarize_thinking_session",
                "description": "Summarize a sequential thinking session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"}
                    },
                    "required": ["session_id"]
                }
            },
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

            # System Tools
            {
                "name": "health_check",
                "description": "Check server health and connectivity",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_version",
                "description": "Get server version and system information",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },

            # Enhanced Code Analysis Tools
            {
                "name": "analyze_code",
                "description": "Analyze code using olympiccoder-32b model for bugs, improvements, and explanations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis (bugs, optimization, explanation, refactor)",
                            "enum": ["bugs", "optimization", "explanation", "refactor"]
                        }
                    },
                    "required": ["code", "analysis_type"]
                }
            },

            {
                "name": "explain_code",
                "description": "Get detailed explanation of how code works using olympiccoder-32b",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to explain"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "suggest_improvements",
                "description": "Get suggestions for code improvements using olympiccoder-32b",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to improve"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "generate_tests",
                "description": "Generate unit tests for given code using olympiccoder-32b",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to generate tests for"
                        },
                        "framework": {
                            "type": "string",
                            "description": "Testing framework to use (pytest, unittest, jest, etc.)",
                            "default": "pytest"
                        }
                    },
                    "required": ["code"]
                }
            },

            # Code Execution & Testing Tools
            {
                "name": "execute_code",
                "description": "Execute code safely in a temporary environment and return results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (python, javascript, bash)",
                            "enum": ["python", "javascript", "bash"],
                            "default": "python"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds",
                            "default": 30
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "run_tests",
                "description": "Run tests in the current project and return results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "test_command": {
                            "type": "string",
                            "description": "Test command to run (e.g., 'pytest', 'npm test')"
                        },
                        "test_file": {
                            "type": "string",
                            "description": "Specific test file to run (optional)"
                        }
                    },
                    "required": ["test_command"]
                }
            },

            # File System & Project Management Tools
            {
                "name": "read_file_content",
                "description": "Read content from a file in the project",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Starting line number (optional)"
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Ending line number (optional)"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "write_file_content",
                "description": "Write content to a file in the project",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Write mode (overwrite, append)",
                            "enum": ["overwrite", "append"],
                            "default": "overwrite"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List files and directories in a path",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list",
                            "default": "."
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Include hidden files",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "search_files",
                "description": "Search for text patterns in files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex supported)"
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "File pattern to search in (e.g., '*.py')",
                            "default": "*"
                        },
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in",
                            "default": "."
                        }
                    },
                    "required": ["pattern"]
                }
            },

            # Memory & Context Management Tools (Enhanced)
            {
                "name": "store_memory",
                "description": "Store information in persistent memory for later retrieval",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key identifier"
                        },
                        "value": {
                            "type": "string",
                            "description": "Information to store"
                        },
                        "category": {
                            "type": "string",
                            "description": "Memory category (e.g., 'error', 'solution', 'pattern')",
                            "default": "general"
                        }
                    },
                    "required": ["key", "value"]
                }
            },
            {
                "name": "retrieve_memory",
                "description": "Retrieve information from persistent memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key to retrieve (optional)"
                        },
                        "category": {
                            "type": "string",
                            "description": "Memory category to search (optional)"
                        },
                        "search_term": {
                            "type": "string",
                            "description": "Search term to find related memories (optional)"
                        }
                    }
                }
            },

            # Performance Monitoring Tools (New)
            {
                "name": "get_performance_stats",
                "description": "Get performance statistics for tool usage and system health",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Specific tool to get stats for (optional)"
                        },
                        "hours": {
                            "type": "integer",
                            "description": "Time period in hours (default: 24)",
                            "default": 24
                        }
                    }
                }
            },
            {
                "name": "get_error_patterns",
                "description": "Get error patterns and trends for debugging and improvement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "Time period in hours (default: 24)",
                            "default": 24
                        }
                    }
                }
            },

            # Advanced Debugging Tools
            {
                "name": "debug_analyze",
                "description": "Analyze code for potential debugging issues and provide detailed analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to debug"
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Error message if available"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context about the issue"
                        }
                    }
                }
            },
            {
                "name": "trace_execution",
                "description": "Trace code execution step by step for debugging",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to trace"
                        },
                        "inputs": {
                            "type": "string",
                            "description": "Input values for testing"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]
    }


            # System Tools
def _sanitize_url(url: str) -> str:
    try:
        # Basic URL sanitation: strip whitespace and guard against data: or javascript:
        u = (url or "").strip()
        if u.startswith("data:") or u.startswith("javascript:"):
            return ""
        return u
    except Exception:
        return ""


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
    try:
        from functions import firecrawl_deep_research_firecrawl_mcp as firecrawl_deep
    except Exception:
        firecrawl_deep = None
    try:
        if firecrawl_deep is not None:
            fc = firecrawl_deep({
                "query": query,
                "maxDepth": max(1, min(max_depth, 10)),
                "timeLimit": max(30, min(time_limit, 300)),
                "maxUrls": 40,
            })
            # Normalize Firecrawl output
            stage1 = {
                "final": (fc.get("data", {}) or {}).get("finalAnalysis") if isinstance(fc, dict) else None,
                "raw": fc,
            }
        else:
            stage1 = {"final": None, "raw": None, "warning": "Firecrawl MCP not available"}
    except Exception as e:
        stage1 = {"final": None, "error": f"Firecrawl error: {str(e)}"}
    artifacts["stage1"] = stage1

    # Stage 2: CrewAI multi-agent synthesis (optional; best-effort)
    try:
        from crewai import Agent, Crew, Task
        # Create agents with focused roles
        analyst = Agent(
            role="Analyst",
            goal="Categorize and evaluate Firecrawl findings, extract key facts and gaps.",
            backstory="Senior research analyst skilled at distilling insights from diverse sources.",
            allow_delegation=False,
            verbose=False,
        )
        researcher = Agent(
            role="Researcher",
            goal="Identify missing angles and perform targeted follow-ups based on gaps.",
            backstory="Curious investigator who knows where to look for authoritative sources.",
            allow_delegation=False,
            verbose=False,
        )
        synthesizer = Agent(
            role="Synthesizer",
            goal="Produce concise, actionable recommendations tailored to the query.",
            backstory="Executive-level writer focusing on clarity and actionability.",
            allow_delegation=False,
            verbose=False,
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
        stage2 = {"report": None, "error": f"CrewAI error: {str(e)}"}

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
        f"Stage 2 (CrewAI): {stage2_note[:600]}\n\n"
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
            "sequential_thinking": (handle_sequential_thinking, True),
            "analyze_code": (lambda args, srv: handle_llm_analysis_tool("analyze_code", args, srv), True),
            "explain_code": (lambda args, srv: handle_llm_analysis_tool("explain_code", args, srv), True),
            "suggest_improvements": (lambda args, srv: handle_llm_analysis_tool("suggest_improvements", args, srv), True),
            "generate_tests": (lambda args, srv: handle_llm_analysis_tool("generate_tests", args, srv), True),
            "execute_code": (handle_code_execution, False),
            "run_tests": (handle_test_execution, False),
            "read_file_content": (handle_file_read, False),
            "write_file_content": (handle_file_write, False),
            "list_directory": (handle_directory_list, False),
            "search_files": (handle_file_search, False),
            "store_memory": (handle_memory_store, True),
            "retrieve_memory": (handle_memory_retrieve, True),
            "get_performance_stats": (handle_performance_stats, True),
            "get_error_patterns": (handle_error_patterns, True),
            "health_check": (handle_health_check, True),
            "get_version": (handle_get_version, True),
            "debug_analyze": (handle_debug_analysis, True),
            "trace_execution": (handle_execution_trace, True),
            "get_thinking_session": (handle_get_thinking_session, True),
            "summarize_thinking_session": (handle_summarize_thinking_session, True),
            "deep_research": (handle_deep_research, True),
            "get_research_details": (handle_get_research_details, True),
        }

        if tool_name not in registry:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        handler, needs_server = registry[tool_name]
        monitored = server.monitor_performance(tool_name)
        if needs_server:
            result = monitored(handler)(arguments, server)
        else:
            result = monitored(handler)(arguments)

            # Unreachable with registry, kept for safety
            pass

        # Always return text content per MCP schema
        payload_text = _to_text_content(result)
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

    try:
        import glob
        import re

        rp = _safe_directory(directory)
        results = []
        search_regex = re.compile(pattern, re.IGNORECASE)

        # Default ignore directories
        ignore_dirs = {".git", "node_modules", "venv", ".venv", "dist", "build"}

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
