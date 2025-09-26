"""
Strands-style multi-agent orchestration integration for the LM Studio MCP server.
This package provides:
- Role definitions and prompts
- AgentFactory for dynamic expert instantiation
- TeamOrchestrator for plan/execute workflows
- TeamMemory for persistent team knowledge
"""

from .agent_roles import ROLE_DEFINITIONS
from .agent_factory import AgentFactory
from .team_orchestrator import TeamOrchestrator, OrchestratorConfig
from .team_memory import TeamMemory
from .workflows import WORKFLOW_TEMPLATES

__all__ = [
    "ROLE_DEFINITIONS",
    "AgentFactory",
    "TeamOrchestrator",
    "OrchestratorConfig",
    "TeamMemory",
    "WORKFLOW_TEMPLATES",
]

