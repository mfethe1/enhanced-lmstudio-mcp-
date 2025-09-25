"""
AgentFactory: create domain experts on-demand from natural language descriptions.
Uses existing LM Studio MCP server routing + circuit breakers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from .agent_roles import ROLE_DEFINITIONS


@dataclass
class AgentSpec:
    role: str
    system: str
    preferred_backend: str | None = None
    tools_allowlist: List[str] | None = None


class AgentFactory:
    def __init__(self, server):
        self.server = server  # EnhancedLMStudioMCPServer

    def from_description(self, description: str) -> List[AgentSpec]:
        """Map free-text task description to a set of AgentSpec.
        Simple heuristic: pick roles by keywords. Extend with LLM routing if desired.
        """
        d = description.lower()
        picked: List[str] = []
        def pick(role: str, *keys: str):
            if any(k in d for k in keys):
                picked.append(role)
        # Software
        pick("frontend_developer", "ui", "frontend", "react", "next")
        pick("backend_developer", "api", "backend", "service", "fastapi")
        pick("devops_engineer", "deploy", "docker", "k8s", "infra")
        pick("cs_optimization", "optimiz", "complexity", "algorithm")
        # Research
        pick("computational_chemist", "docking", "qm", "chem")
        pick("computational_biologist", "omics", "rna", "biology")
        pick("bioinformatician", "pipeline", "genome", "alignment")
        pick("data_scientist", "model", "regression", "classification")
        # Lab
        pick("wet_lab_chemist", "synthesis", "organic", "assay")
        pick("enzymologist", "enzyme", "kinetic")
        pick("fermentation_scientist", "ferment", "bioprocess")
        pick("analytical_chemist", "hplc", "ms", "nmr", "analytics")
        # PM/QA
        pick("research_coordinator", "plan", "milestone", "schedule")
        pick("quality_assurance", "acceptance", "qa", "verify")
        pick("technical_writer", "doc", "report", "manuscript")

        if not picked:
            picked = ["research_coordinator", "technical_writer"]

        agents: List[AgentSpec] = []
        for role in picked:
            rd = ROLE_DEFINITIONS.get(role, {})
            agents.append(AgentSpec(
                role=role,
                system=rd.get("system", f"You are a {role.replace('_', ' ')}."),
                preferred_backend=rd.get("preferred_backend"),
            ))
        return agents

    def instantiate(self, specs: List[AgentSpec]) -> List[Dict[str, Any]]:
        """Return lightweight agent dicts that the orchestrator can route through server"""
        agents = []
        for s in specs:
            agents.append({
                "role": s.role,
                "system": s.system,
                "preferred_backend": s.preferred_backend,
                "tools_allowlist": s.tools_allowlist or None,
            })
        return agents

