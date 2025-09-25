"""
Workflow templates for common research patterns.
"""
from __future__ import annotations
from typing import Dict, Any, List

WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "literature_to_experiment": {
        "stages": [
            {"name": "literature_review", "tools": ["web_search", "firecrawl_search"], "owner": "technical_writer"},
            {"name": "hypothesis", "tools": ["smart_task"], "owner": "data_scientist"},
            {"name": "experiment_design", "tools": ["smart_plan_execute"], "owner": "wet_lab_chemist"},
            {"name": "analysis", "tools": ["smart_task", "memory_retrieve_semantic"], "owner": "computational_biologist"},
        ]
    },
    "software_feature": {
        "stages": [
            {"name": "triage", "tools": ["smart_task"], "owner": "research_coordinator"},
            {"name": "design", "tools": ["smart_task"], "owner": "backend_developer"},
            {"name": "implementation", "tools": ["smart_plan_execute"], "owner": "frontend_developer"},
            {"name": "validation", "tools": ["router_battery", "get_task_status"], "owner": "quality_assurance"},
        ]
    }
}

