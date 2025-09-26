"""
Workflow templates for common research patterns.
"""
from __future__ import annotations
from typing import Dict, Any, List

WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "literature_to_experiment": {
        "stages": [
            {"name": "literature_review", "tools": ["smart_task", "web_search", "deep_research", "memory_consolidate"], "owner": "technical_writer"},
            {"name": "hypothesis", "tools": ["smart_task", "memory_retrieve_semantic", "artifact_create"], "owner": "data_scientist"},
            {"name": "experiment_design", "tools": ["smart_plan_execute", "smart_task"], "owner": "wet_lab_chemist"},
            {"name": "analysis", "tools": ["smart_task", "memory_retrieve_semantic", "artifact_create"], "owner": "computational_biologist"},
        ]
    },
    "software_feature": {
        "stages": [
            {"name": "triage", "tools": ["smart_task", "code_hotspots", "import_graph"], "owner": "research_coordinator"},
            {"name": "design", "tools": ["smart_task", "analyze_code_context"], "owner": "backend_developer"},
            {"name": "implementation", "tools": ["smart_plan_execute", "smart_task", "analyze_code", "generate_tests", "execute_code_sandbox", "debug_interactive"], "owner": "frontend_developer"},
            {"name": "validation", "tools": ["smart_task", "generate_tests", "run_tests", "router_battery", "get_task_status"], "owner": "quality_assurance"},
        ]
    }
}

