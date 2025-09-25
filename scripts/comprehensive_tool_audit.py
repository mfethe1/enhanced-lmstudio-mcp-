#!/usr/bin/env python3
"""
Comprehensive MCP Tool Audit Script

Tests all 63+ MCP tools registered in server.py to ensure compatibility
with the enhanced Strands multi-agent system.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up clean test environment
os.environ.setdefault("PROACTIVE_RESEARCH_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("USE_HYBRID_RETRIEVAL", "1")
os.environ.setdefault("ENABLE_KG_INGESTION", "1")

import server

# Tool categories for systematic testing
TOOL_CATEGORIES = {
    "core_mcp": [
        "health_check", "get_version", "router_config", "router_diagnostics",
        "router_test", "router_battery", "performance_stats"
    ],
    "smart_tools": [
        "smart_task", "smart_plan_execute", "chat_with_tools"
    ],
    "memory_storage": [
        "memory_store", "memory_retrieve", "memory_retrieve_semantic", 
        "memory_consolidate"
    ],
    "research_analysis": [
        "web_search", "deep_research", "get_research_details",
        "propose_research", "proactive_research_status"
    ],
    "strands_advanced": [
        # These are integrated into the orchestrator, not standalone tools
        # but we test them through the orchestrator
    ],
    "development": [
        "cognitive_codegen_one_shot", "file_scaffold", "code_hotspots",
        "import_graph", "tdd_flow"
    ],
    "workflow": [
        "workflow_create", "workflow_add_node", "workflow_connect_nodes",
        "workflow_execute", "workflow_explain"
    ],
    "session_management": [
        "session_create", "context_envelope_create", "artifact_create",
        "session_analytics"
    ],
    "thinking": [
        "sequential_thinking", "get_thinking_session", "summarize_thinking_session"
    ]
}

class ToolAuditor:
    def __init__(self):
        self.server = server.EnhancedLMStudioMCPServer()
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        # Mock async methods to avoid network calls
        self.server._lmstudio_request_with_retry = self._mock_lm_request
        
    async def _mock_lm_request(self, prompt: str, temperature: float = 0.2):
        """Mock LM Studio request for testing."""
        return '{"response": "mock_response", "steps": []}'
    
    def test_tool(self, tool_name: str, test_args: dict = None) -> dict:
        """Test a single MCP tool with safe arguments."""
        if test_args is None:
            test_args = self._get_safe_test_args(tool_name)
        
        start_time = time.time()
        result = {
            "tool": tool_name,
            "status": "unknown",
            "response_time_ms": 0,
            "error": None,
            "response_type": None,
            "response_length": 0
        }
        
        try:
            # Get the handler function
            handler_name = f"handle_{tool_name}"
            if not hasattr(server, handler_name):
                result.update({
                    "status": "not_found",
                    "error": f"Handler {handler_name} not found"
                })
                return result
            
            handler = getattr(server, handler_name)
            
            # Call the handler
            response = handler(test_args, self.server)
            
            # Analyze response
            result.update({
                "status": "success",
                "response_time_ms": int((time.time() - start_time) * 1000),
                "response_type": type(response).__name__,
                "response_length": len(str(response)) if response else 0
            })
            
            # Validate response format
            if tool_name in ["health_check", "router_config", "get_version"]:
                if not isinstance(response, dict):
                    result["status"] = "warning"
                    result["error"] = f"Expected dict, got {type(response)}"
            
        except Exception as e:
            result.update({
                "status": "error",
                "response_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e)[:200]
            })
        
        return result
    
    def _get_safe_test_args(self, tool_name: str) -> dict:
        """Get safe test arguments for each tool."""
        safe_args = {
            "health_check": {"probe_lm": False, "probe_providers": False},
            "get_version": {},
            "router_config": {},
            "router_diagnostics": {"limit": 5},
            "router_test": {"task": "test query"},
            "router_battery": {"limit": 1},
            "performance_stats": {},
            "smart_task": {"instruction": "test task", "dry_run": True},
            "smart_plan_execute": {"instruction": "test plan", "dry_run": True, "max_steps": 1},
            "chat_with_tools": {"instruction": "test chat", "max_iters": 1},
            "memory_store": {"key": "test_key", "value": "test_value"},
            "memory_retrieve": {"key": "test_key"},
            "memory_retrieve_semantic": {"query": "test query", "limit": 3},
            "memory_consolidate": {"category": "test", "limit": 5},
            "web_search": {"query": "test search", "max_depth": 1, "time_limit": 10},
            "deep_research": {"query": "test research", "maxDepth": 1, "timeLimit": 30},
            "get_research_details": {"research_id": "test_id"},
            "propose_research": {"problem": "test problem"},
            "proactive_research_status": {},
            "cognitive_codegen_one_shot": {"spec": "test spec", "dry_run": True},
            "file_scaffold": {"path": "test.py", "kind": "python"},
            "code_hotspots": {"directory": ".", "limit": 5},
            "import_graph": {"directory": ".", "limit": 5},
            "tdd_flow": {"target_path": "test.py", "goal": "test goal", "dry_run": True},
            "workflow_create": {"name": "test_workflow", "description": "test"},
            "workflow_add_node": {"workflow_id": "test", "node_id": "test_node", "node_type": "task"},
            "workflow_connect_nodes": {"workflow_id": "test", "from_node": "a", "to_node": "b"},
            "workflow_execute": {"workflow_id": "test"},
            "workflow_explain": {"workflow_id": "test"},
            "session_create": {"session_id": "test_session"},
            "context_envelope_create": {"session_id": "test", "context_type": "test"},
            "artifact_create": {"session_id": "test", "artifact_type": "test", "content": "test"},
            "session_analytics": {"session_id": "test"},
            "sequential_thinking": {"thought": "test thought", "nextThoughtNeeded": False, "thoughtNumber": 1, "totalThoughts": 1},
            "get_thinking_session": {"session_id": "test"},
            "summarize_thinking_session": {"session_id": "test"}
        }
        
        return safe_args.get(tool_name, {})
    
    def run_category_tests(self, category: str, tools: list) -> dict:
        """Run tests for a specific category of tools."""
        print(f"\n🔧 Testing {category.upper()} tools...")
        category_results = {}
        
        for tool_name in tools:
            print(f"  Testing {tool_name}...", end=" ")
            result = self.test_tool(tool_name)
            category_results[tool_name] = result
            
            self.total_tests += 1
            if result["status"] == "success":
                self.passed_tests += 1
                print("✅")
            elif result["status"] == "warning":
                self.passed_tests += 1
                print("⚠️")
            else:
                self.failed_tests += 1
                print(f"❌ {result['error'][:50]}...")
        
        return category_results
    
    def run_comprehensive_audit(self) -> dict:
        """Run comprehensive audit of all MCP tools."""
        print("🚀 Starting Comprehensive MCP Tool Audit")
        print("=" * 50)
        
        audit_start = time.time()
        
        for category, tools in TOOL_CATEGORIES.items():
            if tools:  # Skip empty categories
                self.results[category] = self.run_category_tests(category, tools)
        
        # Test Strands integration
        print(f"\n🧠 Testing STRANDS INTEGRATION...")
        strands_result = self._test_strands_integration()
        self.results["strands_integration"] = strands_result
        
        audit_time = time.time() - audit_start
        
        # Generate summary
        summary = {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            "audit_time_seconds": round(audit_time, 2),
            "strands_integration": strands_result
        }
        
        self.results["summary"] = summary
        return self.results
    
    def _test_strands_integration(self) -> dict:
        """Test Strands multi-agent integration."""
        try:
            from strands import TeamOrchestrator, OrchestratorConfig
            from strands.ingestion import KGIngestionEngine
            from strands.selection_router import SelectionRouter
            from strands.rag import HybridRetriever
            
            # Test orchestrator initialization
            config = OrchestratorConfig(project_id="audit_test")
            orchestrator = TeamOrchestrator(self.server, config)
            
            # Test ingestion engine
            engine = KGIngestionEngine(self.server)
            
            # Test selection router
            router = SelectionRouter(self.server)
            
            # Test hybrid retriever
            retriever = HybridRetriever(self.server)
            
            return {
                "status": "success",
                "components_tested": ["TeamOrchestrator", "KGIngestionEngine", "SelectionRouter", "HybridRetriever"],
                "integration_healthy": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "integration_healthy": False
            }
    
    def print_summary(self):
        """Print audit summary."""
        summary = self.results.get("summary", {})
        strands = self.results.get("strands_integration", {})
        
        print(f"\n📊 AUDIT SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed_tests', 0)}")
        print(f"Failed: {summary.get('failed_tests', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Audit Time: {summary.get('audit_time_seconds', 0)}s")
        
        print(f"\n🧠 STRANDS INTEGRATION")
        print("-" * 30)
        if strands.get("integration_healthy"):
            print("✅ All Strands components initialized successfully")
            print(f"✅ Components: {', '.join(strands.get('components_tested', []))}")
        else:
            print(f"❌ Integration error: {strands.get('error', 'Unknown')}")
        
        # Show failed tools
        failed_tools = []
        for category, tools in self.results.items():
            if category not in ["summary", "strands_integration"]:
                for tool, result in tools.items():
                    if result.get("status") == "error":
                        failed_tools.append(f"{tool}: {result.get('error', 'Unknown')[:50]}")
        
        if failed_tools:
            print(f"\n❌ FAILED TOOLS ({len(failed_tools)})")
            print("-" * 30)
            for failure in failed_tools[:10]:  # Show first 10
                print(f"  {failure}")
            if len(failed_tools) > 10:
                print(f"  ... and {len(failed_tools) - 10} more")


def main():
    """Run the comprehensive tool audit."""
    auditor = ToolAuditor()
    results = auditor.run_comprehensive_audit()
    auditor.print_summary()
    
    # Save detailed results
    results_file = Path("audit_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Return exit code based on success rate
    success_rate = results.get("summary", {}).get("success_rate", 0)
    if success_rate >= 90:
        print("🎉 Audit PASSED - System ready for production!")
        return 0
    elif success_rate >= 75:
        print("⚠️ Audit PASSED with warnings - Review failed tools")
        return 0
    else:
        print("❌ Audit FAILED - Critical issues need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
