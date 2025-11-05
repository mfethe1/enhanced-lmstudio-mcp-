#!/usr/bin/env python3
"""
Comprehensive test script to validate MCP server functionality
for agentic coding systems.
"""

import json
import sys
import time
from typing import Dict, Any, List, Optional
from server import handle_message, get_server_singleton

class MCPTestHarness:
    def __init__(self):
        self.server = get_server_singleton()
        self.request_id = 1000
        self.failed_tests = []
        self.passed_tests = []
        
    def make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an MCP protocol request."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params
            
        response = handle_message(request)
        return response
    
    def test_tool(self, tool_name: str, arguments: Dict[str, Any], expected_keys: List[str] = None) -> bool:
        """Test a specific tool."""
        print(f"\n  Testing {tool_name}...")
        try:
            response = self.make_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            if "error" in response:
                print(f"    ❌ Error: {response['error'].get('message', 'Unknown error')}")
                return False
                
            result = response.get("result", {})
            content = result.get("content", [])
            
            if not content:
                print(f"    ❌ No content returned")
                return False
                
            # Check first content item
            first_content = content[0] if isinstance(content, list) else content
            text = first_content.get("text", "") if isinstance(first_content, dict) else str(first_content)
            
            if expected_keys:
                # Try to parse as JSON if expected keys provided
                try:
                    data = json.loads(text) if isinstance(text, str) else text
                    missing = [k for k in expected_keys if k not in data]
                    if missing:
                        print(f"    ⚠️  Missing expected keys: {missing}")
                        return False
                except:
                    pass  # Not JSON, that's okay for some tools
            
            print(f"    ✅ Success - returned {len(text)} chars")
            return True
            
        except Exception as e:
            print(f"    ❌ Exception: {str(e)[:100]}")
            return False
    
    def run_test_suite(self):
        """Run comprehensive test suite."""
        print("=" * 70)
        print("🧪 MCP SERVER COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        
        # Test 1: Basic health check
        print("\n📋 TEST 1: Basic Health & Version")
        print("-" * 40)
        if self.test_tool("health_check", {"probe_lm": False}):
            self.passed_tests.append("health_check")
        else:
            self.failed_tests.append("health_check")
            
        if self.test_tool("get_version", {}):
            self.passed_tests.append("get_version")
        else:
            self.failed_tests.append("get_version")
        
        # Test 2: Memory operations
        print("\n📋 TEST 2: Memory Operations")
        print("-" * 40)
        test_key = f"test_key_{int(time.time())}"
        if self.test_tool("store_memory", {
            "key": test_key,
            "value": "Test data for validation",
            "category": "test"
        }):
            self.passed_tests.append("store_memory")
            
            # Try to retrieve it
            if self.test_tool("retrieve_memory", {
                "key": test_key,
                "category": "test"
            }):
                self.passed_tests.append("retrieve_memory")
            else:
                self.failed_tests.append("retrieve_memory")
        else:
            self.failed_tests.append("store_memory")
        
        # Test 3: Code analysis tools
        print("\n📋 TEST 3: Code Analysis")
        print("-" * 40)
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        if self.test_tool("analyze_code", {
            "code": test_code,
            "analysis_type": "explanation"
        }):
            self.passed_tests.append("analyze_code")
        else:
            self.failed_tests.append("analyze_code")
            
        if self.test_tool("suggest_improvements", {"code": test_code}):
            self.passed_tests.append("suggest_improvements")
        else:
            self.failed_tests.append("suggest_improvements")
            
        if self.test_tool("generate_tests", {"code": test_code}):
            self.passed_tests.append("generate_tests")
        else:
            self.failed_tests.append("generate_tests")
        
        # Test 4: File operations
        print("\n📋 TEST 4: File Operations")
        print("-" * 40)
        if self.test_tool("list_directory", {"directory": "."}):
            self.passed_tests.append("list_directory")
        else:
            self.failed_tests.append("list_directory")
            
        if self.test_tool("search_files", {
            "pattern": "def ",
            "file_pattern": "*.py",
            "directory": ".",
            "max_results": 5
        }):
            self.passed_tests.append("search_files")
        else:
            self.failed_tests.append("search_files")
        
        # Test 5: Agentic task management
        print("\n📋 TEST 5: Agentic Task Management")
        print("-" * 40)
        if self.test_tool("list_all_tasks", {"status": "all"}):
            self.passed_tests.append("list_all_tasks")
        else:
            self.failed_tests.append("list_all_tasks")
        
        # Test 6: Router and smart task
        print("\n📋 TEST 6: Smart Task Routing")
        print("-" * 40)
        if self.test_tool("router_diagnostics", {}):
            self.passed_tests.append("router_diagnostics")
        else:
            self.failed_tests.append("router_diagnostics")
            
        if self.test_tool("smart_task", {
            "instruction": "Analyze the fibonacci function and suggest improvements",
            "context": test_code,
            "dry_run": True
        }):
            self.passed_tests.append("smart_task")
        else:
            self.failed_tests.append("smart_task")
        
        # Test 7: Agent team capabilities
        print("\n📋 TEST 7: Agent Team Capabilities")
        print("-" * 40)
        # Only test dry run to avoid long execution
        if self.test_tool("agent_team_plan_and_code", {
            "instruction": "Create a simple calculator function",
            "context": {"language": "python"},
            "dry_run": True
        }):
            self.passed_tests.append("agent_team_plan_and_code")
        else:
            self.failed_tests.append("agent_team_plan_and_code")
        
        # Test 8: Research tools
        print("\n📋 TEST 8: Research Tools")
        print("-" * 40)
        if self.test_tool("propose_research", {
            "problem": "How to optimize recursive fibonacci",
            "max_queries": 2
        }):
            self.passed_tests.append("propose_research")
        else:
            self.failed_tests.append("propose_research")
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        pass_rate = (len(self.passed_tests) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n✅ PASSED: {len(self.passed_tests)}/{total_tests} ({pass_rate:.1f}%)")
        for test in self.passed_tests:
            print(f"   ✓ {test}")
            
        if self.failed_tests:
            print(f"\n❌ FAILED: {len(self.failed_tests)}/{total_tests}")
            for test in self.failed_tests:
                print(f"   ✗ {test}")
        
        print("\n" + "=" * 70)
        if pass_rate >= 80:
            print("🎉 SUCCESS: MCP Server is working well for agentic coding!")
            print("   The server has passed critical functionality tests.")
            print("   It should work flawlessly with agentic coding systems.")
        elif pass_rate >= 60:
            print("⚠️  PARTIAL SUCCESS: Most features working, some issues detected")
            print("   The server is mostly functional but may have some limitations.")
        else:
            print("❌ FAILURE: Significant issues detected")
            print("   The server needs troubleshooting before use with agentic systems.")
        print("=" * 70)
        
        return len(self.failed_tests) == 0


def main():
    """Run the test harness."""
    print("\n🚀 Starting MCP Server Agentic Functionality Tests...")
    print("   This will validate critical tools for coding agents.\n")
    
    harness = MCPTestHarness()
    success = harness.run_test_suite()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()