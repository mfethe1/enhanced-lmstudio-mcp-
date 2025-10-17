"""
Integration tests for workflow patterns (Phase 2 Priority 3)

Tests MCP tool handlers for workflow execution.

Author: Jarvis MCP Team
Date: 2025-01-16
"""

import asyncio
import json
import sys

# Add parent directory to path
sys.path.insert(0, ".")

from server import EnhancedLMStudioMCPServer
from handlers import agent_teams


def test_tool_registration():
    """Test that workflow tools are registered"""
    print("\n" + "="*60)
    print("TEST 1: Tool Registration")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Get all registered tools
        all_tools = server.registry.list_tools()
        print(f"\nTotal tools registered: {len(all_tools)}")
        
        # Check for workflow tools
        workflow_tools = [
            "execute_parallel_workflow",
            "execute_sequential_workflow",
            "execute_evaluator_optimizer_workflow"
        ]
        
        found_tools = [t for t in workflow_tools if t in all_tools]
        print(f"\nWorkflow tools found: {len(found_tools)}/{len(workflow_tools)}")
        
        for tool in workflow_tools:
            if tool in all_tools:
                print(f"  [OK] {tool} registered")
            else:
                print(f"  [FAIL] {tool} NOT registered")
                return False
        
        print("\n[PASSED] Tool registration test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Tool registration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_workflow_execution():
    """Test parallel workflow execution via MCP tool"""
    print("\n" + "="*60)
    print("TEST 2: Parallel Workflow Execution")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create test tasks
        tasks = [
            {"name": "task-1", "function_name": "test_func", "args": [1], "kwargs": {}},
            {"name": "task-2", "function_name": "test_func", "args": [2], "kwargs": {}},
            {"name": "task-3", "function_name": "test_func", "args": [3], "kwargs": {}}
        ]
        
        arguments = {
            "tasks": tasks,
            "max_concurrency": 3,
            "error_strategy": "continue",
            "aggregation_strategy": "all"
        }
        
        print("\n--- Test 1: Execute Parallel Workflow ---")
        result_str = agent_teams.handle_execute_parallel_workflow(arguments, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Parallel workflow executed successfully")
        else:
            print(f"[FAIL] Parallel workflow failed: {result.get('error')}")
            return False
        
        # Check results
        results = result.get("results", [])
        if len(results) == 3:
            print(f"[OK] Got {len(results)} results")
        else:
            print(f"[FAIL] Expected 3 results, got {len(results)}")
            return False
        
        # Check stats
        stats = result.get("stats", {})
        if stats.get("successful") == 3:
            print(f"[OK] All 3 tasks successful")
        else:
            print(f"[FAIL] Expected 3 successful tasks, got {stats.get('successful')}")
            return False
        
        print("\n[PASSED] Parallel workflow execution test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Parallel workflow execution test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential_workflow_execution():
    """Test sequential workflow execution via MCP tool"""
    print("\n" + "="*60)
    print("TEST 3: Sequential Workflow Execution")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create test tasks with dependencies
        tasks = [
            {"name": "task-1", "function_name": "test_func", "args": [1], "kwargs": {}, "dependencies": []},
            {"name": "task-2", "function_name": "test_func", "args": [2], "kwargs": {}, "dependencies": ["task-0"]},
            {"name": "task-3", "function_name": "test_func", "args": [3], "kwargs": {}, "dependencies": ["task-1"]}
        ]
        
        arguments = {
            "tasks": tasks,
            "error_strategy": "continue"
        }
        
        print("\n--- Test 1: Execute Sequential Workflow ---")
        result_str = agent_teams.handle_execute_sequential_workflow(arguments, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Sequential workflow executed successfully")
        else:
            print(f"[FAIL] Sequential workflow failed: {result.get('error')}")
            return False
        
        # Check results
        results = result.get("results", [])
        if len(results) == 3:
            print(f"[OK] Got {len(results)} results")
        else:
            print(f"[FAIL] Expected 3 results, got {len(results)}")
            return False
        
        print("\n[PASSED] Sequential workflow execution test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Sequential workflow execution test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator_optimizer_workflow_execution():
    """Test evaluator-optimizer workflow execution via MCP tool"""
    print("\n" + "="*60)
    print("TEST 4: Evaluator-Optimizer Workflow Execution")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        arguments = {
            "initial_solution": 50,
            "score_threshold": 0.95,
            "max_iterations": 10,
            "no_improvement_limit": 3
        }
        
        print("\n--- Test 1: Execute Evaluator-Optimizer Workflow ---")
        result_str = agent_teams.handle_execute_evaluator_optimizer_workflow(arguments, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Evaluator-optimizer workflow executed successfully")
        else:
            print(f"[FAIL] Evaluator-optimizer workflow failed: {result.get('error')}")
            return False
        
        # Check result
        workflow_result = result.get("result", {})
        if workflow_result.get("iterations") > 0:
            print(f"[OK] Completed {workflow_result.get('iterations')} iterations")
        else:
            print(f"[FAIL] Expected > 0 iterations")
            return False
        
        # Check history
        history = result.get("history", [])
        if len(history) > 0:
            print(f"[OK] Got {len(history)} history entries")
        else:
            print(f"[FAIL] Expected history entries")
            return False
        
        print("\n[PASSED] Evaluator-optimizer workflow execution test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Evaluator-optimizer workflow execution test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("WORKFLOW PATTERNS INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_tool_registration,
        test_parallel_workflow_execution,
        test_sequential_workflow_execution,
        test_evaluator_optimizer_workflow_execution
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status} {test.__name__}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

