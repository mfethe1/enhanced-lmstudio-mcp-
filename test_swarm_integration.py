"""
Integration tests for Swarm Pattern (Phase 2 Priority 4)

Tests MCP tool handlers for swarm execution.

Author: Jarvis MCP Team
Date: 2025-01-16
"""

import asyncio
import json
import sys
import time

# Add parent directory to path
sys.path.insert(0, ".")

from server import EnhancedLMStudioMCPServer
from handlers import swarm


def test_tool_registration():
    """Test that swarm tools are registered"""
    print("\n" + "="*60)
    print("TEST 1: Tool Registration")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Get all registered tools
        all_tools = server.registry.list_tools()
        print(f"\nTotal tools registered: {len(all_tools)}")
        
        # Check for swarm tools
        swarm_tools = [
            "create_swarm",
            "execute_swarm_task",
            "get_swarm_status",
            "visualize_swarm"
        ]
        
        found_tools = [t for t in swarm_tools if t in all_tools]
        print(f"\nSwarm tools found: {len(found_tools)}/{len(swarm_tools)}")
        
        for tool in swarm_tools:
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


def test_create_swarm():
    """Test swarm creation via MCP tool"""
    print("\n" + "="*60)
    print("TEST 2: Create Swarm")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create swarm with 3 agents
        agents = [
            {"agent_id": "planner-1", "specialization": "planner", "max_tasks": 5},
            {"agent_id": "coder-1", "specialization": "coder", "max_tasks": 5},
            {"agent_id": "reviewer-1", "specialization": "reviewer", "max_tasks": 5}
        ]
        
        arguments = {
            "agents": agents,
            "swarm_id": "test-swarm-1"
        }
        
        print("\n--- Test 1: Create Swarm ---")
        result_str = swarm.handle_create_swarm(arguments, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Swarm created successfully")
        else:
            print(f"[FAIL] Swarm creation failed: {result.get('error')}")
            return False
        
        # Check swarm ID
        if result.get("swarm_id") == "test-swarm-1":
            print(f"[OK] Swarm ID: {result.get('swarm_id')}")
        else:
            print(f"[FAIL] Unexpected swarm ID: {result.get('swarm_id')}")
            return False
        
        # Check agents
        if result.get("total_agents") == 3:
            print(f"[OK] Created {result.get('total_agents')} agents")
        else:
            print(f"[FAIL] Expected 3 agents, got {result.get('total_agents')}")
            return False
        
        print("\n[PASSED] Create swarm test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Create swarm test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execute_swarm_task():
    """Test task execution via swarm"""
    print("\n" + "="*60)
    print("TEST 3: Execute Swarm Task")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create swarm first
        agents = [
            {"agent_id": "planner-1", "specialization": "planner", "max_tasks": 5},
            {"agent_id": "coder-1", "specialization": "coder", "max_tasks": 5}
        ]
        
        create_args = {
            "agents": agents,
            "swarm_id": "test-swarm-2"
        }
        
        create_result_str = swarm.handle_create_swarm(create_args, server)
        create_result = json.loads(create_result_str)
        
        if create_result.get("status") != "success":
            print(f"[FAIL] Failed to create swarm: {create_result.get('error')}")
            return False
        
        print(f"[OK] Created swarm: {create_result.get('swarm_id')}")
        
        # Execute task
        task = {
            "task_id": "task-1",
            "description": "Plan a new feature",
            "priority": "high"
        }
        
        execute_args = {
            "swarm_id": "test-swarm-2",
            "task": task,
            "specialization": "planner",
            "timeout": 30.0
        }
        
        print("\n--- Test 1: Execute Task ---")
        result_str = swarm.handle_execute_swarm_task(execute_args, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Task executed successfully")
        else:
            print(f"[FAIL] Task execution failed: {result.get('error')}")
            return False
        
        # Check result
        task_result = result.get("result", {})
        if task_result.get("status") == "success":
            print(f"[OK] Task completed by agent: {task_result.get('agent_id')}")
        else:
            print(f"[FAIL] Task failed")
            return False
        
        print("\n[PASSED] Execute swarm task test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Execute swarm task test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_swarm_status():
    """Test swarm status retrieval"""
    print("\n" + "="*60)
    print("TEST 4: Get Swarm Status")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create swarm first
        agents = [
            {"agent_id": "agent-1", "specialization": "generalist", "max_tasks": 5},
            {"agent_id": "agent-2", "specialization": "generalist", "max_tasks": 5}
        ]
        
        create_args = {
            "agents": agents,
            "swarm_id": "test-swarm-3"
        }
        
        create_result_str = swarm.handle_create_swarm(create_args, server)
        create_result = json.loads(create_result_str)
        
        if create_result.get("status") != "success":
            print(f"[FAIL] Failed to create swarm: {create_result.get('error')}")
            return False
        
        print(f"[OK] Created swarm: {create_result.get('swarm_id')}")
        
        # Get status
        status_args = {
            "swarm_id": "test-swarm-3"
        }
        
        print("\n--- Test 1: Get Status ---")
        result_str = swarm.handle_get_swarm_status(status_args, server)
        result = json.loads(result_str)
        
        print(f"Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print("[OK] Status retrieved successfully")
        else:
            print(f"[FAIL] Status retrieval failed: {result.get('error')}")
            return False
        
        # Check status fields
        if result.get("active_agents") == 2:
            print(f"[OK] Active agents: {result.get('active_agents')}")
        else:
            print(f"[FAIL] Expected 2 active agents, got {result.get('active_agents')}")
            return False
        
        if "agent_status" in result:
            print(f"[OK] Agent status available")
        else:
            print(f"[FAIL] Agent status missing")
            return False
        
        print("\n[PASSED] Get swarm status test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Get swarm status test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualize_swarm():
    """Test swarm visualization"""
    print("\n" + "="*60)
    print("TEST 5: Visualize Swarm")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create swarm first
        agents = [
            {"agent_id": "agent-1", "specialization": "planner", "max_tasks": 5}
        ]
        
        create_args = {
            "agents": agents,
            "swarm_id": "test-swarm-4"
        }
        
        create_result_str = swarm.handle_create_swarm(create_args, server)
        create_result = json.loads(create_result_str)
        
        if create_result.get("status") != "success":
            print(f"[FAIL] Failed to create swarm: {create_result.get('error')}")
            return False
        
        print(f"[OK] Created swarm: {create_result.get('swarm_id')}")
        
        # Test JSON format
        print("\n--- Test 1: JSON Format ---")
        viz_args = {
            "swarm_id": "test-swarm-4",
            "format": "json"
        }
        
        result_str = swarm.handle_visualize_swarm(viz_args, server)
        result = json.loads(result_str)
        
        if result.get("status") == "success" and result.get("format") == "json":
            print("[OK] JSON visualization generated")
        else:
            print(f"[FAIL] JSON visualization failed")
            return False
        
        # Test ASCII format
        print("\n--- Test 2: ASCII Format ---")
        viz_args["format"] = "ascii"
        
        result_str = swarm.handle_visualize_swarm(viz_args, server)
        result = json.loads(result_str)
        
        if result.get("status") == "success" and result.get("format") == "ascii":
            print("[OK] ASCII visualization generated")
            print(f"\n{result.get('visualization')}")
        else:
            print(f"[FAIL] ASCII visualization failed")
            return False
        
        print("\n[PASSED] Visualize swarm test")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Visualize swarm test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("SWARM PATTERN INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_tool_registration,
        test_create_swarm,
        test_execute_swarm_task,
        test_get_swarm_status,
        test_visualize_swarm
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

