"""
Integration test for ephemeral agent MCP tools.

Tests that the tools are properly registered and functional.
"""

import asyncio
import json
import sys

# Add current directory to path
sys.path.insert(0, '.')

from server import EnhancedLMStudioMCPServer


def test_tool_registration():
    """Test that ephemeral agent tools are registered"""
    print("=" * 80)
    print("TEST: Ephemeral Agent Tool Registration")
    print("=" * 80)
    
    # Create server instance
    server = EnhancedLMStudioMCPServer()
    
    # Get all registered tools
    tools = server.registry.list_tools()
    print(f"\nTotal tools registered: {len(tools)}")
    
    # Check for ephemeral agent tools
    expected_tools = [
        "request_ephemeral_agent",
        "release_ephemeral_agent",
        "get_ephemeral_agent_stats"
    ]
    
    found_tools = []
    missing_tools = []
    
    for tool_name in expected_tools:
        if tool_name in tools:
            found_tools.append(tool_name)
            print(f"  [OK] {tool_name} - REGISTERED")
        else:
            missing_tools.append(tool_name)
            print(f"  [FAIL] {tool_name} - MISSING")

    print(f"\nResult: {len(found_tools)}/{len(expected_tools)} tools registered")

    if missing_tools:
        print(f"\n[FAILED]: Missing tools: {', '.join(missing_tools)}")
        return False
    else:
        print(f"\n[PASSED]: All ephemeral agent tools registered")
        return True


def test_tool_execution():
    """Test that ephemeral agent tools can be executed"""
    print("\n" + "=" * 80)
    print("TEST: Ephemeral Agent Tool Execution")
    print("=" * 80)
    
    # Create server instance
    server = EnhancedLMStudioMCPServer()
    
    # Test 1: Get stats (should work even with no agents)
    print("\n1. Testing get_ephemeral_agent_stats...")
    try:
        from handlers import agent_teams
        result = agent_teams.handle_get_ephemeral_agent_stats({}, server)
        print(f"  [OK] get_ephemeral_agent_stats executed successfully")
        print(f"  Result preview: {result[:200]}...")
    except Exception as e:
        print(f"  [FAIL] get_ephemeral_agent_stats failed: {e}")
        return False

    # Test 2: Request agent
    print("\n2. Testing request_ephemeral_agent...")
    try:
        result = agent_teams.handle_request_ephemeral_agent({
            "role": "test",
            "task_description": "Integration test task",
            "priority": 0
        }, server)
        result_json = json.loads(result)
        print(f"  [OK] request_ephemeral_agent executed successfully")
        print(f"  Status: {result_json.get('status')}")
        print(f"  ID: {result_json.get('id')}")

        agent_id = result_json.get('id')

        # Test 3: Release agent
        if agent_id and agent_id.startswith("agent-"):
            print("\n3. Testing release_ephemeral_agent...")
            try:
                result = agent_teams.handle_release_ephemeral_agent({
                    "agent_id": agent_id
                }, server)
                result_json = json.loads(result)
                print(f"  [OK] release_ephemeral_agent executed successfully")
                print(f"  Status: {result_json.get('status')}")
            except Exception as e:
                print(f"  [FAIL] release_ephemeral_agent failed: {e}")
                return False

    except Exception as e:
        print(f"  [FAIL] request_ephemeral_agent failed: {e}")
        return False

    print(f"\n[PASSED]: All ephemeral agent tools executed successfully")
    return True


def test_manager_lifecycle():
    """Test ephemeral agent manager lifecycle"""
    print("\n" + "=" * 80)
    print("TEST: Ephemeral Agent Manager Lifecycle")
    print("=" * 80)
    
    from core.ephemeral_agents import get_ephemeral_agent_manager
    
    # Get manager
    manager = get_ephemeral_agent_manager()
    print(f"\n1. Manager created: {manager}")
    print(f"   Max concurrent: {manager.max_concurrent}")
    print(f"   Max queue size: {manager.max_queue_size}")
    
    # Start manager
    print("\n2. Starting manager...")
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(manager.start())
    print(f"   [OK] Manager started")
    print(f"   Running: {manager._running}")

    # Get stats
    print("\n3. Getting stats...")
    stats = manager.get_stats()
    print(f"   Active agents: {stats['active_agents']}")
    print(f"   Queue size: {stats['queue_size']}")
    print(f"   Total created: {stats['total_created']}")

    # Stop manager
    print("\n4. Stopping manager...")
    loop.run_until_complete(manager.stop())
    print(f"   [OK] Manager stopped")
    print(f"   Running: {manager._running}")

    print(f"\n[PASSED]: Manager lifecycle working correctly")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EPHEMERAL AGENT INTEGRATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Tool Registration", test_tool_registration()))
    results.append(("Tool Execution", test_tool_execution()))
    results.append(("Manager Lifecycle", test_manager_lifecycle()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED - Integration successful!")
        sys.exit(0)
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed")
        sys.exit(1)

