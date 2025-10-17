"""
Integration tests for file locking MCP tools.

Tests:
1. Tool registration
2. Tool execution (acquire, release, stats)
3. Manager lifecycle
"""

import asyncio
import json
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import EnhancedLMStudioMCPServer
from handlers import agent_teams
from core.file_locking import get_file_lock_manager


def test_tool_registration():
    """Test that file locking tools are registered"""
    print("\n" + "="*60)
    print("TEST 1: Tool Registration")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Get all registered tools
        tools = server.registry.list_tools()
        print(f"\nTotal tools registered: {len(tools)}")
        
        # Check for file locking tools
        file_lock_tools = [t for t in tools if 'file_lock' in t.lower() or 'lock' in t.lower()]
        print(f"\nFile locking tools found: {len(file_lock_tools)}")
        for tool in sorted(file_lock_tools):
            print(f"  - {tool}")
        
        # Verify specific tools
        expected_tools = [
            "acquire_file_lock",
            "release_file_lock",
            "get_file_lock_stats"
        ]
        
        for tool in expected_tools:
            if tool in tools:
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


def test_tool_execution():
    """Test executing file locking tools"""
    print("\n" + "="*60)
    print("TEST 2: Tool Execution")
    print("="*60)
    
    try:
        server = EnhancedLMStudioMCPServer()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write("Test content")
        temp_file.close()
        temp_path = temp_file.name
        
        print(f"\nCreated temp file: {temp_path}")
        
        try:
            # Test 1: Acquire lock
            print("\n--- Test 1: Acquire Lock ---")
            result = agent_teams.handle_acquire_file_lock(
                {
                    "file_path": temp_path,
                    "owner_id": "test-agent-1",
                    "timeout_seconds": 60
                },
                server
            )
            result_data = json.loads(result)
            print(f"Result: {json.dumps(result_data, indent=2)}")
            
            if result_data.get("status") == "success":
                print("[OK] Lock acquired successfully")
                lock_id = result_data.get("lock_id")
            else:
                print(f"[FAIL] Lock acquisition failed: {result_data}")
                return False
            
            # Test 2: Get stats
            print("\n--- Test 2: Get Stats ---")
            stats_result = agent_teams.handle_get_file_lock_stats({}, server)
            print(f"Stats:\n{stats_result}")

            if "Active Locks**: 1" in stats_result or "Active Locks: 1" in stats_result:
                print("[OK] Stats show 1 active lock")
            else:
                print("[FAIL] Stats don't show active lock")
                return False
            
            # Test 3: Try to acquire same lock (should fail or wait)
            print("\n--- Test 3: Try to Acquire Same Lock (should conflict) ---")
            result2 = agent_teams.handle_acquire_file_lock(
                {
                    "file_path": temp_path,
                    "owner_id": "test-agent-2",
                    "timeout_seconds": 1,
                    "wait": False
                },
                server
            )
            result2_data = json.loads(result2)
            print(f"Result: {json.dumps(result2_data, indent=2)}")
            
            if result2_data.get("status") in ["failed", "timeout"]:
                print("[OK] Second lock attempt failed as expected")
            else:
                print(f"[FAIL] Second lock should have failed: {result2_data}")
                return False
            
            # Test 4: Release lock
            print("\n--- Test 4: Release Lock ---")
            release_result = agent_teams.handle_release_file_lock(
                {
                    "file_path": temp_path,
                    "owner_id": "test-agent-1"
                },
                server
            )
            release_data = json.loads(release_result)
            print(f"Result: {json.dumps(release_data, indent=2)}")
            
            if release_data.get("status") == "success":
                print("[OK] Lock released successfully")
            else:
                print(f"[FAIL] Lock release failed: {release_data}")
                return False
            
            # Test 5: Verify lock is released
            print("\n--- Test 5: Verify Lock Released ---")
            stats_result2 = agent_teams.handle_get_file_lock_stats({}, server)
            print(f"Stats:\n{stats_result2}")

            if "Active Locks**: 0" in stats_result2 or "Active Locks: 0" in stats_result2 or "No active locks" in stats_result2:
                print("[OK] Stats show 0 active locks")
            else:
                print("[FAIL] Stats still show active locks")
                return False
            
            print("\n[PASSED] Tool execution test")
            return True
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"\nCleaned up temp file: {temp_path}")
        
    except Exception as e:
        print(f"\n[FAILED] Tool execution test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_manager_lifecycle_async():
    """Test file lock manager start/stop (async version)"""
    try:
        # Create a fresh manager instance for this test
        from core.file_locking import FileLockManager
        manager = FileLockManager()

        # Test start
        print("\n--- Test 1: Start Manager ---")
        await manager.start()

        if manager._running:
            print("[OK] Manager started")
        else:
            print("[FAIL] Manager not running")
            return False

        # Test stop
        print("\n--- Test 2: Stop Manager ---")
        await manager.stop()

        if not manager._running:
            print("[OK] Manager stopped")
        else:
            print("[FAIL] Manager still running")
            return False

        # Verify all locks released
        stats = manager.get_stats()
        if stats["active_locks"] == 0:
            print("[OK] All locks released on stop")
        else:
            print(f"[FAIL] {stats['active_locks']} locks still active")
            return False

        print("\n[PASSED] Manager lifecycle test")
        return True

    except Exception as e:
        print(f"\n[FAILED] Manager lifecycle test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manager_lifecycle():
    """Test file lock manager start/stop"""
    print("\n" + "="*60)
    print("TEST 3: Manager Lifecycle")
    print("="*60)

    return asyncio.run(test_manager_lifecycle_async())


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("FILE LOCKING INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Tool Registration", test_tool_registration()))
    results.append(("Tool Execution", test_tool_execution()))
    results.append(("Manager Lifecycle", test_manager_lifecycle()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status} {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All integration tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

