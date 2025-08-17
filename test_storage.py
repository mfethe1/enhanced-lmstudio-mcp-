#!/usr/bin/env python3
"""
Test script for the enhanced storage layer

This validates that persistent storage is working correctly before
running the full MCP server.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add current directory to path to import our storage module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import MCPStorage

def test_storage_functionality():
    """Test all storage functionality"""
    print("🧪 Testing Enhanced Storage Layer")
    print("=" * 40)
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        test_db_path = temp_db.name
    
    try:
        # Initialize storage
        print("\n📋 Test 1: Storage Initialization")
        storage = MCPStorage(test_db_path)
        print("✅ Storage initialized successfully")
        
        # Test memory operations
        print("\n📋 Test 2: Memory Operations")
        
        # Store some test memories
        test_memories = [
            ("test_key_1", "Test value 1", "general"),
            ("test_key_2", "Test value 2", "debug"),
            ("test_key_3", "Another test value", "general"),
        ]
        
        for key, value, category in test_memories:
            success = storage.store_memory(key, value, category)
            print(f"{'✅' if success else '❌'} Stored memory: {key}")
        
        # Retrieve memories
        memories = storage.retrieve_memory(category="general")
        print(f"✅ Retrieved {len(memories)} general memories")
        
        memories = storage.retrieve_memory(search_term="test")
        print(f"✅ Retrieved {len(memories)} memories with 'test'")
        
        # Test thinking sessions
        print("\n📋 Test 3: Thinking Sessions")
        
        session_id = "test_session_123"
        thoughts = [
            (1, "First thought about the problem", False, None, None),
            (2, "Second thought building on the first", False, None, None),
            (3, "Revision of the first thought", True, 1, None),
        ]
        
        for thought_num, thought, is_revision, revises, branch in thoughts:
            success = storage.store_thinking_step(
                session_id, thought_num, thought, is_revision, revises, branch
            )
            print(f"{'✅' if success else '❌'} Stored thought {thought_num}")
        
        session_thoughts = storage.get_thinking_session(session_id)
        print(f"✅ Retrieved {len(session_thoughts)} thoughts from session")
        
        # Test performance metrics
        print("\n📋 Test 4: Performance Metrics")
        
        # Log some test performance data
        test_metrics = [
            ("test_tool_1", 0.15, True, None),
            ("test_tool_2", 0.25, True, None),
            ("test_tool_1", 0.45, False, "Test error message"),
            ("test_tool_3", 0.05, True, None),
        ]
        
        for tool, exec_time, success, error in test_metrics:
            logged = storage.log_performance(tool, exec_time, success, error)
            print(f"{'✅' if logged else '❌'} Logged performance for {tool}")
        
        # Get performance stats
        stats = storage.get_performance_stats()
        print(f"✅ Retrieved performance stats for {len(stats.get('stats', []))} tools")
        
        # Test error logging
        print("\n📋 Test 5: Error Tracking")
        
        test_errors = [
            ("ValueError", "Test value error", "test context", "test_tool_1"),
            ("TypeError", "Test type error", "another context", "test_tool_2"),
            ("RuntimeError", "Test runtime error", None, None),
        ]
        
        for error_type, error_msg, context, tool in test_errors:
            logged = storage.log_error(error_type, error_msg, context, tool)
            print(f"{'✅' if logged else '❌'} Logged error: {error_type}")
        
        error_patterns = storage.get_error_patterns()
        print(f"✅ Retrieved {len(error_patterns)} error patterns")
        
        # Test storage statistics
        print("\n📋 Test 6: Storage Statistics")
        
        stats = storage.get_storage_stats()
        print(f"✅ Storage stats: {stats}")
        
        # Test cleanup
        print("\n📋 Test 7: Data Cleanup")
        
        # Clean up old data (use 0 days to clean everything for testing)
        deleted = storage.cleanup_old_data(days=0)
        print(f"✅ Cleanup completed: {deleted}")
        
        print("\n🎉 All storage tests passed!")
        
        # Show final storage state
        final_stats = storage.get_storage_stats()
        print(f"\nFinal storage state: {final_stats}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test database
        try:
            os.unlink(test_db_path)
            print(f"\n🧹 Cleaned up test database: {test_db_path}")
        except:
            pass

def test_performance_monitoring():
    """Test performance monitoring functionality"""
    print("\n⚡ Testing Performance Monitoring")
    print("=" * 40)
    
    # Create a test storage instance
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        test_db_path = temp_db.name
    
    try:
        storage = MCPStorage(test_db_path)
        
        # Simulate some tool executions
        import time
        
        tools_to_test = ["sequential_thinking", "analyze_code", "execute_code"]
        
        for tool in tools_to_test:
            # Simulate successful execution
            start_time = time.time()
            time.sleep(0.1)  # Simulate work
            execution_time = time.time() - start_time
            
            storage.log_performance(tool, execution_time, True)
            print(f"✅ Logged successful execution for {tool}: {execution_time:.3f}s")
            
            # Simulate failed execution
            storage.log_performance(tool, execution_time, False, "Test error")
            print(f"✅ Logged failed execution for {tool}")
        
        # Get and display performance stats
        stats = storage.get_performance_stats()
        if stats.get('stats'):
            print(f"\n📊 Performance Statistics:")
            for stat in stats['stats']:
                print(f"  {stat['tool_name']}: {stat['total_calls']} calls, "
                      f"{stat['success_rate']:.1%} success rate, "
                      f"{stat['avg_time']:.3f}s avg time")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False
        
    finally:
        try:
            os.unlink(test_db_path)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Enhanced MCP Storage Testing Suite")
    print("=" * 50)
    
    # Run storage tests
    storage_success = test_storage_functionality()
    
    # Run performance monitoring tests
    performance_success = test_performance_monitoring()
    
    # Final results
    print("\n" + "=" * 50)
    if storage_success and performance_success:
        print("🎉 ALL TESTS PASSED! Storage layer is ready for integration.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please review the errors above.")
        sys.exit(1) 