#!/usr/bin/env python3
"""
Test runtime error cleanup and production configuration.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_proactive_research_disabled():
    """Test that proactive research can be disabled"""
    print("=" * 80)
    print("PHASE 2.1: Proactive Research Disable Test")
    print("=" * 80)
    
    # Set environment to disable proactive research
    os.environ["PROACTIVE_RESEARCH_ENABLED"] = "0"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    try:
        # Import server with disabled proactive research
        import server
        
        # Create server instance
        test_server = server.EnhancedLMStudioMCPServer()
        
        # Check if proactive research is disabled
        if hasattr(test_server, 'proactive_research'):
            if hasattr(test_server.proactive_research, '_disabled'):
                if test_server.proactive_research._disabled:
                    print("✓ Proactive research properly disabled")
                    return True
                else:
                    print("✗ Proactive research not disabled despite PROACTIVE_RESEARCH_ENABLED=0")
                    return False
            else:
                print("✗ Proactive research missing _disabled attribute")
                return False
        else:
            print("✓ Proactive research not initialized (disabled)")
            return True
        
    except Exception as e:
        print(f"✗ Error testing proactive research disable: {e}")
        return False

def test_logging_level():
    """Test that logging level can be controlled"""
    print("\n" + "=" * 80)
    print("PHASE 2.2: Logging Level Control Test")
    print("=" * 80)
    
    try:
        import logging
        
        # Test WARNING level
        os.environ["LOG_LEVEL"] = "WARNING"
        
        # Re-import to pick up new log level
        import importlib
        import server
        importlib.reload(server)
        
        # Check if root logger level is set correctly
        root_logger = logging.getLogger()
        current_level = root_logger.level
        
        print(f"✓ Current logging level: {logging.getLevelName(current_level)}")
        
        if current_level >= logging.WARNING:
            print("✓ Logging level set to WARNING or higher (less verbose)")
            return True
        else:
            print("✗ Logging level not set correctly")
            return False
        
    except Exception as e:
        print(f"✗ Error testing logging level: {e}")
        return False

def test_singleton_pattern():
    """Test that proactive research orchestrator follows singleton pattern"""
    print("\n" + "=" * 80)
    print("PHASE 2.3: Singleton Pattern Test")
    print("=" * 80)
    
    try:
        # Enable proactive research for this test
        os.environ["PROACTIVE_RESEARCH_ENABLED"] = "1"
        
        import server
        from proactive_research import ProactiveResearchOrchestrator, _global_orchestrator_instance
        
        # Create first server instance
        server1 = server.EnhancedLMStudioMCPServer()
        
        # Create second server instance
        server2 = server.EnhancedLMStudioMCPServer()
        
        # Check if both servers share the same proactive research instance
        if hasattr(server1, 'proactive_research') and hasattr(server2, 'proactive_research'):
            if server1.proactive_research is server2.proactive_research:
                print("✓ Singleton pattern working - same instance shared")
                return True
            else:
                print("✗ Singleton pattern broken - different instances")
                return False
        else:
            print("✗ Proactive research not initialized on both servers")
            return False
        
    except Exception as e:
        print(f"✗ Error testing singleton pattern: {e}")
        return False

def test_graceful_shutdown():
    """Test graceful shutdown of background threads"""
    print("\n" + "=" * 80)
    print("PHASE 2.4: Graceful Shutdown Test")
    print("=" * 80)
    
    try:
        # Enable proactive research for this test
        os.environ["PROACTIVE_RESEARCH_ENABLED"] = "1"
        
        import server
        
        # Create server instance
        test_server = server.EnhancedLMStudioMCPServer()
        
        if hasattr(test_server, 'proactive_research'):
            orchestrator = test_server.proactive_research
            
            # Check if background thread is running
            if hasattr(orchestrator, '_bg_thread') and orchestrator._bg_thread:
                print("✓ Background thread detected")
                
                # Test graceful stop
                orchestrator.stop()
                
                # Wait a moment for shutdown
                time.sleep(1)
                
                if not orchestrator._running:
                    print("✓ Orchestrator stopped gracefully")
                    return True
                else:
                    print("✗ Orchestrator did not stop")
                    return False
            else:
                print("✓ No background thread to stop (disabled or async)")
                return True
        else:
            print("✓ No proactive research to stop")
            return True
        
    except Exception as e:
        print(f"✗ Error testing graceful shutdown: {e}")
        return False

def main():
    """Run runtime cleanup tests"""
    print("LM Studio MCP Server - Runtime Error Cleanup Tests")
    print("=" * 80)
    
    tests = [
        test_proactive_research_disabled,
        test_logging_level,
        test_singleton_pattern,
        test_graceful_shutdown
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("RUNTIME CLEANUP TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        phase_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"{status} Phase 2.{i+1}: {phase_name}")
    
    if passed == total:
        print("\n🎉 RUNTIME CLEANUP SUCCESSFUL!")
        print("✅ All runtime error cleanup measures working")
        print("✅ Production configuration optimized")
        return True
    else:
        print(f"\n⚠️ RUNTIME CLEANUP ISSUES")
        print(f"✗ {total - passed} cleanup measure(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
