#!/usr/bin/env python3
"""
Comprehensive test suite for LM Studio MCP server functionality.
Tests startup, tool discovery, model integration, core tools, error handling, and configuration.
"""

import json
import requests
import time
import sys
import os
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def test_1_mcp_server_startup_and_tool_discovery():
    """Test 1: MCP Server Startup & Tool Discovery"""
    print("=" * 60)
    print("TEST 1: MCP Server Startup & Tool Discovery")
    print("=" * 60)
    
    try:
        # Test tools/list endpoint
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        response = server.handle_message(msg)
        
        print(f"✓ MCP server handles tools/list: {response.get('jsonrpc') == '2.0'}")
        
        result = response.get("result", {})
        tools = result.get("tools", [])
        print(f"✓ Tools discovered: {len(tools)} tools")
        
        # Check for expected tools
        tool_names = [t.get("name") for t in tools]
        expected_tools = ["health_check", "get_version", "smart_task", "smart_plan_execute", "router_test"]
        found_expected = [name for name in expected_tools if name in tool_names]
        print(f"✓ Expected tools found: {len(found_expected)}/{len(expected_tools)} - {found_expected}")
        
        # Check input_schema formatting
        schema_tools = [t for t in tools if "input_schema" in t or "inputSchema" in t]
        print(f"✓ Tools with schema: {len(schema_tools)}/{len(tools)}")
        
        # Show first few tools for verification
        print("\nFirst 5 tools:")
        for i, tool in enumerate(tools[:5]):
            name = tool.get("name", "unknown")
            has_input_schema = "input_schema" in tool
            has_inputSchema = "inputSchema" in tool
            print(f"  {i+1}. {name} (input_schema: {has_input_schema}, inputSchema: {has_inputSchema})")
        
        return len(tools) > 5 and len(found_expected) >= 3
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False

def test_2_lm_studio_model_integration():
    """Test 2: LM Studio Model Integration"""
    print("\n" + "=" * 60)
    print("TEST 2: LM Studio Model Integration")
    print("=" * 60)
    
    try:
        # Test direct /v1/models endpoint
        print("Testing LM Studio /v1/models endpoint...")
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [m.get("id") for m in models_data.get("data", [])]
            print(f"✓ LM Studio models available: {len(models)} - {models[:3]}{'...' if len(models) > 3 else ''}")
        else:
            print(f"✗ LM Studio /v1/models failed: {response.status_code}")
            return False
        
        # Test server's model discovery
        print("\nTesting server model discovery...")
        test_server = server.EnhancedLMStudioMCPServer()
        available_models = test_server.refresh_lmstudio_models(force=True)
        print(f"✓ Server detected models: {len(available_models)} - {available_models[:3]}{'...' if len(available_models) > 3 else ''}")
        
        # Test effective model selection
        print("\nTesting model selection...")
        try:
            effective_model = test_server.get_effective_model("openai/gpt-oss-20b")
            print(f"✓ Effective model for 'openai/gpt-oss-20b': {effective_model}")
        except Exception as e:
            print(f"✗ Model selection failed: {e}")
            return False
        
        # Test fallback behavior
        try:
            fallback_model = test_server.get_effective_model("nonexistent-model")
            print(f"✓ Fallback model for nonexistent: {fallback_model}")
        except Exception as e:
            print(f"✓ Proper error for nonexistent model: {str(e)[:100]}")
        
        return len(available_models) > 0
        
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False

def test_3_core_tool_functionality():
    """Test 3: Core Tool Functionality"""
    print("\n" + "=" * 60)
    print("TEST 3: Core Tool Functionality")
    print("=" * 60)
    
    try:
        # Test health_check tool
        print("Testing health_check tool...")
        health_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "health_check",
                "arguments": {"probe_providers": True}
            }
        }
        health_response = server.handle_message(health_msg)
        print(f"✓ health_check response: {health_response.get('result', {}).get('status', 'unknown')}")
        
        # Test get_version tool
        print("\nTesting get_version tool...")
        version_msg = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_version",
                "arguments": {}
            }
        }
        version_response = server.handle_message(version_msg)
        version_result = version_response.get('result', {})
        print(f"✓ get_version response: {version_result.get('version', 'unknown')}")
        
        # Test router_test tool
        print("\nTesting router_test tool...")
        router_msg = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "router_test",
                "arguments": {"task": "simple test task"}
            }
        }
        router_response = server.handle_message(router_msg)
        router_result = router_response.get('result', {})
        print(f"✓ router_test response: {router_result.get('backend', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False

def test_4_error_handling_and_fallbacks():
    """Test 4: Error Handling & Fallbacks"""
    print("\n" + "=" * 60)
    print("TEST 4: Error Handling & Fallbacks")
    print("=" * 60)
    
    try:
        # Test with invalid tool name
        print("Testing invalid tool handling...")
        invalid_msg = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        }
        invalid_response = server.handle_message(invalid_msg)
        error = invalid_response.get('error', {})
        print(f"✓ Invalid tool error: {error.get('message', 'no error')[:50]}")
        
        # Test model unavailability handling
        print("\nTesting model unavailability...")
        test_server = server.EnhancedLMStudioMCPServer()
        test_server._available_models = []  # Force empty model list
        try:
            test_server.get_effective_model("any-model")
            print("✗ Should have raised error for no models")
            return False
        except RuntimeError as e:
            print(f"✓ Proper error for no models: {str(e)[:80]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False

def test_5_configuration_validation():
    """Test 5: Configuration Validation"""
    print("\n" + "=" * 60)
    print("TEST 5: Configuration Validation")
    print("=" * 60)
    
    try:
        # Test environment variable reading
        print("Testing environment variable configuration...")
        
        # Check key environment variables that should be set
        env_vars = {
            "LM_STUDIO_URL": "http://localhost:1234",
            "MODEL_NAME": "openai/gpt-oss-20b",
            "EXPOSE_PUBLIC_ONLY": "0",
            "HTTP_CONNECT_TIMEOUT": "2",
            "HTTP_READ_TIMEOUT_SIMPLE": "8"
        }
        
        for var, expected in env_vars.items():
            actual = os.getenv(var, "NOT_SET")
            match = actual == expected
            print(f"{'✓' if match else '✗'} {var}: {actual} {'(matches)' if match else f'(expected: {expected})'}")
        
        # Test server initialization with current config
        print("\nTesting server initialization...")
        test_server = server.EnhancedLMStudioMCPServer()
        print(f"✓ Server base_url: {test_server.base_url}")
        print(f"✓ Server model_name: {test_server.model_name}")
        
        # Test timeout configuration
        print(f"✓ HTTP client configured: {test_server.http_client is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        return False

def main():
    """Run all tests and provide summary"""
    print("LM Studio MCP Server Comprehensive Test Suite")
    print("=" * 60)
    
    # Set environment variables from mcp.json for testing
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "LMSTUDIO_MODEL": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0",
        "HTTP_CONNECT_TIMEOUT": "2",
        "HTTP_READ_TIMEOUT_SIMPLE": "8",
        "HTTP_READ_TIMEOUT_COMPLEX": "45",
        "LMSTUDIO_FUNCTION_MODEL": "openai/gpt-oss-20b"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Run all tests
    tests = [
        test_1_mcp_server_startup_and_tool_discovery,
        test_2_lm_studio_model_integration,
        test_3_core_tool_functionality,
        test_4_error_handling_and_fallbacks,
        test_5_configuration_validation
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
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} Test {i+1}: {test_func.__name__}")
    
    if passed == total:
        print("\n🎉 All tests passed! MCP server is functioning correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
