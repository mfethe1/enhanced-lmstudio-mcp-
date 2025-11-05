#!/usr/bin/env python3
"""
Direct function validation - tests critical functions without full server import
"""
import json
import os
import sys
import time

def test_basic_imports():
    """Test that we can import basic modules"""
    print("🔧 Testing basic imports...")
    
    try:
        import requests
        print("  ✅ requests")
    except Exception as e:
        print(f"  ❌ requests: {e}")
        return False
    
    try:
        import aiohttp
        print("  ✅ aiohttp")
    except Exception as e:
        print(f"  ❌ aiohttp: {e}")
        return False
    
    try:
        from pathlib import Path
        print("  ✅ pathlib")
    except Exception as e:
        print(f"  ❌ pathlib: {e}")
        return False
    
    return True

def test_file_operations():
    """Test basic file operations"""
    print("\n📁 Testing file operations...")
    
    try:
        # Test file exists
        if os.path.exists('server.py'):
            print("  ✅ server.py exists")
        else:
            print("  ❌ server.py missing")
            return False
        
        # Test directory listing
        files = os.listdir('.')
        if len(files) > 0:
            print(f"  ✅ Directory listing: {len(files)} files")
        else:
            print("  ❌ Directory listing failed")
            return False
        
        # Test file reading
        with open('server.py', 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars
            if 'EnhancedLMStudioMCPServer' in content:
                print("  ✅ File reading works")
            else:
                print("  ❌ File reading failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ File operations failed: {e}")
        return False

def test_json_operations():
    """Test JSON operations"""
    print("\n📋 Testing JSON operations...")
    
    try:
        # Test JSON parsing
        test_data = {"test": "value", "number": 42}
        json_str = json.dumps(test_data)
        parsed = json.loads(json_str)
        
        if parsed["test"] == "value" and parsed["number"] == 42:
            print("  ✅ JSON serialization/deserialization")
        else:
            print("  ❌ JSON operations failed")
            return False
        
        # Test MCP config file
        if os.path.exists('recommendations/mcp.json'):
            with open('recommendations/mcp.json', 'r') as f:
                config = json.load(f)
                if 'mcpServers' in config and 'jarvis' in config['mcpServers']:
                    print("  ✅ MCP config file valid")
                else:
                    print("  ❌ MCP config structure invalid")
                    return False
        else:
            print("  ❌ MCP config file missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ JSON operations failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling"""
    print("\n⚙️  Testing environment variables...")
    
    try:
        # Set test env var
        os.environ['TEST_VAR'] = 'test_value'
        
        # Read it back
        if os.getenv('TEST_VAR') == 'test_value':
            print("  ✅ Environment variable set/get")
        else:
            print("  ❌ Environment variable handling failed")
            return False
        
        # Test default values
        default_val = os.getenv('NONEXISTENT_VAR', 'default')
        if default_val == 'default':
            print("  ✅ Environment variable defaults")
        else:
            print("  ❌ Environment variable defaults failed")
            return False
        
        # Clean up
        del os.environ['TEST_VAR']
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment variable test failed: {e}")
        return False

def test_timeout_configurations():
    """Test that timeout configurations are properly set"""
    print("\n⏱️  Testing timeout configurations...")
    
    try:
        # Check if we can read the MCP config
        with open('recommendations/mcp.json', 'r') as f:
            config = json.load(f)
        
        env_vars = config['mcpServers']['jarvis']['env']
        
        # Check critical timeout settings
        critical_timeouts = {
            'HTTP_READ_TIMEOUT_SIMPLE': '200',
            'HTTP_READ_TIMEOUT_COMPLEX': '500',
            'CREW_TOOL_TIMEOUT': '420',
            'SMART_PLAN_IMMEDIATE_TIMEOUT_SEC': '360',
            'ROUTER_BG_TIMEOUT_SEC': '500'
        }
        
        all_good = True
        for key, expected in critical_timeouts.items():
            actual = env_vars.get(key)
            if actual == expected:
                print(f"  ✅ {key}: {actual}")
            else:
                print(f"  ❌ {key}: expected {expected}, got {actual}")
                all_good = False
        
        # Check no fallback setting
        no_fallback = env_vars.get('NO_FALLBACK_PROVIDERS')
        if no_fallback == '1':
            print("  ✅ NO_FALLBACK_PROVIDERS: 1")
        else:
            print(f"  ❌ NO_FALLBACK_PROVIDERS: expected 1, got {no_fallback}")
            all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"  ❌ Timeout configuration test failed: {e}")
        return False

def test_server_structure():
    """Test server.py structure and key functions"""
    print("\n🔧 Testing server structure...")
    
    try:
        with open('server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key classes and functions
        required_elements = [
            'class EnhancedLMStudioMCPServer',
            'def handle_tool_call',
            'def get_all_tools',
            'def handle_health_check',
            'def handle_chat_with_tools',
            '_lmstudio_request_with_retry',
            'NO_FALLBACK_PROVIDERS'
        ]
        
        all_found = True
        for element in required_elements:
            if element in content:
                print(f"  ✅ Found: {element}")
            else:
                print(f"  ❌ Missing: {element}")
                all_found = False
        
        # Check for timeout configurations in code
        timeout_checks = [
            'HTTP_READ_TIMEOUT_SIMPLE',
            'HTTP_READ_TIMEOUT_COMPLEX',
            'CREW_TOOL_TIMEOUT',
            'CIRCUIT_LMSTUDIO_THRESHOLD'
        ]
        
        for check in timeout_checks:
            if check in content:
                print(f"  ✅ Timeout config: {check}")
            else:
                print(f"  ❌ Missing timeout config: {check}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  ❌ Server structure test failed: {e}")
        return False

def test_tool_registry():
    """Test that tools are properly registered"""
    print("\n🛠️  Testing tool registry...")
    
    try:
        with open('server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for the registry definition
        if 'registry = {' in content:
            print("  ✅ Tool registry found")
        else:
            print("  ❌ Tool registry not found")
            return False
        
        # Check for critical tools in registry
        critical_tools = [
            '"health_check"',
            '"chat_with_tools"',
            '"agent_team_plan_and_code"',
            '"smart_task"',
            '"router_diagnostics"'
        ]
        
        all_found = True
        for tool in critical_tools:
            if tool in content:
                print(f"  ✅ Tool registered: {tool}")
            else:
                print(f"  ❌ Tool missing: {tool}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  ❌ Tool registry test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 DIRECT FUNCTION VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Operations", test_file_operations),
        ("JSON Operations", test_json_operations),
        ("Environment Variables", test_environment_variables),
        ("Timeout Configurations", test_timeout_configurations),
        ("Server Structure", test_server_structure),
        ("Tool Registry", test_tool_registry)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ System is properly configured and ready for testing")
        return 0
    else:
        print(f"\n⚠️  {total - passed} VALIDATIONS FAILED")
        print("❌ System needs fixes before production use")
        return 1

if __name__ == '__main__':
    sys.exit(main())
