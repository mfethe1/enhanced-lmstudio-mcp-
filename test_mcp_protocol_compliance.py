#!/usr/bin/env python3
"""
Test MCP protocol compliance for Augment Code integration.
"""

import json
import os
import sys
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def test_mcp_initialize():
    """Test MCP initialize protocol"""
    print("=" * 80)
    print("TESTING: MCP Initialize Protocol")
    print("=" * 80)
    
    # Set environment variables
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Test initialize message
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": True
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "augment-code",
                "version": "1.0.0"
            }
        }
    }
    
    print("Testing MCP initialize...")
    
    try:
        response = server.handle_message(init_msg)
        
        print(f"Response structure:")
        print(f"- jsonrpc: {response.get('jsonrpc')}")
        print(f"- id: {response.get('id')}")
        print(f"- has result: {'result' in response}")
        print(f"- has error: {'error' in response}")
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got initialize result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for required MCP fields
                if 'protocolVersion' in result:
                    print(f"✓ Protocol version: {result['protocolVersion']}")
                
                if 'capabilities' in result:
                    caps = result['capabilities']
                    print(f"✓ Capabilities: {list(caps.keys())}")
                    
                    if 'tools' in caps:
                        tools_caps = caps['tools']
                        print(f"✓ Tools capabilities: {tools_caps}")
                        
                        if tools_caps.get('listChanged'):
                            print("✓ Tools listChanged advertised")
                
                if 'serverInfo' in result:
                    server_info = result['serverInfo']
                    print(f"✓ Server info: {server_info}")
            
            return True
        
        elif 'error' in response:
            error = response['error']
            print(f"✗ Initialize error: {error.get('message', '')}")
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_mcp_tools_list():
    """Test MCP tools/list protocol"""
    print("\n" + "=" * 80)
    print("TESTING: MCP tools/list Protocol")
    print("=" * 80)
    
    # Test tools/list message
    tools_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    print("Testing MCP tools/list...")
    
    try:
        response = server.handle_message(tools_msg)
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got tools/list result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                if 'tools' in result:
                    tools = result['tools']
                    print(f"✓ Tools array: {len(tools)} tools")
                    
                    # Check first few tools for MCP compliance
                    for i, tool in enumerate(tools[:3]):
                        if isinstance(tool, dict):
                            name = tool.get('name', 'unknown')
                            has_input_schema = 'input_schema' in tool
                            has_inputSchema = 'inputSchema' in tool
                            has_description = 'description' in tool
                            
                            print(f"  Tool {i+1}: {name}")
                            print(f"    - description: {has_description}")
                            print(f"    - input_schema: {has_input_schema}")
                            print(f"    - inputSchema: {has_inputSchema}")
                            
                            if has_input_schema:
                                schema = tool['input_schema']
                                if isinstance(schema, dict) and 'type' in schema:
                                    print(f"    - schema type: {schema['type']}")
                    
                    # Check for expected tools
                    tool_names = [t.get('name') for t in tools if isinstance(t, dict)]
                    expected = ['health_check', 'get_version', 'smart_task']
                    found = [name for name in expected if name in tool_names]
                    print(f"✓ Expected tools found: {len(found)}/{len(expected)}")
                    
                    return len(tools) > 10 and len(found) >= 2
            
            return False
        
        elif 'error' in response:
            error = response['error']
            print(f"✗ tools/list error: {error.get('message', '')}")
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_mcp_tool_call():
    """Test MCP tools/call protocol"""
    print("\n" + "=" * 80)
    print("TESTING: MCP tools/call Protocol")
    print("=" * 80)
    
    # Test tools/call message
    call_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_version",
            "arguments": {}
        }
    }
    
    print("Testing MCP tools/call with get_version...")
    
    try:
        response = server.handle_message(call_msg)
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got tools/call result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for MCP tool call response format
                if 'content' in result:
                    content = result['content']
                    print(f"✓ Tool content: {str(content)[:100]}...")
                
                # Check for isError field (MCP spec)
                if 'isError' in result:
                    is_error = result['isError']
                    print(f"✓ isError field: {is_error}")
                    
                    if not is_error:
                        print("✓ Tool call succeeded")
                        return True
                    else:
                        print("✗ Tool call reported error")
                        return False
                else:
                    # If no isError field, assume success if we got content
                    return 'content' in result
            
            return True
        
        elif 'error' in response:
            error = response['error']
            print(f"✗ tools/call error: {error.get('message', '')}")
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def main():
    """Run MCP protocol compliance tests"""
    print("MCP Protocol Compliance Test Suite")
    print("=" * 80)
    
    tests = [
        test_mcp_initialize,
        test_mcp_tools_list,
        test_mcp_tool_call
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
    print("MCP PROTOCOL COMPLIANCE SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_func.__name__}")
    
    if passed == total:
        print("\n🎉 MCP protocol compliance tests passed!")
        print("The server should work correctly with Augment Code.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("There may be issues with Augment Code integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
