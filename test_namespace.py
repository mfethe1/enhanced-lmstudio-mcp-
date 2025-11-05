#!/usr/bin/env python3
"""Test that tool namespacing works correctly and prevents duplicates"""

import os
import sys
import json

# Set the prefix
os.environ['MCP_TOOLS_PREFIX'] = 'jarvis_mcp'

# Import server functions
from server import get_all_tools, get_public_tools, _apply_tools_namespace, _get_tools_prefix

def test_tool_namespacing():
    """Test that all tools are properly namespaced"""
    
    print("Testing tool namespacing...")
    print(f"Current prefix: {_get_tools_prefix()}")
    
    # Get all tools
    all_tools = get_all_tools()
    public_tools = get_public_tools()
    
    # Apply namespace
    all_tools_namespaced = _apply_tools_namespace(all_tools)
    public_tools_namespaced = _apply_tools_namespace(public_tools)
    
    # Check all tools
    print(f"\nAll tools count: {len(all_tools.get('tools', []))}")
    print(f"All tools namespaced count: {len(all_tools_namespaced.get('tools', []))}")
    
    # Check for duplicates in namespaced version
    all_names = [t['name'] for t in all_tools_namespaced.get('tools', []) if 'name' in t]
    duplicates = [name for name in all_names if all_names.count(name) > 1]
    
    if duplicates:
        print(f"\n❌ Found duplicate tool names after namespacing:")
        for dup in set(duplicates):
            print(f"  - {dup}")
        return False
    else:
        print("\n✅ No duplicate tool names after namespacing")
    
    # Check that all tools have the prefix
    prefix = _get_tools_prefix()
    unprefixed = [name for name in all_names if not name.startswith(prefix + "_")]
    if unprefixed:
        print(f"\n❌ Found tools without prefix:")
        for name in unprefixed[:5]:
            print(f"  - {name}")
    else:
        print(f"✅ All tools have the prefix '{prefix}_'")
    
    # Show sample of namespaced tools
    print(f"\nSample of namespaced tool names:")
    for tool in all_tools_namespaced.get('tools', [])[:10]:
        if 'name' in tool:
            print(f"  - {tool['name']}")
    
    # Test public tools
    print(f"\n\nPublic tools count: {len(public_tools.get('tools', []))}")
    print(f"Public tools namespaced count: {len(public_tools_namespaced.get('tools', []))}")
    
    public_names = [t['name'] for t in public_tools_namespaced.get('tools', []) if 'name' in t]
    print(f"\nPublic tool names:")
    for name in public_names:
        print(f"  - {name}")
    
    return True

def test_prefix_stripping():
    """Test that tool name prefix stripping works in handle_tool_call"""
    from server import handle_tool_call
    
    print("\n\nTesting prefix stripping in handle_tool_call...")
    
    # Mock a tools/call message with prefixed name
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "jarvis_mcp_health_check",
            "arguments": {}
        }
    }
    
    try:
        result = handle_tool_call(message)
        if "error" not in result:
            print("✅ Prefix stripping works - tool call succeeded")
            return True
        else:
            print(f"❌ Tool call failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Exception during tool call: {e}")
        return False

def test_no_conflicts():
    """Test that common tool names are properly namespaced"""
    
    print("\n\nTesting for potential conflicts...")
    
    common_names = [
        'execute_code', 'read_file', 'write_file', 'list_directory',
        'analyze_code', 'search', 'test', 'run', 'debug'
    ]
    
    all_tools = get_all_tools()
    namespaced = _apply_tools_namespace(all_tools)
    
    tool_names = [t['name'] for t in namespaced.get('tools', []) if 'name' in t]
    
    conflicts = []
    for common in common_names:
        # Check if any tool has this exact name (without prefix)
        if common in tool_names:
            conflicts.append(common)
    
    if conflicts:
        print(f"❌ Found unprefixed common names that could conflict:")
        for name in conflicts:
            print(f"  - {name}")
        return False
    else:
        print("✅ No unprefixed common tool names found")
        return True

if __name__ == "__main__":
    success = True
    
    success &= test_tool_namespacing()
    success &= test_prefix_stripping()
    success &= test_no_conflicts()
    
    print("\n" + "="*50)
    if success:
        print("✅ All namespace tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)