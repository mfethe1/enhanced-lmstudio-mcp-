#!/usr/bin/env python3
"""
Test with environment override to expose all tools.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

def test_with_env_override():
    """Test with EXPOSE_PUBLIC_ONLY=0 environment override."""
    print("🔧 Testing with Environment Override")
    print("=" * 50)
    
    # Set environment variable
    os.environ["EXPOSE_PUBLIC_ONLY"] = "0"
    print("✅ Set EXPOSE_PUBLIC_ONLY=0")
    
    try:
        # Import the server module
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        print("✅ Server module imported successfully")
        
        # Test MCP message handling with override
        tools_list_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        response = server_module.handle_message(tools_list_message)
        print(f"✅ tools/list response type: {type(response)}")
        
        if isinstance(response, dict):
            result = response.get("result", {})
            if isinstance(result, dict):
                tools = result.get("tools", [])
                print(f"✅ MCP tools/list returned {len(tools)} tools with EXPOSE_PUBLIC_ONLY=0")
                
                # Show first 10 tools from MCP response
                for i, tool in enumerate(tools[:10]):
                    if isinstance(tool, dict):
                        name = tool.get("name", "unknown")
                        desc = tool.get("description", "No description")[:50]
                        has_schema = "inputSchema" in tool or "input_schema" in tool
                        print(f"   {i+1:2d}. {name} - {desc} (schema: {has_schema})")
                    else:
                        print(f"   {i+1:2d}. Invalid tool format: {type(tool)}")
                
                if len(tools) > 10:
                    print(f"   ... and {len(tools) - 10} more tools")
                
                return len(tools)
            else:
                print(f"❌ Invalid result format: {type(result)}")
                return 0
        else:
            print(f"❌ Invalid response format: {type(response)}")
            return 0
            
    except Exception as e:
        print(f"❌ Error testing with override: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Main test function."""
    print("🚀 Environment Override Test")
    print("=" * 50)
    
    tool_count = test_with_env_override()
    
    if tool_count > 20:  # Should be around 63 tools
        print(f"\n🎉 Success! {tool_count} tools exposed with EXPOSE_PUBLIC_ONLY=0")
        print("\n💡 The MCP server is working correctly!")
        print("If tools still don't appear in your client:")
        print("1. Make sure you're using the correct mcp.json file")
        print("2. Restart your MCP client completely")
        print("3. Check client logs for connection errors")
        print("4. Verify the server path in mcp.json is absolute and correct")
    elif tool_count > 10:
        print(f"\n⚠️  Only {tool_count} tools exposed - EXPOSE_PUBLIC_ONLY might still be 1")
        print("Check your environment variable configuration")
    else:
        print(f"\n❌ Only {tool_count} tools found - something is wrong")

if __name__ == "__main__":
    main()
