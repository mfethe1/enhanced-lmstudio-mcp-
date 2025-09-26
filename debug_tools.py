#!/usr/bin/env python3
"""
Debug tool registration and MCP protocol.
"""
import sys
import json
import importlib.util
from pathlib import Path

def debug_tools():
    """Debug tool registration."""
    print("🔧 Debugging Tool Registration")
    print("=" * 50)
    
    try:
        # Import the server module
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        print("✅ Server module imported successfully")
        
        # Test get_all_tools
        all_tools = server_module.get_all_tools()
        print(f"✅ get_all_tools() returned: {type(all_tools)}")
        
        if isinstance(all_tools, dict):
            tools = all_tools.get("tools", [])
            print(f"✅ Found {len(tools)} tools in get_all_tools()")
            
            # Show first few tools
            for i, tool in enumerate(tools[:5]):
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "No description")[:50]
                    print(f"   {i+1}. {name} - {desc}")
                else:
                    print(f"   {i+1}. Invalid tool format: {type(tool)}")
        else:
            print(f"❌ get_all_tools() returned invalid format: {type(all_tools)}")
        
        # Test get_public_tools
        public_tools = server_module.get_public_tools()
        print(f"\n✅ get_public_tools() returned: {type(public_tools)}")
        
        if isinstance(public_tools, dict):
            tools = public_tools.get("tools", [])
            print(f"✅ Found {len(tools)} tools in get_public_tools()")
            
            # Show all public tools
            for i, tool in enumerate(tools):
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "No description")[:50]
                    print(f"   {i+1}. {name} - {desc}")
                else:
                    print(f"   {i+1}. Invalid tool format: {type(tool)}")
        else:
            print(f"❌ get_public_tools() returned invalid format: {type(public_tools)}")
        
        # Test MCP message handling
        print(f"\n🔧 Testing MCP Message Handling")
        print("=" * 50)
        
        # Test tools/list message
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
                print(f"✅ MCP tools/list returned {len(tools)} tools")
                
                # Show first few tools from MCP response
                for i, tool in enumerate(tools[:5]):
                    if isinstance(tool, dict):
                        name = tool.get("name", "unknown")
                        desc = tool.get("description", "No description")[:50]
                        has_schema = "inputSchema" in tool or "input_schema" in tool
                        print(f"   {i+1}. {name} - {desc} (schema: {has_schema})")
                    else:
                        print(f"   {i+1}. Invalid tool format: {type(tool)}")
            else:
                print(f"❌ Invalid result format: {type(result)}")
        else:
            print(f"❌ Invalid response format: {type(response)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error debugging tools: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("🚀 Tool Registration Debug")
    print("=" * 50)
    
    success = debug_tools()
    
    if success:
        print("\n🎉 Debug completed successfully!")
        print("\n💡 Next steps:")
        print("1. If tools are showing up here but not in your client, restart the client")
        print("2. Check that the mcp.json path is correct")
        print("3. Verify EXPOSE_PUBLIC_ONLY=0 in your config")
    else:
        print("\n❌ Debug failed - check errors above")

if __name__ == "__main__":
    main()
