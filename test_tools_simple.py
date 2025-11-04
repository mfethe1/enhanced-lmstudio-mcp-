#!/usr/bin/env python3
"""
Simple test to verify tools are being registered correctly.
"""
import sys
import importlib.util
from pathlib import Path

def test_server_tools():
    """Test that the server has tools registered."""
    print("🔧 Testing Server Tools Registration")
    print("=" * 50)
    
    try:
        # Import the server module
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        print("✅ Server module imported successfully")
        
        # Create server instance
        server = server_module.EnhancedLMStudioMCPServer()
        print("✅ Server instance created")
        
        # Check if server has tools
        if hasattr(server, '_tools') and server._tools:
            tools = server._tools
            print(f"✅ Found {len(tools)} tools registered")
            
            # Show first 10 tools
            tool_names = list(tools.keys())
            print("\n📋 Registered tools:")
            for i, name in enumerate(tool_names[:15]):
                print(f"   {i+1:2d}. {name}")
            
            if len(tool_names) > 15:
                print(f"   ... and {len(tool_names) - 15} more tools")
            
            return True
        else:
            print("❌ No tools found in server._tools")
            
            # Check for alternative tool storage
            for attr in dir(server):
                if 'tool' in attr.lower():
                    value = getattr(server, attr)
                    if isinstance(value, dict) and value:
                        print(f"⚠️  Found tools in {attr}: {len(value)} tools")
                        return True
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing server tools: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_info():
    """Test MCP server info."""
    print("\n🔧 Testing MCP Server Info")
    print("=" * 50)
    
    try:
        # Import the server module
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        # Create server instance
        server = server_module.EnhancedLMStudioMCPServer()
        
        # Test get_server_info
        if hasattr(server, 'get_server_info'):
            info = server.get_server_info()
            print(f"✅ Server name: {info.get('name', 'unknown')}")
            print(f"✅ Server version: {info.get('version', 'unknown')}")
            return True
        else:
            print("❌ get_server_info method not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing server info: {e}")
        return False

def test_health_check():
    """Test health check functionality."""
    print("\n🔧 Testing Health Check")
    print("=" * 50)
    
    try:
        # Import the server module
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        # Create server instance
        server = server_module.EnhancedLMStudioMCPServer()
        
        # Test health check
        if hasattr(server_module, 'handle_health_check'):
            health = server_module.handle_health_check({}, server)
            print(f"✅ Health check status: {health.get('status', 'unknown')}")
            print(f"✅ LM Studio: {health.get('lm_studio', {}).get('status', 'unknown')}")
            return True
        else:
            print("❌ handle_health_check function not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing health check: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Simple Tools Test")
    print("=" * 50)
    
    tools_ok = test_server_tools()
    info_ok = test_mcp_info()
    health_ok = test_health_check()
    
    if tools_ok and info_ok and health_ok:
        print("\n🎉 All tests passed!")
        print("\n💡 If tools still don't appear in your MCP client:")
        print("1. Make sure you're using the correct mcp.json file")
        print("2. Restart your MCP client completely")
        print("3. Check that the server path in mcp.json is correct")
        print("4. Verify EXPOSE_PUBLIC_ONLY=0 in the config")
    else:
        print("\n❌ Some tests failed")
        print("Check the errors above for troubleshooting")

if __name__ == "__main__":
    main()
