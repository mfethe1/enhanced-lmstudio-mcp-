#!/usr/bin/env python3
"""
Test MCP protocol directly to verify tools are being exposed correctly.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

def test_mcp_tools():
    """Test MCP protocol by sending initialize and list_tools requests."""
    print("🔧 Testing MCP Protocol Direct Communication")
    print("=" * 60)
    
    try:
        # Start the MCP server process
        server_path = Path(__file__).parent / "server.py"
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        print("✅ MCP server process started")
        
        # Send initialize request
        initialize_request = {
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
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("📤 Sending initialize request...")
        process.stdin.write(json.dumps(initialize_request) + "\n")
        process.stdin.flush()
        
        # Read initialize response
        response_line = process.stdout.readline()
        if response_line:
            try:
                init_response = json.loads(response_line.strip())
                print(f"✅ Initialize response: {init_response.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
            except json.JSONDecodeError:
                print(f"⚠️  Non-JSON response: {response_line[:100]}...")
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        print("📤 Sending initialized notification...")
        process.stdin.write(json.dumps(initialized_notification) + "\n")
        process.stdin.flush()
        
        # Send list_tools request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        print("📤 Sending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()
        
        # Read tools response
        tools_response_line = process.stdout.readline()
        if tools_response_line:
            try:
                tools_response = json.loads(tools_response_line.strip())
                tools = tools_response.get('result', {}).get('tools', [])
                print(f"✅ Tools found: {len(tools)}")
                
                if tools:
                    print("\n📋 Available tools:")
                    for i, tool in enumerate(tools[:10]):  # Show first 10
                        name = tool.get('name', 'unknown')
                        description = tool.get('description', 'No description')[:60]
                        print(f"   {i+1:2d}. {name} - {description}")
                    
                    if len(tools) > 10:
                        print(f"   ... and {len(tools) - 10} more tools")
                    
                    return True
                else:
                    print("❌ No tools found in response")
                    return False
                    
            except json.JSONDecodeError:
                print(f"⚠️  Non-JSON tools response: {tools_response_line[:100]}...")
                return False
        else:
            print("❌ No response to tools/list request")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MCP protocol: {e}")
        return False
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

def test_config_validation():
    """Test the mcp.json configuration."""
    print("\n🔧 Testing MCP Configuration")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "recommendations" / "mcp.json"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Config file loaded successfully")
        
        # Check structure
        if "mcpServers" not in config:
            print("❌ Missing 'mcpServers' key in config")
            return False
        
        servers = config["mcpServers"]
        print(f"📋 Found servers: {list(servers.keys())}")

        if "jarvis" not in servers:
            print("❌ Missing 'jarvis' server in config")
            print(f"Available servers: {list(servers.keys())}")
            return False
        
        jarvis_config = servers["jarvis"]
        
        # Check required fields
        required_fields = ["type", "command", "args"]
        for field in required_fields:
            if field not in jarvis_config:
                print(f"❌ Missing required field '{field}' in jarvis config")
                return False
        
        print("✅ Config structure is valid")
        
        # Check environment variables
        env = jarvis_config.get("env", {})
        important_vars = [
            "EXPOSE_PUBLIC_ONLY",
            "LMSTUDIO_API_BASE", 
            "LMSTUDIO_MODEL"
        ]
        
        for var in important_vars:
            if var in env:
                print(f"✅ {var} = {env[var]}")
            else:
                print(f"⚠️  {var} not set")
        
        # Check EXPOSE_PUBLIC_ONLY specifically
        expose_public = env.get("EXPOSE_PUBLIC_ONLY", "1")
        if expose_public == "0":
            print("✅ EXPOSE_PUBLIC_ONLY=0 (all tools exposed)")
        else:
            print("⚠️  EXPOSE_PUBLIC_ONLY=1 (limited tools exposed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 MCP Tools Troubleshooting")
    print("=" * 60)
    
    # Test configuration first
    config_ok = test_config_validation()
    
    # Test MCP protocol
    if config_ok:
        protocol_ok = test_mcp_tools()
        
        if protocol_ok:
            print("\n🎉 MCP server is working correctly!")
            print("\n💡 If tools still don't appear in your client:")
            print("1. Restart your MCP client (Claude Desktop, etc.)")
            print("2. Check client logs for connection errors")
            print("3. Verify the server path in mcp.json is correct")
        else:
            print("\n❌ MCP protocol test failed")
            print("\n🔧 Troubleshooting steps:")
            print("1. Check server.py for syntax errors")
            print("2. Verify all dependencies are installed")
            print("3. Check server logs for errors")
    else:
        print("\n❌ Configuration validation failed")
        print("Fix the configuration issues above first")

if __name__ == "__main__":
    main()
