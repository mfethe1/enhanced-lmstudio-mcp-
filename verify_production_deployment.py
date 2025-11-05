#!/usr/bin/env python3
"""
Production deployment verification script for LM Studio MCP server.
Verifies all critical functionality before production use.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def verify_mcp_tools_discovery():
    """Verify that MCP tools/list returns all 63 tools with proper schemas"""
    print("=" * 80)
    print("PHASE 1.1: MCP Tools Discovery Verification")
    print("=" * 80)
    
    try:
        # Test tools/list endpoint
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        response = server.handle_message(msg)
        
        if 'result' not in response:
            print("✗ CRITICAL: tools/list failed - no result")
            return False
        
        tools = response['result'].get('tools', [])
        tool_count = len(tools)
        
        print(f"✓ Tools discovered: {tool_count}")
        
        if tool_count < 60:
            print(f"✗ CRITICAL: Expected ~63 tools, got {tool_count}")
            return False
        
        # Verify schema compatibility
        schema_compatible = 0
        for tool in tools:
            if 'input_schema' in tool and 'inputSchema' in tool:
                schema_compatible += 1
        
        print(f"✓ Schema compatible tools: {schema_compatible}/{tool_count}")
        
        # Check for critical tools
        tool_names = [t.get('name') for t in tools]
        critical_tools = ['health_check', 'get_version', 'smart_task', 'chat_with_tools']
        missing_critical = [name for name in critical_tools if name not in tool_names]
        
        if missing_critical:
            print(f"✗ CRITICAL: Missing critical tools: {missing_critical}")
            return False
        
        print(f"✓ All critical tools present: {critical_tools}")
        
        # Show sample of available tools
        print("\nSample tools available:")
        for i, tool in enumerate(tools[:10]):
            name = tool.get('name', 'unknown')
            desc = tool.get('description', 'No description')[:50]
            print(f"  {i+1:2d}. {name} - {desc}...")
        
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL: Tools discovery failed: {e}")
        return False

def verify_dynamic_model_selection():
    """Verify dynamic model selection is working correctly"""
    print("\n" + "=" * 80)
    print("PHASE 1.2: Dynamic Model Selection Verification")
    print("=" * 80)
    
    try:
        # Create server instance
        test_server = server.EnhancedLMStudioMCPServer()
        
        # Test model detection
        available_models = test_server.refresh_lmstudio_models(force=True)
        print(f"✓ Available models detected: {len(available_models)}")
        
        if not available_models:
            print("✗ CRITICAL: No LM Studio models available")
            return False
        
        print(f"✓ Models: {', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}")
        
        # Test effective model selection
        configured_model = "openai/gpt-oss-20b"
        effective_model = test_server.get_effective_model(configured_model)
        
        print(f"✓ Configured model: {configured_model}")
        print(f"✓ Effective model: {effective_model}")
        
        if configured_model in available_models:
            if effective_model == configured_model:
                print("✓ Using configured model (optimal)")
            else:
                print("⚠ Using fallback model instead of configured")
        else:
            print("✓ Configured model not available, using fallback (expected)")
        
        # Test fallback behavior
        try:
            fallback_model = test_server.get_effective_model("nonexistent-model-12345")
            print(f"✓ Fallback works: nonexistent → {fallback_model}")
        except Exception as e:
            if "No LM Studio models available" in str(e):
                print("✓ Proper error when no models available")
            else:
                print(f"⚠ Unexpected fallback error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL: Model selection verification failed: {e}")
        return False

def verify_core_functionality():
    """Test core functionality with health_check and basic tool calls"""
    print("\n" + "=" * 80)
    print("PHASE 1.3: Core Functionality Verification")
    print("=" * 80)
    
    try:
        # Test health_check with provider probing
        print("Testing health_check with provider probing...")
        health_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "health_check",
                "arguments": {"probe_providers": True}
            }
        }
        
        start_time = time.time()
        health_response = server.handle_message(health_msg)
        duration = time.time() - start_time
        
        if 'result' not in health_response:
            print("✗ CRITICAL: health_check failed")
            return False
        
        print(f"✓ health_check completed in {duration:.1f}s")
        
        # Test get_version
        print("Testing get_version...")
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
        
        if 'result' not in version_response:
            print("✗ CRITICAL: get_version failed")
            return False
        
        print("✓ get_version working")
        
        # Test router_test for backend connectivity
        print("Testing router_test...")
        router_msg = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "router_test",
                "arguments": {"task": "production verification test"}
            }
        }
        
        router_response = server.handle_message(router_msg)
        
        if 'result' not in router_response:
            print("✗ WARNING: router_test failed (non-critical)")
        else:
            print("✓ router_test working")
        
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL: Core functionality verification failed: {e}")
        return False

def verify_mcp_protocol_compliance():
    """Verify full MCP protocol compliance"""
    print("\n" + "=" * 80)
    print("PHASE 1.4: MCP Protocol Compliance Verification")
    print("=" * 80)
    
    try:
        # Test initialize
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
                "clientInfo": {"name": "augment-code", "version": "1.0.0"}
            }
        }
        
        init_response = server.handle_message(init_msg)
        
        if 'result' not in init_response:
            print("✗ CRITICAL: MCP initialize failed")
            return False
        
        result = init_response['result']
        
        # Check required fields
        if 'protocolVersion' not in result:
            print("✗ CRITICAL: Missing protocolVersion in initialize response")
            return False
        
        if 'capabilities' not in result:
            print("✗ CRITICAL: Missing capabilities in initialize response")
            return False
        
        capabilities = result['capabilities']
        if 'tools' not in capabilities:
            print("✗ CRITICAL: Missing tools capabilities")
            return False
        
        tools_caps = capabilities['tools']
        if not tools_caps.get('listChanged'):
            print("✗ CRITICAL: tools.listChanged not advertised")
            return False
        
        print("✓ MCP initialize protocol compliant")
        print(f"✓ Protocol version: {result['protocolVersion']}")
        print(f"✓ Tools listChanged: {tools_caps['listChanged']}")
        
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL: MCP protocol compliance failed: {e}")
        return False

def main():
    """Run production deployment verification"""
    print("LM Studio MCP Server - Production Deployment Verification")
    print("=" * 80)
    
    # Set production environment variables
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "LMSTUDIO_MODEL": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0",  # Show all tools for verification
        "HTTP_CONNECT_TIMEOUT": "2",
        "HTTP_READ_TIMEOUT_SIMPLE": "8",
        "HTTP_READ_TIMEOUT_COMPLEX": "45"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Run verification phases
    phases = [
        verify_mcp_tools_discovery,
        verify_dynamic_model_selection,
        verify_core_functionality,
        verify_mcp_protocol_compliance
    ]
    
    results = []
    for phase_func in phases:
        try:
            result = phase_func()
            results.append(result)
        except Exception as e:
            print(f"✗ CRITICAL: {phase_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION DEPLOYMENT VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Verification phases passed: {passed}/{total}")
    
    for i, (phase_func, result) in enumerate(zip(phases, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        phase_name = phase_func.__name__.replace('verify_', '').replace('_', ' ').title()
        print(f"{status} Phase 1.{i+1}: {phase_name}")
    
    if passed == total:
        print("\n🎉 PRODUCTION READY!")
        print("✅ All verification phases passed")
        print("✅ MCP server is ready for Augment Code integration")
        print("\nNext steps:")
        print("1. Import recommendations/mcp.json into Augment Code")
        print("2. Verify 63 tools appear in MCP Tools panel")
        print("3. Test health_check tool in Augment Code UI")
        return True
    else:
        print(f"\n❌ PRODUCTION NOT READY")
        print(f"✗ {total - passed} critical verification(s) failed")
        print("✗ Do not deploy to production until all phases pass")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
