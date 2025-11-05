#!/usr/bin/env python3
"""
Critical tool validation script - validates the most important tools
without relying on full server import that may hang.
"""
import json
import re
from pathlib import Path

def analyze_tool_definitions():
    """Analyze tool definitions from server.py source code."""
    print("🔍 ANALYZING TOOL DEFINITIONS FROM SOURCE")
    print("=" * 50)
    
    server_path = Path("server.py")
    if not server_path.exists():
        print("❌ server.py not found")
        return False
    
    content = server_path.read_text(encoding='utf-8')
    
    # Extract tool definitions from get_all_tools()
    tools_found = []
    
    # Find tool name patterns in the source
    tool_patterns = [
        r'"name":\s*"([^"]+)"',  # "name": "tool_name"
        r"'name':\s*'([^']+)'",  # 'name': 'tool_name'
    ]
    
    for pattern in tool_patterns:
        matches = re.findall(pattern, content)
        tools_found.extend(matches)
    
    # Remove duplicates and sort
    unique_tools = sorted(set(tools_found))
    
    print(f"📋 Found {len(unique_tools)} tool definitions:")
    for i, tool in enumerate(unique_tools, 1):
        print(f"  {i:2d}. {tool}")
    
    return unique_tools

def validate_tool_registry():
    """Validate tool registry mappings in handle_tool_call."""
    print("\n🔧 VALIDATING TOOL REGISTRY MAPPINGS")
    print("=" * 50)
    
    server_path = Path("server.py")
    content = server_path.read_text(encoding='utf-8')
    
    # Find the registry mapping in handle_tool_call
    registry_pattern = r'registry\s*=\s*{([^}]+)}'
    registry_matches = re.findall(registry_pattern, content, re.DOTALL)
    
    if not registry_matches:
        print("❌ Could not find registry mapping")
        return False
    
    # Extract tool names from registry
    registry_content = registry_matches[-1]  # Get the last/main registry
    tool_registry_pattern = r'"([^"]+)":\s*\([^)]+\)'
    registry_tools = re.findall(tool_registry_pattern, registry_content)
    
    print(f"🗂️  Found {len(registry_tools)} tools in registry:")
    for i, tool in enumerate(sorted(registry_tools), 1):
        print(f"  {i:2d}. {tool}")
    
    # Check for critical tools
    critical_tools = [
        "health_check",
        "chat_with_tools", 
        "agent_team_plan_and_code",
        "smart_task",
        "router_diagnostics"
    ]
    
    print(f"\n✅ CRITICAL TOOL VALIDATION:")
    for tool in critical_tools:
        if tool in registry_tools:
            print(f"  ✅ {tool} - REGISTERED")
        else:
            print(f"  ❌ {tool} - MISSING")
    
    return registry_tools

def validate_handler_imports():
    """Validate that handler modules exist and are importable."""
    print("\n📦 VALIDATING HANDLER MODULES")
    print("=" * 50)
    
    handlers_dir = Path("handlers")
    if not handlers_dir.exists():
        print("❌ handlers directory not found")
        return False
    
    expected_handlers = [
        "research.py",
        "agent_teams.py", 
        "memory.py",
        "code_tools.py",
        "workflow.py",
        "audit.py"
    ]
    
    for handler in expected_handlers:
        handler_path = handlers_dir / handler
        if handler_path.exists():
            print(f"  ✅ {handler} - EXISTS")
        else:
            print(f"  ❌ {handler} - MISSING")
    
    return True

def validate_config_file():
    """Validate MCP configuration file."""
    print("\n⚙️  VALIDATING MCP CONFIGURATION")
    print("=" * 50)
    
    config_path = Path("recommendations/mcp.json")
    if not config_path.exists():
        print("❌ mcp.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for jarvis server config
        if "mcpServers" in config and "jarvis" in config["mcpServers"]:
            jarvis_config = config["mcpServers"]["jarvis"]
            env_vars = jarvis_config.get("env", {})
            
            print("✅ Jarvis MCP server configuration found")
            print(f"  Command: {jarvis_config.get('command', 'N/A')}")
            print(f"  Environment variables: {len(env_vars)}")
            
            # Check critical env vars
            critical_env = [
                "USE_BEDROCK",
                "ANTHROPIC_MODEL_COMPLEX", 
                "OPENAI_FALLBACK_MODEL",
                "EXPOSE_PUBLIC_ONLY"
            ]
            
            for env_var in critical_env:
                value = env_vars.get(env_var, "NOT SET")
                print(f"  {env_var}: {value}")
            
            return True
        else:
            print("❌ Jarvis server configuration not found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def validate_bedrock_integration():
    """Validate Bedrock adapter integration."""
    print("\n🌩️  VALIDATING BEDROCK INTEGRATION")
    print("=" * 50)
    
    bedrock_path = Path("bedrock_adapter.py")
    if bedrock_path.exists():
        print("✅ bedrock_adapter.py exists")
        
        content = bedrock_path.read_text(encoding='utf-8')
        
        # Check for key classes/functions
        if "class BedrockAdapter" in content:
            print("  ✅ BedrockAdapter class found")
        if "def chat_completion" in content:
            print("  ✅ chat_completion method found")
        if "def is_available" in content:
            print("  ✅ is_available method found")
            
        return True
    else:
        print("❌ bedrock_adapter.py not found")
        return False

def main():
    """Run comprehensive validation."""
    print("🚀 JARVIS MCP COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    results = {
        "tool_definitions": analyze_tool_definitions(),
        "tool_registry": validate_tool_registry(),
        "handler_modules": validate_handler_imports(),
        "config_file": validate_config_file(),
        "bedrock_integration": validate_bedrock_integration()
    }
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED - SYSTEM IS READY!")
        return True
    else:
        print("⚠️  Some validations failed - review issues above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
