#!/usr/bin/env python3
"""
Comprehensive test suite for all Jarvis MCP tools.
Tests each tool systematically to ensure full functionality.
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path

# Set minimal timeouts to avoid hanging on network calls
os.environ['HTTP_CONNECT_TIMEOUT'] = '2'
os.environ['HTTP_READ_TIMEOUT_SIMPLE'] = '3'
os.environ['HTTP_READ_TIMEOUT_COMPLEX'] = '5'
os.environ['AGENT_TEAM_FORCE_FALLBACK'] = '1'  # Use fallback to avoid CrewAI issues

def test_tool(tool_name, arguments=None, expected_keys=None):
    """Test a single tool and return results."""
    if arguments is None:
        arguments = {}

    try:
        # Import server module without triggering main()
        import sys
        import importlib.util

        # Load server module without executing __main__ block
        spec = importlib.util.spec_from_file_location("server", "server.py")
        m = importlib.util.module_from_spec(spec)

        # Add to sys.modules to avoid re-import issues
        sys.modules["server"] = m
        spec.loader.exec_module(m)

        # Call the tool
        start_time = time.time()
        response = m.handle_tool_call({
            'jsonrpc': '2.0',
            'id': 999,
            'method': 'tools/call',
            'params': {
                'name': tool_name,
                'arguments': arguments
            }
        })
        duration = time.time() - start_time
        
        # Analyze response
        has_error = 'error' in response
        error_msg = response.get('error', {}).get('message', '') if has_error else None
        result_keys = list(response.get('result', {}).keys()) if not has_error else []
        
        # Check expected keys if provided
        missing_keys = []
        if expected_keys and not has_error:
            result_content = response.get('result', {})
            if isinstance(result_content, dict) and 'content' in result_content:
                # MCP format - check content array
                content = result_content.get('content', [])
                if content and isinstance(content[0], dict):
                    actual_data = content[0].get('text', '')
                    # For JSON responses, try to parse and check keys
                    try:
                        if actual_data.strip().startswith('{'):
                            parsed = json.loads(actual_data)
                            missing_keys = [k for k in expected_keys if k not in parsed]
                    except:
                        pass
        
        return {
            'tool': tool_name,
            'success': not has_error,
            'duration': round(duration, 3),
            'error': error_msg,
            'result_keys': result_keys,
            'missing_keys': missing_keys,
            'response_type': type(response.get('result', {})).__name__
        }
        
    except Exception as e:
        return {
            'tool': tool_name,
            'success': False,
            'duration': 0,
            'error': f"Exception: {str(e)}",
            'result_keys': [],
            'missing_keys': [],
            'response_type': 'exception'
        }

def get_all_tools():
    """Get list of all available tools."""
    try:
        # Import server module without triggering main()
        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location("server", "server.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["server"] = m
        spec.loader.exec_module(m)

        response = m.handle_message({
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/list'
        })
        tools = response.get('result', {}).get('tools', [])
        return [t.get('name') for t in tools if t.get('name')]
    except Exception as e:
        print(f"Error getting tools: {e}")
        traceback.print_exc()
        return []

def main():
    print("🔧 COMPREHENSIVE JARVIS MCP TOOL TESTING")
    print("=" * 60)
    
    # Get all tools
    all_tools = get_all_tools()
    print(f"📋 Found {len(all_tools)} tools to test")
    print()
    
    if not all_tools:
        print("❌ No tools found! Exiting.")
        return
    
    # Test configurations for each tool
    test_configs = {
        # Core system tools
        'health_check': {'arguments': {'probe_lm': False, 'probe_providers': True}},
        'get_version': {'arguments': {}},
        'router_config': {'arguments': {}},
        
        # Memory tools
        'store_memory': {'arguments': {'key': 'test_key', 'value': 'test_value', 'category': 'test'}},
        'retrieve_memory': {'arguments': {'key': 'test_key', 'category': 'test'}},
        'memory_retrieve_semantic': {'arguments': {'query': 'test semantic query'}},
        
        # File operations
        'list_directory': {'arguments': {'path': '.', 'depth': 1}},
        'read_file_range': {'arguments': {'file_path': 'server.py', 'start_line': 1, 'end_line': 5}},
        
        # Code analysis
        'analyze_code': {'arguments': {'code': 'def hello(): return "world"', 'analysis_type': 'bugs'}},
        'suggest_improvements': {'arguments': {'code': 'def hello(): return "world"'}},
        'generate_tests': {'arguments': {'code': 'def add(a, b): return a + b'}},
        
        # Execution tools
        'execute_code': {'arguments': {'code': 'print("Hello from test")', 'language': 'python'}},
        
        # Agent team tools
        'agent_team_plan_and_code': {'arguments': {'task': 'Create a simple hello world function', 'apply_changes': False}},
        
        # Router tools
        'router_diagnostics': {'arguments': {'limit': 5}},
        'router_test': {'arguments': {'task': 'simple test task'}},
        
        # Smart tools
        'smart_task': {'arguments': {'instruction': 'list current directory', 'dry_run': True}},
        'smart_plan_execute': {'arguments': {'instruction': 'test plan execution', 'dry_run': True}},
        'tool_match': {'arguments': {'task': 'analyze code'}},

        # Task management
        'get_task_status': {'arguments': {'task_id': 'test_task_123'}},

        # Research tools
        'web_search': {'arguments': {'query': 'test search'}},

        # Chat tools
        'chat_with_tools': {'arguments': {'instruction': 'Say hello', 'tool_choice': 'none', 'max_iters': 1}},
    }
    
    # Run tests
    results = []
    failed_tools = []
    
    for i, tool_name in enumerate(sorted(all_tools), 1):
        print(f"🧪 Testing {i:2d}/{len(all_tools)}: {tool_name}")
        
        config = test_configs.get(tool_name, {'arguments': {}})
        result = test_tool(tool_name, **config)
        results.append(result)
        
        if result['success']:
            print(f"   ✅ SUCCESS ({result['duration']}s)")
        else:
            print(f"   ❌ FAILED: {result['error']}")
            failed_tools.append(tool_name)
        
        # Brief pause to avoid overwhelming the system
        time.sleep(0.1)
    
    # Summary
    print()
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful = len([r for r in results if r['success']])
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ Failed: {len(failed_tools)}/{total}")
    
    if failed_tools:
        print(f"\n🚨 FAILED TOOLS:")
        for tool in failed_tools:
            result = next(r for r in results if r['tool'] == tool)
            print(f"   • {tool}: {result['error']}")
    
    # Performance summary
    print(f"\n⏱️  PERFORMANCE:")
    avg_duration = sum(r['duration'] for r in results) / len(results)
    max_duration = max(r['duration'] for r in results)
    slow_tools = [r for r in results if r['duration'] > 2.0]
    
    print(f"   Average: {avg_duration:.3f}s")
    print(f"   Maximum: {max_duration:.3f}s")
    if slow_tools:
        print(f"   Slow tools (>2s): {[r['tool'] for r in slow_tools]}")
    
    return len(failed_tools) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
