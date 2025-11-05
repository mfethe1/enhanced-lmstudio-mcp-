#!/usr/bin/env python3
"""
Test external integrations like web_search and research tools.
"""

import json
import os
import sys
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def test_web_search():
    """Test web_search functionality"""
    print("=" * 80)
    print("TESTING: web_search")
    print("=" * 80)
    
    # Set environment variables
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "LMSTUDIO_MODEL": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Test web_search tool
    web_search_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {
                "query": "Python PLIP protein ligand interaction",
                "max_depth": 1,
                "time_limit": 30
            }
        }
    }
    
    print("Testing web_search with PLIP query...")
    
    try:
        response = server.handle_message(web_search_msg)
        
        print(f"Response structure:")
        print(f"- jsonrpc: {response.get('jsonrpc')}")
        print(f"- id: {response.get('id')}")
        print(f"- has result: {'result' in response}")
        print(f"- has error: {'error' in response}")
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got web_search result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for research content
                if 'final_analysis' in result:
                    analysis = str(result['final_analysis'])
                    print(f"✓ Final analysis ({len(analysis)} chars): {analysis[:200]}...")
                
                if 'research_id' in result:
                    print(f"✓ Research ID: {result['research_id']}")
                
                if 'sources' in result:
                    sources = result['sources']
                    print(f"✓ Sources found: {len(sources) if isinstance(sources, list) else 'N/A'}")
            
            return True
        
        elif 'error' in response:
            error = response['error']
            print(f"Error: {error.get('message', '')[:200]}...")
            
            # Some errors are expected (e.g., API key missing)
            if 'API key' in str(error.get('message', '')):
                print("✓ Expected error - API key configuration needed")
                return True
            
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_health_check_with_providers():
    """Test health_check with provider probing"""
    print("\n" + "=" * 80)
    print("TESTING: health_check with provider probing")
    print("=" * 80)
    
    health_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "health_check",
            "arguments": {
                "probe_providers": True
            }
        }
    }
    
    print("Testing health_check with provider probing...")
    
    try:
        response = server.handle_message(health_msg)
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got health_check result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for provider status
                if 'providers' in result:
                    providers = result['providers']
                    print(f"✓ Provider status: {providers}")
                
                if 'lmstudio_status' in result:
                    lm_status = result['lmstudio_status']
                    print(f"✓ LM Studio status: {lm_status}")
                
                if 'tools_count' in result:
                    tools_count = result['tools_count']
                    print(f"✓ Tools count: {tools_count}")
            
            return True
        
        elif 'error' in response:
            error = response['error']
            print(f"Error: {error.get('message', '')[:200]}...")
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_router_diagnostics():
    """Test router diagnostics functionality"""
    print("\n" + "=" * 80)
    print("TESTING: router_diagnostics")
    print("=" * 80)
    
    router_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "router_diagnostics",
            "arguments": {
                "limit": 5
            }
        }
    }
    
    print("Testing router_diagnostics...")
    
    try:
        response = server.handle_message(router_msg)
        
        if 'result' in response:
            result = response['result']
            print(f"✓ Got router_diagnostics result")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for diagnostic info
                if 'recent_decisions' in result:
                    decisions = result['recent_decisions']
                    print(f"✓ Recent decisions: {len(decisions) if isinstance(decisions, list) else 'N/A'}")
                
                if 'backend_usage' in result:
                    usage = result['backend_usage']
                    print(f"✓ Backend usage: {usage}")
                
                if 'avg_latency' in result:
                    latency = result['avg_latency']
                    print(f"✓ Average latency: {latency}")
            
            return True
        
        elif 'error' in response:
            error = response['error']
            print(f"Error: {error.get('message', '')[:200]}...")
            return False
        
        return False
        
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def main():
    """Run external integration tests"""
    print("External Integration Test Suite")
    print("=" * 80)
    
    tests = [
        test_web_search,
        test_health_check_with_providers,
        test_router_diagnostics
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
    print("EXTERNAL INTEGRATION TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_func.__name__}")
    
    if passed == total:
        print("\n🎉 External integration tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
