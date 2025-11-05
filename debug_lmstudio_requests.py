#!/usr/bin/env python3
"""
Debug the actual LM Studio API requests to see what's going wrong
"""
import requests
import json
import time
import os

def test_raw_lmstudio_api():
    """Test the raw LM Studio API directly to see what works"""
    print("🔍 Testing Raw LM Studio API")
    print("=" * 50)
    
    base_url = "http://localhost:1234"
    
    # Test 1: Basic chat completion (what our MCP server should be sending)
    print("\n1. Testing basic chat completion...")
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print(f"   URL: {base_url}/v1/chat/completions")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code} ({elapsed:.1f}s)")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            
            # Check what we actually got
            choices = data.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '')
                reasoning = message.get('reasoning', '')
                
                print(f"\n   Analysis:")
                print(f"     Content: '{content}'")
                print(f"     Reasoning: '{reasoning[:100]}...' ({len(reasoning)} chars)")
                print(f"     Finish reason: {choices[0].get('finish_reason')}")
                
                if content.strip():
                    print(f"   ✅ SUCCESS: Got content")
                    return True
                elif reasoning.strip():
                    print(f"   ⚠️  Got reasoning but no content")
                    return True
                else:
                    print(f"   ❌ EMPTY: No content or reasoning")
            else:
                print(f"   ❌ NO CHOICES")
        else:
            print(f"   ❌ HTTP ERROR: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"   ⏰ TIMEOUT after 30s")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    return False

def test_completions_endpoint():
    """Test the completions endpoint"""
    print("\n2. Testing completions endpoint...")
    payload = {
        "model": "openai/gpt-oss-20b",
        "prompt": "What is 2+2? Answer:",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:1234/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code} ({elapsed:.1f}s)")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            
            choices = data.get('choices', [])
            if choices:
                text = choices[0].get('text', '')
                print(f"   Text: '{text}'")
                
                if text.strip():
                    print(f"   ✅ SUCCESS: Got text")
                    return True
                else:
                    print(f"   ❌ EMPTY: No text")
            else:
                print(f"   ❌ NO CHOICES")
        else:
            print(f"   ❌ HTTP ERROR: {response.text}")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    return False

def test_different_parameters():
    """Test different parameter combinations"""
    print("\n3. Testing different parameters...")
    
    test_cases = [
        {
            "name": "Minimal params",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        },
        {
            "name": "With stream=false",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False
            }
        },
        {
            "name": "Higher max_tokens",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "temperature": 0.1
            }
        },
        {
            "name": "With system message",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Always respond."},
                    {"role": "user", "content": "Hi"}
                ],
                "max_tokens": 50
            }
        }
    ]
    
    working_configs = []
    
    for test_case in test_cases:
        print(f"\n   Testing: {test_case['name']}")
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json=test_case['payload'],
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout
            )
            elapsed = time.time() - start_time
            
            print(f"     Status: {response.status_code} ({elapsed:.1f}s)")
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content', '').strip()
                    reasoning = message.get('reasoning', '').strip()
                    
                    if content:
                        print(f"     ✅ Content: '{content}'")
                        working_configs.append(test_case)
                    elif reasoning:
                        print(f"     ⚠️  Reasoning: '{reasoning[:50]}...'")
                        working_configs.append(test_case)
                    else:
                        print(f"     ❌ Empty response")
                else:
                    print(f"     ❌ No choices")
            else:
                print(f"     ❌ HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"     ⏰ Timeout after 60s")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    return working_configs

def compare_with_mcp_server():
    """Compare with what our MCP server is actually sending"""
    print("\n4. Comparing with MCP server requests...")
    
    # Import our server to see what it's actually doing
    import sys
    sys.path.append('.')
    
    try:
        from server import EnhancedLMStudioMCPServer
        server = EnhancedLMStudioMCPServer()
        
        print(f"   Server base_url: {server.base_url}")
        print(f"   Server model: {server.model_name}")
        
        # Try to make a request through the server
        print("\n   Testing through MCP server...")
        
        import asyncio
        
        async def test_mcp_request():
            try:
                result = await server.make_llm_request_with_retry(
                    "What is 2+2?",
                    temperature=0.1
                )
                print(f"   MCP Result: '{result}'")
                return result
            except Exception as e:
                print(f"   MCP Error: {e}")
                return None
        
        # Run the async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(test_mcp_request())
        
        if result and result.strip():
            print(f"   ✅ MCP server working")
        else:
            print(f"   ❌ MCP server not working")
            
    except Exception as e:
        print(f"   ❌ Failed to test MCP server: {e}")

def main():
    print("🚀 LM Studio API Request Debugging")
    print("=" * 60)
    
    # Test 1: Raw API
    chat_works = test_raw_lmstudio_api()
    
    # Test 2: Completions endpoint
    completions_works = test_completions_endpoint()
    
    # Test 3: Different parameters
    working_configs = test_different_parameters()
    
    # Test 4: Compare with MCP server
    compare_with_mcp_server()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING SUMMARY")
    print("=" * 60)
    
    print(f"Chat completions working: {chat_works}")
    print(f"Completions working: {completions_works}")
    print(f"Working configurations: {len(working_configs)}")
    
    if working_configs:
        print("\n✅ Working configurations found:")
        for config in working_configs:
            print(f"   • {config['name']}")
    
    if not chat_works and not completions_works:
        print("\n❌ PROBLEM IDENTIFIED:")
        print("   • LM Studio API is not responding properly")
        print("   • This explains why MCP tests are 'passing' but using fallbacks")
        print("\n💡 NEXT STEPS:")
        print("   1. Check if model is actually loaded in LM Studio")
        print("   2. Try restarting LM Studio")
        print("   3. Check LM Studio logs for errors")
        print("   4. Try a different model to isolate the issue")
    elif working_configs:
        print("\n✅ SOLUTION FOUND:")
        print("   • Use the working configuration in MCP server")
        print("   • Update request formatting to match working pattern")

if __name__ == "__main__":
    main()
