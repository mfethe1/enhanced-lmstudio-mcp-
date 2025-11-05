#!/usr/bin/env python3
"""
Test direct requests to LM Studio with explicit timeouts
"""

import requests
import time

def test_direct_requests():
    """Test direct requests with explicit timeout"""
    print("🔧 Testing Direct Requests with Explicit Timeout")
    print("=" * 50)
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print("Testing with 5-second timeout...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(2, 5)  # (connect_timeout, read_timeout)
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Request completed in {elapsed:.1f} seconds")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                print(f"Response: '{content}'")
                if content:
                    print("✅ Got non-empty response!")
                    return "success"
                else:
                    print("❌ Got empty response (this is the original issue)")
                    return "empty_response"
            else:
                print("❌ No choices in response")
                return "no_choices"
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return "http_error"
            
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start_time
        print(f"⏰ Request timed out after {elapsed:.1f} seconds: {e}")
        print("✅ This is expected behavior - timeout is working!")
        return "timeout"
        
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start_time
        print(f"🔌 Connection error after {elapsed:.1f} seconds: {e}")
        print("❌ LM Studio may not be running")
        return "connection_error"
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Unexpected error after {elapsed:.1f} seconds: {e}")
        return "other_error"

def test_fallback_providers():
    """Test fallback to OpenAI/Anthropic"""
    print("\n🔄 Testing Fallback Providers")
    print("=" * 50)
    
    # Test OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("Testing OpenAI fallback...")
        try:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello, respond with 'Hi there!'"}],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                timeout=(5, 10)
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                print(f"✅ OpenAI works: '{content}'")
                return "openai_works"
            else:
                print(f"❌ OpenAI error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ OpenAI failed: {e}")
    
    # Test Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("Testing Anthropic fallback...")
        try:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello, respond with 'Hi there!'"}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Authorization": f"Bearer {anthropic_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=(5, 10)
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"].strip()
                print(f"✅ Anthropic works: '{content}'")
                return "anthropic_works"
            else:
                print(f"❌ Anthropic error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Anthropic failed: {e}")
    
    return "no_fallback"

if __name__ == "__main__":
    import os
    
    # Test LM Studio
    lm_result = test_direct_requests()
    
    # Test fallback providers
    fallback_result = test_fallback_providers()
    
    print(f"\n📊 Results Summary:")
    print(f"  LM Studio: {lm_result}")
    print(f"  Fallback: {fallback_result}")
    
    if lm_result == "timeout" and fallback_result in ["openai_works", "anthropic_works"]:
        print("\n🎯 PERFECT: This is exactly what our retry logic should handle!")
        print("✅ LM Studio times out → fallback providers work")
    elif lm_result == "empty_response" and fallback_result in ["openai_works", "anthropic_works"]:
        print("\n🎯 GOOD: This is the original issue our retry logic fixes!")
        print("✅ LM Studio returns empty → fallback providers work")
    else:
        print(f"\n⚠️  Need to investigate: LM Studio={lm_result}, Fallback={fallback_result}")
