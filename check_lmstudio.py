#!/usr/bin/env python3
"""
Quick check of LM Studio status and configuration
"""
import requests
import json
import os

def check_lmstudio():
    print("🔍 Checking LM Studio Status...")
    print("=" * 50)
    
    try:
        # Check if LM Studio is running
        r = requests.get('http://localhost:1234/v1/models', timeout=5)
        print("✅ LM Studio Status: RUNNING")
        
        models = r.json().get('data', [])
        print(f"📊 Available models: {len(models)}")
        
        if models:
            print("🎯 Top 3 models:")
            for i, model in enumerate(models[:3]):
                print(f"  {i+1}. {model.get('id', 'unknown')}")
        else:
            print("❌ No models loaded!")
            
    except requests.exceptions.ConnectRefused:
        print("❌ LM Studio Status: NOT RUNNING (Connection refused)")
        return False
    except requests.exceptions.Timeout:
        print("❌ LM Studio Status: TIMEOUT (>5s)")
        return False
    except Exception as e:
        print(f"❌ LM Studio Status: ERROR - {e}")
        return False
    
    # Test a simple chat completion
    print("\n🧪 Testing Chat Completion...")
    try:
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Say 'Hello' in one word"}],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        r = requests.post(
            'http://localhost:1234/v1/chat/completions',
            json=payload,
            timeout=30  # Generous timeout for testing
        )
        
        if r.status_code == 200:
            data = r.json()
            choices = data.get('choices', [])
            if choices:
                content = choices[0].get('message', {}).get('content', '').strip()
                print(f"✅ Response: '{content}'")
                if content:
                    print("✅ LM Studio is working correctly!")
                    return True
                else:
                    print("❌ Got empty response")
            else:
                print("❌ No choices in response")
        else:
            print(f"❌ HTTP {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        
    return False

def check_environment():
    print("\n🔧 Environment Configuration:")
    print("=" * 50)
    
    configs = {
        'HTTP_READ_TIMEOUT_SIMPLE': '60',
        'HTTP_READ_TIMEOUT_COMPLEX': '180', 
        'NO_FALLBACK_PROVIDERS': '0',
        'OPENAI_API_KEY': 'NOT SET',
        'ANTHROPIC_API_KEY': 'NOT SET',
        'CIRCUIT_LMSTUDIO_THRESHOLD': '8',
        'CIRCUIT_LMSTUDIO_RECOVERY': '120'
    }
    
    for key, default in configs.items():
        value = os.getenv(key, default)
        if 'API_KEY' in key:
            value = 'SET' if os.getenv(key) else 'NOT SET'
        print(f"  {key}: {value}")

if __name__ == "__main__":
    lm_working = check_lmstudio()
    check_environment()
    
    print("\n" + "=" * 50)
    if lm_working:
        print("🎉 LM Studio is ready for MCP use!")
    else:
        print("⚠️  LM Studio needs attention before MCP will work properly")
        print("\n💡 Recommendations:")
        print("  1. Start LM Studio application")
        print("  2. Load a model (e.g., openai/gpt-oss-20b)")
        print("  3. Ensure server is running on localhost:1234")
