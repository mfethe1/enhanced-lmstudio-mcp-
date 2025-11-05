#!/usr/bin/env python3
"""Quick test of new OpenAI API key"""

import os
import requests
from pathlib import Path

# Load environment
env_path = Path(".secrets/.env.local")
with open(env_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

print("🔍 Testing New OpenAI API Key\n")
print("="*60)

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Mask key for display
masked_key = api_key[:15] + "..." + api_key[-10:] if len(api_key) > 25 else "***"
print(f"API Key: {masked_key}")
print(f"Model: {model}")
print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")

print("\n" + "="*60)
print("Test 1: Listing available models...\n")

try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        f"{os.getenv('OPENAI_BASE_URL')}/models",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        models = response.json()
        count = len(models.get('data', []))
        print(f"✅ SUCCESS: Listed {count} models")
        
        # Check if our model is available
        model_ids = [m.get('id') for m in models.get('data', [])]
        if model in model_ids:
            print(f"✅ Model '{model}' is available")
        else:
            print(f"⚠️  Model '{model}' not found, but API key works!")
    else:
        print(f"❌ FAILED: HTTP {response.status_code}")
        print(f"Response: {response.text[:200]}")
        exit(1)
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    exit(1)

print("\n" + "="*60)
print("Test 2: Chat completion test...\n")

try:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'OpenAI API is working!' and nothing else."}
        ],
        "max_tokens": 50,
        "temperature": 0
    }
    
    response = requests.post(
        f"{os.getenv('OPENAI_BASE_URL')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"✅ SUCCESS: Chat completion works!")
        print(f"📝 Response: {message}")
        print(f"\n🎉 OpenAI API is fully operational!")
    else:
        print(f"❌ FAILED: HTTP {response.status_code}")
        error_data = response.json() if response.text else {}
        error_msg = error_data.get('error', {}).get('message', response.text[:200])
        print(f"Error: {error_msg}")
        
        if response.status_code == 429:
            print("\n⚠️  Quota exceeded - check billing")
        elif response.status_code == 401:
            print("\n⚠️  Authentication failed - check API key")
        
        exit(1)
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    exit(1)

print("\n" + "="*60)
print("🎯 ALL TESTS PASSED!")
print("="*60)
print("\nYour OpenAI API key is working perfectly.")
print("The MCP server will now use OpenAI when needed.")
