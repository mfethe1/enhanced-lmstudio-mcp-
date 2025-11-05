#!/usr/bin/env python3
"""
Direct API test to check what Anthropic models actually work
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('ANTHROPIC_API_KEY')

print("🔍 Testing Anthropic API Directly")
print("=" * 80)
print(f"API Key: {api_key[:20]}...")
print()

# Test different model names
models_to_test = [
    # Claude 4.x models you mentioned
    "claude-sonnet-4.0",
    "claude-4.5-sonnet",
    "claude-opus-4.0",
    "claude-opus-4.1",
    # Current known models
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

for model in models_to_test:
    print(f"Testing: {model}...", end=" ", flush=True)
    
    payload = {
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hi"}]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ WORKS!")
        else:
            error_data = response.json()
            error_type = error_data.get('error', {}).get('type', 'unknown')
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            print(f"❌ {response.status_code} - {error_type}: {error_msg[:60]}")
    except Exception as e:
        print(f"❌ Exception: {str(e)[:60]}")

print()
print("=" * 80)
