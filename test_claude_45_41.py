#!/usr/bin/env python3
"""
Test Claude 4.5 Sonnet and Opus 4.1
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('ANTHROPIC_API_KEY')

models = [
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (NEWEST)"),
    ("claude-opus-4-1-20250805", "Claude Opus 4.1"),
    ("claude-opus-4-20250514", "Claude Opus 4.0"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4.0"),
]

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

print("🧪 Testing Claude 4.5 and 4.1 Models")
print("=" * 80)
print()

for model_id, model_name in models:
    print(f"Testing: {model_name}")
    print(f"  Model ID: {model_id}")
    
    payload = {
        "model": model_id,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is your model name and version?"}]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', [{}])[0].get('text', '')
            print(f"  ✅ WORKS! Response: {content[:100]}")
        else:
            print(f"  ❌ Failed: {response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
    
    print()

print("=" * 80)
