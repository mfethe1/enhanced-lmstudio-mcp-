#!/usr/bin/env python3
"""
Comprehensive discovery of all Anthropic Claude models
"""

import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('ANTHROPIC_API_KEY')

print("🔍 COMPREHENSIVE ANTHROPIC MODEL DISCOVERY")
print("=" * 80)
print()

# Try to get model list from multiple endpoints
endpoints_to_try = [
    "/v1/models",
    "/v1/complete/models",
    "/v1/messages/models",
]

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

print("📡 Attempting to query models endpoints...")
print()

for endpoint in endpoints_to_try:
    url = f"https://api.anthropic.com{endpoint}"
    print(f"Trying: {url}...", end=" ")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ SUCCESS! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"   Response: {response.text[:100]}")
    except Exception as e:
        print(f"❌ {str(e)[:60]}")
    print()

print("=" * 80)
print("🧪 TESTING ALL POSSIBLE MODEL NAMING PATTERNS")
print("=" * 80)
print()

# Comprehensive list of all possible model names to test
all_possible_models = [
    # Claude 4.x Sonnet variants
    "claude-4-sonnet",
    "claude-4-sonnet-20250201",
    "claude-4-sonnet-20250415",
    "claude-4-sonnet-20250514",
    "claude-4.0-sonnet",
    "claude-4.0-sonnet-20250201",
    "claude-4.5-sonnet",
    "claude-4.5-sonnet-20250415",
    "claude-sonnet-4",
    "claude-sonnet-4.0",
    "claude-sonnet-4.5",
    "sonnet-4",
    "sonnet-4.0",
    "sonnet-4.5",
    
    # Claude 4.x Opus variants
    "claude-4-opus",
    "claude-4-opus-20250201",
    "claude-4-opus-20250415",
    "claude-4-opus-20250514",
    "claude-4.0-opus",
    "claude-4.0-opus-20250201",
    "claude-4.1-opus",
    "claude-4.1-opus-20250514",
    "claude-opus-4",
    "claude-opus-4.0",
    "claude-opus-4.1",
    "opus-4",
    "opus-4.0",
    "opus-4.1",
    
    # Claude 3.7 (between 3.5 and 4?)
    "claude-3.7-sonnet",
    "claude-3-7-sonnet",
    "claude-3.7-sonnet-20241201",
    
    # Claude 3.5 updates
    "claude-3-5-sonnet-20250101",
    "claude-3-5-sonnet-20250201",
    "claude-3-5-sonnet-20250301",
    "claude-3.5-sonnet-latest",
    "claude-sonnet-3.5-v2",
    
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-20250101",
    "claude-3.5-haiku",
    
    # Maybe they renamed to just "sonnet" and "opus"?
    "sonnet",
    "sonnet-latest",
    "opus",
    "opus-latest",
    "claude-sonnet",
    "claude-opus",
    "claude-sonnet-latest",
    "claude-opus-latest",
    
    # Current known working models
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

working_models = []
message_payload = {
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
}

print(f"Testing {len(all_possible_models)} possible model names...")
print()

for i, model in enumerate(all_possible_models, 1):
    print(f"[{i:3d}/{len(all_possible_models)}] {model:45s} ", end="", flush=True)
    
    payload = {**message_payload, "model": model}
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ WORKS!")
            working_models.append(model)
        elif response.status_code == 404:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', '')
            if 'model:' in error_msg:
                print("❌ Not found")
            else:
                print(f"❌ 404: {error_msg[:40]}")
        elif response.status_code == 400:
            print("❌ Invalid request")
        elif response.status_code == 429:
            print("⏸️  Rate limited")
        else:
            print(f"❌ {response.status_code}")
    except requests.exceptions.Timeout:
        print("⏱️  Timeout")
    except Exception as e:
        print(f"❌ {str(e)[:30]}")

print()
print("=" * 80)
print("📊 DISCOVERY RESULTS")
print("=" * 80)
print()

if working_models:
    print(f"✅ FOUND {len(working_models)} WORKING MODELS:")
    print()
    for model in working_models:
        print(f"   ✓ {model}")
else:
    print("❌ No models found")

print()
print("=" * 80)
print("💡 CHECKING ANTHROPIC DOCUMENTATION")
print("=" * 80)
print()

# Try to scrape documentation or check status page
try:
    print("Checking Anthropic documentation page...")
    doc_response = requests.get(
        "https://docs.anthropic.com/en/docs/about-claude/models",
        timeout=10
    )
    if "claude-4" in doc_response.text.lower():
        print("✅ Found references to Claude 4 in documentation!")
        # Try to extract model names
        import re
        model_pattern = r'claude-[\w\-\.]+'
        found_models = re.findall(model_pattern, doc_response.text.lower())
        unique_models = sorted(set(found_models))
        print("   Models mentioned in docs:")
        for m in unique_models[:20]:
            print(f"      • {m}")
    else:
        print("❌ No Claude 4 references found in documentation")
except Exception as e:
    print(f"⚠️  Could not check documentation: {str(e)[:60]}")

print()
print("=" * 80)
