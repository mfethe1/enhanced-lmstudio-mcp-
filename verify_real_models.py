#!/usr/bin/env python3
"""Verify which top models are actually available and working"""

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

print("🔍 Verifying Real Available Models\n" + "="*80)

api_key = os.getenv("OPENAI_API_KEY")

# Test different models with simple requests
test_models = [
    ("gpt-5", "GPT-5"),
    ("gpt-4o", "GPT-4o"),
    ("o1", "O1"),
    ("o1-pro", "O1-Pro"),
    ("o3", "O3"),
    ("o3-mini", "O3-Mini"),
    ("chatgpt-4o-latest", "ChatGPT-4o-Latest"),
]

print("\n🧪 Testing Models:")
working_models = []

for model_id, model_name in test_models:
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_completion_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✅ {model_name}: WORKING")
            working_models.append(model_id)
        else:
            error = response.json().get('error', {})
            error_code = error.get('code', '')
            error_msg = error.get('message', '')[:50]
            
            if 'model_not_found' in error_code or 'does not exist' in error_msg:
                print(f"  ❌ {model_name}: NOT AVAILABLE")
            elif 'parameter' in error_msg.lower():
                print(f"  ⚠️  {model_name}: Available but different parameters")
                working_models.append(model_id)
            else:
                print(f"  ⚠️  {model_name}: {error_msg}")
    except Exception as e:
        print(f"  ❌ {model_name}: Error - {str(e)[:50]}")

print("\n" + "="*80)
print("🏆 RECOMMENDED CONFIGURATION:")
print("="*80)

if "gpt-5" in working_models:
    print("\n✅ GPT-5 is available!")
    print("   Recommended: OPENAI_MODEL=gpt-5")
elif "o1-pro" in working_models:
    print("\n✅ O1-Pro is available (best reasoning)")
    print("   Recommended: OPENAI_MODEL=o1-pro")
elif "o1" in working_models:
    print("\n✅ O1 is available (advanced reasoning)")
    print("   Recommended: OPENAI_MODEL=o1")
elif "chatgpt-4o-latest" in working_models:
    print("\n✅ ChatGPT-4o-Latest is available")
    print("   Recommended: OPENAI_MODEL=chatgpt-4o-latest")
else:
    print("\n✅ GPT-4o is the best available")
    print("   Recommended: OPENAI_MODEL=gpt-4o")

print("\n📌 IMPORTANT NOTE:")
print("   The 'gpt-5' model ID in the models list may be a placeholder.")
print("   Until GPT-5 is officially released, use the best working model above.")

print("\n💎 For Anthropic:")
print("   • claude-3-5-sonnet-20241022 (Best currently available)")
print("   • claude-3-opus-20240229 (Maximum intelligence)")
print("   • Claude 4/4.5 NOT YET RELEASED")

print("\n" + "="*80)
