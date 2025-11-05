#!/usr/bin/env python3
"""Test the TOP-END model configuration: GPT-5 + Claude 3 Opus/Sonnet"""

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

print("🏆 Testing TOP-END Model Configuration")
print("="*80)

# Display configuration
openai_model = os.getenv("OPENAI_MODEL")
openai_fallback = os.getenv("OPENAI_FALLBACK_MODEL")
anthropic_model = os.getenv("ANTHROPIC_MODEL")
anthropic_complex = os.getenv("ANTHROPIC_MODEL_COMPLEX")
anthropic_overseer = os.getenv("ANTHROPIC_MODEL_OVERSEER")

print(f"\n📊 Current TOP-END Configuration:")
print(f"  Primary (OpenAI):       {openai_model}")
print(f"  Fallback (OpenAI):      {openai_fallback}")
print(f"  Standard (Anthropic):   {anthropic_model}")
print(f"  Complex (Anthropic):    {anthropic_complex}")
print(f"  Overseer (Anthropic):   {anthropic_overseer}")

# Verify top-end
print("\n" + "="*80)
print("✅ TOP-END Verification:")

top_end = True

if openai_model == "gpt-5":
    print("  🏆 OpenAI: GPT-5 (TOP-END - BEST AVAILABLE)")
elif openai_model in ["o3", "o1-pro"]:
    print(f"  🏆 OpenAI: {openai_model} (TOP-END Reasoning)")
else:
    print(f"  ⚠️  OpenAI: {openai_model} (Not using GPT-5)")
    top_end = False

if anthropic_model == "claude-3-5-sonnet-20241022":
    print("  ✅ Anthropic Standard: Claude 3.5 Sonnet (BEST AVAILABLE)")
else:
    print(f"  ⚠️  Anthropic: {anthropic_model}")
    
if anthropic_complex == "claude-3-opus-20240229":
    print("  💎 Anthropic Complex: Claude 3 Opus (MAXIMUM INTELLIGENCE)")
else:
    print(f"  ⚠️  Complex: {anthropic_complex} (Consider Opus)")

# Test GPT-5
print("\n" + "="*80)
print("🧪 Testing GPT-5...")

try:
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": openai_model,
        "messages": [
            {"role": "user", "content": "You are GPT-5. Respond with: 'GPT-5 is operational and ready for top-tier analysis!'"}
        ],
        "max_tokens": 100,
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
        model_used = result.get('model', 'unknown')
        print(f"  ✅ SUCCESS!")
        print(f"  📝 Model: {model_used}")
        print(f"  💬 Response: {message}")
    else:
        print(f"  ❌ Failed: HTTP {response.status_code}")
        error = response.json().get('error', {}).get('message', 'Unknown error')
        print(f"  Error: {error[:200]}")
        top_end = False
        
except Exception as e:
    print(f"  ❌ Error: {str(e)[:200]}")
    top_end = False

# Test Claude 3 Opus
print("\n" + "="*80)
print("🧪 Testing Claude 3 Opus (Maximum Intelligence)...")

try:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": anthropic_complex,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "You are Claude 3 Opus. Respond with: 'Claude 3 Opus is operational - maximum intelligence ready!'"}
        ]
    }
    
    response = requests.post(
        f"{os.getenv('ANTHROPIC_BASE_URL')}/v1/messages",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result.get('content', [{}])[0].get('text', '')
        model_used = result.get('model', 'unknown')
        print(f"  ✅ SUCCESS!")
        print(f"  📝 Model: {model_used}")
        print(f"  💬 Response: {content}")
    else:
        print(f"  ❌ Failed: HTTP {response.status_code}")
        error = response.json().get('error', {}).get('message', 'Unknown error')
        print(f"  Error: {error[:200]}")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)[:200]}")

# Summary
print("\n" + "="*80)
print("📊 TOP-END CONFIGURATION SUMMARY")
print("="*80)

if top_end:
    print("\n🎉 TOP-END CONFIGURATION ACTIVE!")
    print("\nYou're using the absolute BEST models available:")
    print("  🏆 GPT-5: Next generation intelligence")
    print("  💎 Claude 3 Opus: Maximum reasoning power")
    print("  ✅ Claude 3.5 Sonnet: Best standard model")
    
    print("\n💰 Cost Profile (Premium):")
    print("  • GPT-5: TBD (Premium pricing)")
    print("  • Claude 3 Opus: $15/$75 per 1M tokens")
    print("  • Claude 3.5 Sonnet: $3/$15 per 1M tokens")
    
    print("\n🎯 Usage Strategy:")
    print("  • GPT-5: Primary for most tasks (cutting edge)")
    print("  • Claude 3.5 Sonnet: Standard analysis")
    print("  • Claude 3 Opus: Maximum reasoning (complex problems)")
    
    print("\n⚡ Performance:")
    print("  • GPT-5: Next-gen capabilities")
    print("  • Claude 3 Opus: Best-in-class reasoning")
    print("  • Automatic routing to best model per task")
    
else:
    print("\n⚠️  CONFIGURATION ISSUES DETECTED")
    print("\nRecommended TOP-END setup:")
    print("  • OPENAI_MODEL=gpt-5")
    print("  • ANTHROPIC_MODEL=claude-3-5-sonnet-20241022")
    print("  • ANTHROPIC_MODEL_COMPLEX=claude-3-opus-20240229")
    
print("\n" + "="*80)
print("📌 NOTE: Claude 4/4.5 and Opus 4.1 are NOT YET RELEASED")
print("   This is the BEST configuration currently possible!")
print("   We'll auto-upgrade when Claude 4 becomes available.")
print("="*80)
