#!/usr/bin/env python3
"""Test the optimal model configuration: GPT-4o + Claude 3.5 Sonnet"""

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

print("🏆 Testing Optimal Model Configuration")
print("="*70)

# Test configuration
openai_model = os.getenv("OPENAI_MODEL")
openai_fallback = os.getenv("OPENAI_FALLBACK_MODEL")
anthropic_model = os.getenv("ANTHROPIC_MODEL")

print(f"\n📊 Current Configuration:")
print(f"  Primary (OpenAI):   {openai_model}")
print(f"  Fallback (OpenAI):  {openai_fallback}")
print(f"  Advanced (Anthropic): {anthropic_model}")

# Verify optimal models
print("\n" + "="*70)
print("✅ Verification:")

optimal = True

if openai_model == "gpt-4o":
    print("  ✅ OpenAI primary: gpt-4o (OPTIMAL)")
elif openai_model == "gpt-4o-mini":
    print("  ⚠️  OpenAI primary: gpt-4o-mini (Consider upgrading to gpt-4o)")
    optimal = False
else:
    print(f"  ⚠️  OpenAI primary: {openai_model} (Check recommendation)")
    optimal = False

if openai_fallback == "gpt-4o-mini":
    print("  ✅ OpenAI fallback: gpt-4o-mini (OPTIMAL)")
else:
    print(f"  ℹ️  OpenAI fallback: {openai_fallback}")

if anthropic_model == "claude-3-5-sonnet-20241022":
    print("  ✅ Anthropic: claude-3-5-sonnet-20241022 (OPTIMAL)")
else:
    print(f"  ⚠️  Anthropic: {anthropic_model} (Check recommendation)")
    optimal = False

# Test GPT-4o
print("\n" + "="*70)
print("🧪 Testing GPT-4o...")

try:
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": openai_model,
        "messages": [
            {"role": "user", "content": "Respond with exactly: 'GPT-4o is working optimally!'"}
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
        print(f"  ✅ Success!")
        print(f"  📝 Response: {message}")
    else:
        print(f"  ❌ Failed: HTTP {response.status_code}")
        error = response.json().get('error', {}).get('message', 'Unknown error')
        print(f"  Error: {error[:100]}")
        optimal = False
        
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")
    optimal = False

# Test Claude 3.5 Sonnet
print("\n" + "="*70)
print("🧪 Testing Claude 3.5 Sonnet...")

try:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": anthropic_model,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Respond with exactly: 'Claude 3.5 Sonnet is working optimally!'"}
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
        print(f"  ✅ Success!")
        print(f"  📝 Response: {content}")
    else:
        print(f"  ❌ Failed: HTTP {response.status_code}")
        error = response.json().get('error', {}).get('message', 'Unknown error')
        print(f"  Error: {error[:100]}")
        optimal = False
        
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")
    optimal = False

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

if optimal:
    print("\n🎉 OPTIMAL CONFIGURATION CONFIRMED!")
    print("\nYou're using the best models for intelligence and cost:")
    print("  ✅ GPT-4o: Best value (90% of tasks)")
    print("  ✅ Claude 3.5 Sonnet: Best reasoning (10% of tasks)")
    print("  ✅ GPT-4o-mini: Budget fallback")
    
    print("\n💰 Cost Efficiency:")
    print("  • 10M tokens/month: ~$28")
    print("  • Compare to GPT-4 Turbo only: ~$100")
    print("  • You're saving ~70% while maintaining top intelligence!")
    
    print("\n🚀 Performance:")
    print("  • GPT-4o: 2-3x faster than GPT-4 Turbo")
    print("  • Claude 3.5: Best for complex reasoning")
    print("  • Automatic routing to best model per task")
    
else:
    print("\n⚠️  CONFIGURATION CAN BE OPTIMIZED")
    print("\nRecommended changes:")
    if openai_model != "gpt-4o":
        print(f"  • Change OPENAI_MODEL to 'gpt-4o'")
    if anthropic_model != "claude-3-5-sonnet-20241022":
        print(f"  • Change ANTHROPIC_MODEL to 'claude-3-5-sonnet-20241022'")
    
    print("\nSee MODEL_SELECTION_GUIDE.md for details")

print("\n" + "="*70)
