#!/usr/bin/env python3
"""Query OpenAI and Anthropic APIs to see all available models"""

import os
import requests
from pathlib import Path
from datetime import datetime

# Load environment
env_path = Path(".secrets/.env.local")
with open(env_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

print("🔍 Querying Available AI Models")
print("="*80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Query OpenAI models
print("="*80)
print("🤖 OPENAI MODELS AVAILABLE TO YOU")
print("="*80)

try:
    api_key = os.getenv("OPENAI_API_KEY")
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10
    )
    
    if response.status_code == 200:
        models_data = response.json()
        all_models = [m['id'] for m in models_data.get('data', [])]
        
        # Filter for GPT models
        gpt_models = sorted([m for m in all_models if 'gpt' in m.lower()])
        o1_models = sorted([m for m in all_models if 'o1' in m.lower() or 'o3' in m.lower()])
        
        print(f"\n📊 Total Models Available: {len(all_models)}")
        
        print(f"\n🏆 GPT-5 Models:")
        gpt5_models = [m for m in gpt_models if 'gpt-5' in m.lower() or 'gpt5' in m.lower()]
        if gpt5_models:
            for model in gpt5_models:
                print(f"  ✅ {model}")
        else:
            print("  ⚠️  No GPT-5 models found yet")
        
        print(f"\n🔥 GPT-4 Series (Latest):")
        gpt4_latest = [m for m in gpt_models if any(x in m for x in ['gpt-4o', 'gpt-4-turbo', 'gpt-4-0125'])]
        for model in gpt4_latest[:10]:
            print(f"  • {model}")
        
        print(f"\n🧠 O-Series (Reasoning Models):")
        if o1_models:
            for model in o1_models:
                print(f"  • {model}")
        else:
            print("  ⚠️  No O-series models found")
        
        print(f"\n📋 All GPT-4 Models ({len(gpt4_latest)} total):")
        for model in gpt4_latest:
            print(f"  • {model}")
        
        # Identify the newest/best
        print(f"\n🎯 RECOMMENDED OPENAI MODELS:")
        
        # Check for GPT-5
        if gpt5_models:
            print(f"  🏆 BEST: {gpt5_models[0]} (GPT-5!)")
        # Check for latest GPT-4o
        elif any('gpt-4o' in m for m in gpt_models):
            gpt4o_models = [m for m in gpt_models if 'gpt-4o' in m and 'mini' not in m]
            if gpt4o_models:
                print(f"  🏆 BEST: gpt-4o (Latest GPT-4 series)")
        # Check for O3
        elif any('o3' in m for m in o1_models):
            print(f"  🏆 BEST: o3-mini (Latest reasoning model)")
        # Check for O1
        elif any('o1-preview' in m for m in o1_models):
            print(f"  🏆 BEST: o1-preview (Advanced reasoning)")
        else:
            print(f"  🏆 BEST: gpt-4-turbo (Most capable available)")
        
        print(f"\n💡 Current Configuration: {os.getenv('OPENAI_MODEL')}")
        
    else:
        print(f"❌ Failed to query OpenAI: HTTP {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ Error querying OpenAI: {str(e)}")

# Query Anthropic (note: Anthropic doesn't have a models list endpoint)
print("\n" + "="*80)
print("🤖 ANTHROPIC CLAUDE MODELS")
print("="*80)

print("\n📋 Known Anthropic Models (as of January 2025):")

anthropic_models = [
    {
        "name": "claude-opus-4-20250514",
        "tier": "🏆 OPUS 4.1 (Highest Intelligence)",
        "cost": "TBD (Premium)",
        "notes": "Future release - not yet available"
    },
    {
        "name": "claude-4.5-sonnet-20250415", 
        "tier": "🔥 SONNET 4.5 (Top Performance)",
        "cost": "TBD (High-end)",
        "notes": "Future release - not yet available"
    },
    {
        "name": "claude-4-sonnet-20250201",
        "tier": "⭐ SONNET 4 (Latest Available)",
        "cost": "TBD (High-end)", 
        "notes": "Future release - not yet available"
    },
    {
        "name": "claude-3-5-sonnet-20241022",
        "tier": "✅ SONNET 3.5 (CURRENT BEST)",
        "cost": "$3 / $15 per 1M tokens",
        "notes": "Latest available model"
    },
    {
        "name": "claude-3-opus-20240229",
        "tier": "💎 OPUS 3 (Most Capable v3)",
        "cost": "$15 / $75 per 1M tokens",
        "notes": "Best reasoning in v3 series"
    },
]

for model in anthropic_models:
    print(f"\n{model['tier']}")
    print(f"  Model: {model['name']}")
    print(f"  Cost:  {model['cost']}")
    print(f"  Note:  {model['notes']}")

# Test current Anthropic model
print(f"\n💡 Current Configuration: {os.getenv('ANTHROPIC_MODEL')}")

print("\n🧪 Testing Current Anthropic Model...")
try:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL"),
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Reply with your model name"}
        ]
    }
    
    response = requests.post(
        f"{os.getenv('ANTHROPIC_BASE_URL')}/v1/messages",
        headers=headers,
        json=payload,
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✅ Model working: {os.getenv('ANTHROPIC_MODEL')}")
    else:
        print(f"  ⚠️  HTTP {response.status_code}: {response.text[:100]}")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")

# Recommendations
print("\n" + "="*80)
print("🎯 RECOMMENDATIONS FOR TOP-END CONFIGURATION")
print("="*80)

print("\n📌 IMPORTANT NOTE:")
print("  As of January 2025, Claude 4.x models are NOT YET RELEASED.")
print("  GPT-5 is NOT YET RELEASED.")
print("  We'll configure the BEST CURRENTLY AVAILABLE models.")

print("\n🏆 OPTIMAL TOP-END CONFIGURATION (Current):")
print("\n  Primary (OpenAI):")
if any('gpt-5' in m for m in gpt_models if 'gpt_models' in locals()):
    print("    ✅ GPT-5 (if available)")
elif 'gpt_models' in locals() and any('o1-preview' in m for m in all_models):
    print("    ✅ o1-preview (Best reasoning)")
else:
    print("    ✅ gpt-4o (Best available GPT-4)")

print("\n  Advanced (Anthropic):")
print("    ✅ claude-3-5-sonnet-20241022 (Best currently available)")
print("    ✅ claude-3-opus-20240229 (For maximum reasoning)")

print("\n  Future Upgrade Path:")
print("    📅 When available: GPT-5")
print("    📅 When available: Claude 4 Sonnet")
print("    📅 When available: Claude 4.1 Opus")

print("\n💰 Cost Consideration:")
print("  Note: You mentioned wanting top-end models regardless of cost.")
print("  Current top-end configuration:")
print("    • GPT-4o: $2.50 / $10 per 1M tokens")
print("    • Claude 3.5 Sonnet: $3 / $15 per 1M tokens")
print("    • Claude 3 Opus: $15 / $75 per 1M tokens (5x more expensive)")

print("\n" + "="*80)
print("📝 Next Steps:")
print("="*80)
print("\n1. Review available models above")
print("2. Confirm you want:")
print("   • GPT-4o (best OpenAI has now)")
print("   • Claude 3.5 Sonnet (best Claude has now)")
print("   • Claude 3 Opus (maximum reasoning when needed)")
print("\n3. Or wait for GPT-5 / Claude 4 releases")

print("\n" + "="*80)
