#!/usr/bin/env python3
"""
Test script to verify Claude 4.x model availability
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Models you requested to test
models_to_test = [
    "claude-sonnet-4.0",
    "claude-sonnet-4-0",
    "claude-4-sonnet",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4.5",
    "claude-sonnet-4-5",
    "claude-4.5-sonnet",
    "claude-4.5-sonnet-20250415",
    "claude-opus-4.0",
    "claude-opus-4-0", 
    "claude-4-opus",
    "claude-4-opus-20250514",
    "claude-opus-4.1",
    "claude-opus-4-1",
    "claude-4.1-opus",
    "claude-opus-4-20250514",
    # Try some alternative naming patterns
    "claude-4.0-sonnet-20250201",
    "claude-4-0-sonnet-20250201",
    "sonnet-4.0",
    "sonnet-4.5",
    "opus-4.0",
    "opus-4.1",
]

print("🧪 Testing Claude 4.x Model Availability")
print("=" * 80)
print()

working_models = []
failed_models = []

for model in models_to_test:
    try:
        print(f"Testing: {model}...", end=" ")
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ WORKING!")
        working_models.append(model)
    except Exception as e:
        error_msg = str(e)
        if "model:" in error_msg.lower() or "not found" in error_msg.lower():
            print(f"❌ Not available")
        else:
            print(f"❌ Error: {error_msg[:50]}")
        failed_models.append((model, error_msg))

print()
print("=" * 80)
print(f"📊 RESULTS")
print("=" * 80)
print()

if working_models:
    print(f"✅ WORKING MODELS ({len(working_models)}):")
    for model in working_models:
        print(f"   • {model}")
else:
    print("❌ NO CLAUDE 4.x MODELS ARE AVAILABLE YET")

print()
print(f"❌ FAILED MODELS ({len(failed_models)}):")
for model, _ in failed_models[:5]:  # Show first 5
    print(f"   • {model}")

print()
print("=" * 80)
print("🔍 Testing Currently Known Working Models")
print("=" * 80)
print()

known_working = [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

for model in known_working:
    try:
        print(f"Testing: {model}...", end=" ")
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ WORKING")
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")

print()
print("=" * 80)
