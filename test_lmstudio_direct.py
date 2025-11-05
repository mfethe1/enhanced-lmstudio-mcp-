#!/usr/bin/env python3
"""
Direct test of LM Studio to see if it's responding
"""

import requests
import json

def test_lmstudio_direct():
    """Test LM Studio directly with simple requests"""
    base_url = "http://localhost:1234"
    
    print("🔧 Testing LM Studio Direct Connection")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        print("1. Testing server health...")
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Available models: {len(models.get('data', []))}")
            for model in models.get('data', [])[:3]:  # Show first 3 models
                print(f"     - {model.get('id', 'unknown')}")
        else:
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to connect to LM Studio: {e}")
        print("   Make sure LM Studio is running on localhost:1234")
        return False
    
    # Test 2: Try multiple models to find one that works
    models_to_test = [
        "openai/gpt-oss-20b",
        "deepseek/deepseek-r1-0528-qwen3-8b",
        "text-embedding-nomic-embed-text-v1.5"
    ]

    working_model = None

    for model_name in models_to_test:
        try:
            print(f"\n2. Testing chat completion with {model_name}...")
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, respond with just 'Hi there!'"}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    print(f"   Response: '{content}'")
                    if content:
                        print(f"   ✅ {model_name} is working correctly!")
                        working_model = model_name
                        break
                    else:
                        print(f"   ❌ {model_name} returned empty content")
                else:
                    print("   ❌ No choices in response")
                    print(f"   Full response: {data}")
            else:
                print(f"   ❌ Error: {response.text}")

        except Exception as e:
            print(f"   ❌ Chat completion failed for {model_name}: {e}")
            continue

    if working_model:
        print(f"\n✅ Found working model: {working_model}")
        return working_model
    else:
        print("\n❌ No models are responding with content")
        return None

if __name__ == "__main__":
    working_model = test_lmstudio_direct()
    if working_model:
        print(f"\n🎉 LM Studio is working with model: {working_model}")
        print(f"💡 Update your mcp.json to use: {working_model}")
    else:
        print("\n❌ LM Studio has issues - fix these before testing chat_with_tools")
