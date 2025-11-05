#!/usr/bin/env python3
"""
Deep debug of openai/gpt-oss-20b empty response issue
"""
import requests
import json
import time

def get_full_response():
    """Get the complete response to see what's actually being returned"""
    print("🔍 Getting full response from openai/gpt-oss-20b...")
    print("=" * 60)
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Say the word 'hello'"}],
        "max_tokens": 20,
        "temperature": 0.1
    }
    
    try:
        r = requests.post(
            'http://localhost:1234/v1/chat/completions',
            json=payload,
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            print("📊 COMPLETE RESPONSE:")
            print(json.dumps(data, indent=2))
            
            # Analyze the response structure
            choices = data.get('choices', [])
            if choices:
                choice = choices[0]
                message = choice.get('message', {})
                content = message.get('content')
                
                print(f"\n🔍 DETAILED ANALYSIS:")
                print(f"   Choices count: {len(choices)}")
                print(f"   First choice: {json.dumps(choice, indent=2)}")
                print(f"   Message: {json.dumps(message, indent=2)}")
                print(f"   Content type: {type(content)}")
                print(f"   Content repr: {repr(content)}")
                print(f"   Content length: {len(content) if content else 'None'}")
                
                if content is not None:
                    print(f"   Content stripped: '{content.strip()}'")
                    print(f"   Content bytes: {content.encode('utf-8') if content else 'None'}")
                
                # Check finish reason
                finish_reason = choice.get('finish_reason')
                print(f"   Finish reason: {finish_reason}")
                
                return content
        else:
            print(f"❌ HTTP {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def test_different_prompts():
    """Test with different types of prompts to see if any work"""
    print("\n🧪 Testing different prompt types...")
    print("=" * 60)
    
    test_prompts = [
        "Hello",
        "Hi there",
        "Say 'test'",
        "1+1=",
        "Complete: The sky is",
        "Q: What is 2+2? A:",
        "Please respond with just the word 'yes'",
        "Output: hello",
        "Return the string 'working'",
        "Echo: test"
    ]
    
    for prompt in test_prompts:
        print(f"\n🔧 Testing prompt: '{prompt}'")
        
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0  # Deterministic
        }
        
        try:
            r = requests.post(
                'http://localhost:1234/v1/chat/completions',
                json=payload,
                timeout=30
            )
            
            if r.status_code == 200:
                data = r.json()
                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    finish_reason = choices[0].get('finish_reason')
                    
                    if content and content.strip():
                        print(f"   ✅ SUCCESS: '{content.strip()}' (reason: {finish_reason})")
                        return True
                    else:
                        print(f"   ❌ Empty (reason: {finish_reason})")
                else:
                    print(f"   ❌ No choices")
            else:
                print(f"   ❌ HTTP {r.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def test_completions_endpoint():
    """Test the /v1/completions endpoint instead of chat/completions"""
    print("\n🔧 Testing /v1/completions endpoint...")
    print("=" * 60)
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "prompt": "Hello, my name is",
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        r = requests.post(
            'http://localhost:1234/v1/completions',
            json=payload,
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            print("📊 Completions response:")
            print(json.dumps(data, indent=2))
            
            choices = data.get('choices', [])
            if choices:
                text = choices[0].get('text', '')
                if text and text.strip():
                    print(f"✅ Completions endpoint works: '{text.strip()}'")
                    return True
                else:
                    print(f"❌ Completions endpoint also returns empty")
            
        else:
            print(f"❌ Completions HTTP {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"❌ Completions error: {e}")
    
    return False

def check_model_parameters():
    """Check if we can get model parameters or configuration"""
    print("\n🔧 Checking model configuration...")
    print("=" * 60)
    
    # Try to get model details
    try:
        r = requests.get('http://localhost:1234/v1/models/openai/gpt-oss-20b', timeout=10)
        if r.status_code == 200:
            data = r.json()
            print("📊 Model details:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Model details not available (HTTP {r.status_code})")
    except Exception as e:
        print(f"❌ Model details error: {e}")
    
    # Try different parameter combinations
    print("\n🧪 Testing parameter variations...")
    
    param_tests = [
        {"max_tokens": 1, "temperature": 0.0},
        {"max_tokens": 50, "temperature": 0.0},
        {"max_tokens": 10, "temperature": 1.0},
        {"max_tokens": 10, "temperature": 0.5, "top_p": 0.9},
        {"max_tokens": 10, "temperature": 0.1, "presence_penalty": 0.0},
        {"max_tokens": 10, "temperature": 0.1, "frequency_penalty": 0.0},
    ]
    
    for params in param_tests:
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hi"}],
            **params
        }
        
        try:
            r = requests.post(
                'http://localhost:1234/v1/chat/completions',
                json=payload,
                timeout=30
            )
            
            if r.status_code == 200:
                data = r.json()
                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    if content and content.strip():
                        print(f"✅ Working params: {params} -> '{content.strip()}'")
                        return params
                    else:
                        print(f"❌ Empty with params: {params}")
            else:
                print(f"❌ HTTP {r.status_code} with params: {params}")
                
        except Exception as e:
            print(f"❌ Error with params {params}: {e}")
    
    return None

def main():
    print("🚀 Deep Debug: OpenAI GPT-OSS-20B Empty Response Issue")
    print("=" * 60)
    
    # Step 1: Get full response details
    content = get_full_response()
    
    # Step 2: Test different prompts
    prompt_success = test_different_prompts()
    
    # Step 3: Test completions endpoint
    completions_success = test_completions_endpoint()
    
    # Step 4: Check model parameters
    working_params = check_model_parameters()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if prompt_success:
        print("✅ Found working prompts - model is functional")
    elif completions_success:
        print("✅ Completions endpoint works - use that instead")
    elif working_params:
        print(f"✅ Found working parameters: {working_params}")
    else:
        print("❌ Model appears to be loaded but not generating content")
        print("\n💡 POSSIBLE CAUSES:")
        print("   1. Model is not fully initialized in LM Studio")
        print("   2. Model requires specific prompt format")
        print("   3. Model has generation issues with current settings")
        print("   4. LM Studio configuration problem")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. In LM Studio, go to Chat tab and test the model manually")
        print("   2. Try unloading and reloading the model")
        print("   3. Check LM Studio logs for any warnings/errors")
        print("   4. Try a different model temporarily")
        print("   5. Update LM Studio to the latest version")

if __name__ == "__main__":
    main()
