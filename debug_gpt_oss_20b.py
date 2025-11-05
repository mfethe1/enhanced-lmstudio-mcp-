#!/usr/bin/env python3
"""
Debug why openai/gpt-oss-20b is not working and fix it
"""
import requests
import json
import time
import os

def check_model_status():
    """Check if the model is loaded and available"""
    print("🔍 Checking openai/gpt-oss-20b model status...")
    print("=" * 60)
    
    try:
        # Get all models
        r = requests.get('http://localhost:1234/v1/models', timeout=10)
        models = r.json().get('data', [])
        
        # Find our target model
        target_model = None
        for model in models:
            if model.get('id') == 'openai/gpt-oss-20b':
                target_model = model
                break
        
        if target_model:
            print("✅ Model found in LM Studio:")
            print(f"   ID: {target_model.get('id')}")
            print(f"   Object: {target_model.get('object')}")
            print(f"   Created: {target_model.get('created')}")
            print(f"   Owned by: {target_model.get('owned_by')}")
            return True
        else:
            print("❌ Model 'openai/gpt-oss-20b' not found in available models")
            print(f"📊 Available models ({len(models)}):")
            for i, model in enumerate(models[:10]):
                print(f"   {i+1:2d}. {model.get('id')}")
            if len(models) > 10:
                print(f"       ... and {len(models) - 10} more")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check models: {e}")
        return False

def test_model_with_different_configs():
    """Test the model with various configurations to find what works"""
    print("\n🧪 Testing openai/gpt-oss-20b with different configurations...")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            "name": "Minimal",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            },
            "timeout": 30
        },
        {
            "name": "With temperature",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10,
                "temperature": 0.1
            },
            "timeout": 30
        },
        {
            "name": "Longer timeout",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.1
            },
            "timeout": 120  # Much longer timeout
        },
        {
            "name": "System message",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            },
            "timeout": 60
        },
        {
            "name": "Stream disabled",
            "payload": {
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.1,
                "stream": False
            },
            "timeout": 60
        }
    ]
    
    working_configs = []
    
    for config in test_configs:
        print(f"\n🔧 Testing: {config['name']}")
        print(f"   Timeout: {config['timeout']}s")
        print(f"   Payload: {json.dumps(config['payload'], indent=2)}")
        
        start_time = time.time()
        try:
            r = requests.post(
                'http://localhost:1234/v1/chat/completions',
                json=config['payload'],
                timeout=config['timeout']
            )
            
            elapsed = time.time() - start_time
            
            if r.status_code == 200:
                data = r.json()
                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '').strip()
                    if content:
                        print(f"   ✅ SUCCESS ({elapsed:.1f}s): '{content}'")
                        working_configs.append(config)
                    else:
                        print(f"   ❌ Empty response ({elapsed:.1f}s)")
                        # Check for other response fields
                        print(f"   📊 Response data: {json.dumps(data, indent=2)[:200]}...")
                else:
                    print(f"   ❌ No choices ({elapsed:.1f}s)")
                    print(f"   📊 Response: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"   ❌ HTTP {r.status_code} ({elapsed:.1f}s): {r.text[:100]}")
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"   ⏰ TIMEOUT ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ ERROR ({elapsed:.1f}s): {e}")
    
    return working_configs

def check_lm_studio_logs():
    """Check if we can get any insights from LM Studio itself"""
    print("\n📋 LM Studio Server Information...")
    print("=" * 60)
    
    try:
        # Try to get server info
        r = requests.get('http://localhost:1234/v1/models', timeout=5)
        print(f"✅ LM Studio server responding (HTTP {r.status_code})")
        
        # Check headers for any useful info
        headers = dict(r.headers)
        if headers:
            print("📊 Server headers:")
            for key, value in headers.items():
                if key.lower() in ['server', 'x-powered-by', 'content-type']:
                    print(f"   {key}: {value}")
        
        # Try a simple health check
        try:
            health_r = requests.get('http://localhost:1234/health', timeout=5)
            if health_r.status_code == 200:
                print(f"✅ Health endpoint: {health_r.text}")
        except:
            print("ℹ️  No health endpoint available")
            
    except Exception as e:
        print(f"❌ Server check failed: {e}")

def main():
    print("🚀 OpenAI GPT-OSS-20B Model Debugger")
    print("=" * 60)
    
    # Step 1: Check if model is available
    model_available = check_model_status()
    
    if not model_available:
        print("\n💡 SOLUTION: The model is not loaded in LM Studio")
        print("   1. Open LM Studio application")
        print("   2. Go to the Models tab")
        print("   3. Search for 'openai/gpt-oss-20b' or 'gpt-oss-20b'")
        print("   4. Download and load the model")
        print("   5. Make sure it's the active model in the chat")
        return
    
    # Step 2: Test different configurations
    working_configs = test_model_with_different_configs()
    
    # Step 3: Check server info
    check_lm_studio_logs()
    
    # Step 4: Provide recommendations
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS RESULTS")
    print("=" * 60)
    
    if working_configs:
        print(f"✅ Found {len(working_configs)} working configuration(s):")
        for config in working_configs:
            print(f"   • {config['name']}")
        
        best_config = working_configs[0]
        print(f"\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   Name: {best_config['name']}")
        print(f"   Timeout: {best_config['timeout']}s")
        print(f"   Payload: {json.dumps(best_config['payload'], indent=2)}")
        
        # Generate environment variables
        print(f"\n🔧 ENVIRONMENT VARIABLES TO SET:")
        print(f"   MODEL_NAME=openai/gpt-oss-20b")
        print(f"   HTTP_READ_TIMEOUT_SIMPLE={best_config['timeout']}")
        if best_config['timeout'] > 60:
            print(f"   HTTP_READ_TIMEOUT_COMPLEX={best_config['timeout'] * 2}")
        
    else:
        print("❌ No working configurations found")
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("   1. Ensure the model is fully loaded (not just downloaded)")
        print("   2. Try switching to the model in LM Studio chat interface")
        print("   3. Check LM Studio logs for any error messages")
        print("   4. Try restarting LM Studio")
        print("   5. Consider using a different model temporarily")

if __name__ == "__main__":
    main()
