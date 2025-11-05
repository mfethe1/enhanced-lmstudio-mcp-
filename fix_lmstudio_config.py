#!/usr/bin/env python3
"""
Fix LM Studio configuration for world-class MCP platform
"""
import requests
import json
import os
import sys

def get_available_models():
    """Get all available models from LM Studio"""
    try:
        r = requests.get('http://localhost:1234/v1/models', timeout=10)
        models = r.json().get('data', [])
        return [model.get('id') for model in models]
    except Exception as e:
        print(f"❌ Failed to get models: {e}")
        return []

def test_model(model_id, timeout=60):
    """Test if a model works with a simple request"""
    print(f"🧪 Testing model: {model_id}")
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say 'Hello' in one word"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        r = requests.post(
            'http://localhost:1234/v1/chat/completions',
            json=payload,
            timeout=timeout
        )
        
        if r.status_code == 200:
            data = r.json()
            choices = data.get('choices', [])
            if choices:
                content = choices[0].get('message', {}).get('content', '').strip()
                if content:
                    print(f"✅ {model_id}: '{content}'")
                    return True
                else:
                    print(f"❌ {model_id}: Empty response")
            else:
                print(f"❌ {model_id}: No choices")
        else:
            print(f"❌ {model_id}: HTTP {r.status_code}")
            
    except requests.exceptions.Timeout:
        print(f"⏰ {model_id}: Timeout ({timeout}s)")
    except Exception as e:
        print(f"❌ {model_id}: {e}")
        
    return False

def create_production_env():
    """Create production-ready environment configuration"""
    
    print("🔧 Creating production environment configuration...")
    
    # Production-ready timeouts (much longer than 8s!)
    env_config = {
        # HTTP timeouts - much more generous
        'HTTP_CONNECT_TIMEOUT': '15',
        'HTTP_READ_TIMEOUT_SIMPLE': '120',  # 15x longer than current 8s
        'HTTP_READ_TIMEOUT_COMPLEX': '300', # For complex operations
        
        # LM Studio specific
        'LMSTUDIO_MAX_RETRIES': '3',
        'LMSTUDIO_RETRY_BACKOFF': '2.0',
        
        # Circuit breaker - more tolerant
        'CIRCUIT_LMSTUDIO_THRESHOLD': '10',  # Allow more failures before opening
        'CIRCUIT_LMSTUDIO_RECOVERY': '180',  # Longer recovery time
        
        # Crew AI and other tools
        'CREW_TOOL_TIMEOUT': '420',
        'SMART_PLAN_IMMEDIATE_TIMEOUT_SEC': '360',
        'ROUTER_BG_TIMEOUT_SEC': '500',
        
        # Fallback control
        'NO_FALLBACK_PROVIDERS': '0',  # Allow fallbacks when LM Studio fails
        
        # Logging
        'LOG_LEVEL': 'INFO'
    }
    
    # Write to .env file
    env_content = []
    for key, value in env_config.items():
        env_content.append(f"{key}={value}")
    
    with open('.env.production', 'w') as f:
        f.write('\n'.join(env_content))
    
    print("✅ Created .env.production with optimized settings")
    print("\n📋 Key improvements:")
    print("  • HTTP timeouts: 8s → 120s (15x increase)")
    print("  • Complex operations: 180s → 300s")
    print("  • Circuit breaker: More tolerant (10 failures)")
    print("  • Retry logic: 3 attempts with 2s backoff")
    print("  • Fallback providers: Enabled for reliability")

def main():
    print("🚀 LM Studio MCP Configuration Optimizer")
    print("=" * 60)
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available. Please start LM Studio and load a model.")
        return
    
    print(f"📊 Found {len(models)} available models:")
    for i, model in enumerate(models[:10]):  # Show first 10
        print(f"  {i+1:2d}. {model}")
    
    if len(models) > 10:
        print(f"     ... and {len(models) - 10} more")
    
    # Test the first few models to find working ones
    print(f"\n🧪 Testing models for compatibility...")
    working_models = []
    
    for model in models[:5]:  # Test first 5 models
        if test_model(model, timeout=30):
            working_models.append(model)
    
    if working_models:
        print(f"\n✅ Working models found:")
        for model in working_models:
            print(f"  • {model}")
        
        # Recommend the first working model
        recommended = working_models[0]
        print(f"\n🎯 Recommended model: {recommended}")
        
        # Update server.py with the working model
        print(f"\n🔧 To use this model, update your MCP configuration:")
        print(f"   MODEL_NAME={recommended}")
        
    else:
        print(f"\n❌ No working models found. Common issues:")
        print("  • Model not loaded in LM Studio")
        print("  • Model taking too long to respond")
        print("  • Model configuration issues")
    
    # Create production environment
    create_production_env()
    
    print(f"\n🎉 Next steps:")
    print("  1. Load the recommended model in LM Studio")
    print("  2. Copy .env.production settings to your environment")
    print("  3. Restart the MCP server")
    print("  4. Test with: python test_all_tools.py")

if __name__ == "__main__":
    main()
