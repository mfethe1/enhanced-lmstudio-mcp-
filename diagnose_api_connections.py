#!/usr/bin/env python3
"""
Comprehensive API Connection Diagnostic Tool
Tests OpenAI and Anthropic API connections and provides detailed troubleshooting.
"""

import os
import sys
import json
import requests
from pathlib import Path

# ANSI color codes for Windows
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def load_env_file():
    """Load environment variables from .secrets/.env.local"""
    env_path = Path(".secrets/.env.local")
    if not env_path.exists():
        print_error("Environment file not found at .secrets/.env.local")
        return {}
    
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
                os.environ[key.strip()] = value.strip()
    
    return env_vars

def test_openai_api():
    """Test OpenAI API connection"""
    print_header("🔍 TESTING OPENAI API CONNECTION")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment")
        return False
    
    # Mask API key for display
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print_info(f"API Key: {masked_key}")
    print_info(f"Base URL: {base_url}")
    print_info(f"Model: {model}")
    
    # Test 1: List models endpoint
    print("\n📋 Test 1: Listing available models...")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models_data = response.json()
            models = [m.get('id') for m in models_data.get('data', [])]
            print_success(f"Successfully listed {len(models)} models")
            
            # Check if requested model is available
            if model in models:
                print_success(f"Model '{model}' is available")
            else:
                print_warning(f"Model '{model}' not in list, but may still work")
                # Show some available models
                gpt_models = [m for m in models if 'gpt' in m.lower()][:5]
                if gpt_models:
                    print_info(f"Available GPT models: {', '.join(gpt_models)}")
        else:
            print_error(f"Failed to list models: HTTP {response.status_code}")
            print_error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception while listing models: {str(e)}")
        return False
    
    # Test 2: Simple chat completion
    print("\n💬 Test 2: Testing chat completion...")
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Say 'Hello, API is working!' and nothing else."}
            ],
            "max_tokens": 50,
            "temperature": 0
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print_success(f"Chat completion successful!")
            print_info(f"Response: {message}")
            return True
        else:
            print_error(f"Chat completion failed: HTTP {response.status_code}")
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text[:200])
            print_error(f"Error: {error_msg}")
            
            # Provide specific guidance
            if response.status_code == 401:
                print_warning("Authentication failed - API key may be invalid")
            elif response.status_code == 429:
                print_warning("Rate limit or quota exceeded")
            elif response.status_code == 404:
                print_warning(f"Model '{model}' not found - try 'gpt-4o-mini' or 'gpt-3.5-turbo'")
            
            return False
            
    except Exception as e:
        print_error(f"Exception during chat completion: {str(e)}")
        return False

def test_anthropic_api():
    """Test Anthropic API connection"""
    print_header("🔍 TESTING ANTHROPIC API CONNECTION")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    
    if not api_key:
        print_error("ANTHROPIC_API_KEY not found in environment")
        return False
    
    # Mask API key for display
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print_info(f"API Key: {masked_key}")
    print_info(f"Base URL: {base_url}")
    print_info(f"Model: {model}")
    
    # Test: Simple message completion
    print("\n💬 Testing message completion...")
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": model,
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Say 'Hello, Anthropic API is working!' and nothing else."}
            ]
        }
        
        response = requests.post(
            f"{base_url}/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('content', [{}])[0].get('text', '')
            print_success(f"Message completion successful!")
            print_info(f"Response: {content}")
            return True
        else:
            print_error(f"Message completion failed: HTTP {response.status_code}")
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text[:200])
            print_error(f"Error: {error_msg}")
            
            # Provide specific guidance
            if response.status_code == 401:
                print_warning("Authentication failed - API key may be invalid")
            elif response.status_code == 429:
                print_warning("Rate limit exceeded")
            elif response.status_code == 404:
                print_warning(f"Model '{model}' not found")
                print_info("Try: claude-3-5-sonnet-20241022, claude-3-opus-20240229, or claude-3-haiku-20240307")
            
            return False
            
    except Exception as e:
        print_error(f"Exception during message completion: {str(e)}")
        return False

def test_lmstudio_connection():
    """Test LM Studio local connection"""
    print_header("🔍 TESTING LM STUDIO CONNECTION")
    
    base_url = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
    api_key = os.getenv("LMSTUDIO_API_KEY", "sk-noauth")
    
    print_info(f"Base URL: {base_url}")
    
    # Test: List models
    print("\n📋 Testing model listing...")
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            models_data = response.json()
            models = [m.get('id') for m in models_data.get('data', [])]
            print_success(f"LM Studio is running with {len(models)} models loaded")
            if models:
                print_info(f"Available models: {', '.join(models[:5])}")
            return True
        else:
            print_error(f"Failed to connect: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to LM Studio - is it running?")
        print_info("Start LM Studio and load a model, then try again")
        return False
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return False

def generate_fixed_env_file(openai_working, anthropic_working):
    """Generate a fixed .env file with correct settings"""
    print_header("🔧 GENERATING FIXED CONFIGURATION")
    
    env_template = """# LM Studio (OpenAI-compatible) - Local AI models
LMSTUDIO_API_BASE=http://localhost:1234/v1
LMSTUDIO_API_KEY=sk-noauth
LMSTUDIO_MODEL=openai/gpt-oss-20b

"""
    
    if openai_working:
        env_template += """# OpenAI API - Cloud service
OPENAI_API_KEY={openai_key}
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo

"""
    else:
        env_template += """# OpenAI API - DISABLED (not working)
# OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

"""
    
    if anthropic_working:
        env_template += """# Anthropic API - Claude models
ANTHROPIC_API_KEY={anthropic_key}
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MODEL_COMPLEX=claude-3-opus-20240229
ANTHROPIC_MODEL_OVERSEER=claude-3-opus-20240229

"""
    else:
        env_template += """# Anthropic API - DISABLED (not working)
# ANTHROPIC_API_KEY=your_key_here
# ANTHROPIC_BASE_URL=https://api.anthropic.com
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

"""
    
    env_template += """# Firecrawl API - Web research
FIRECRAWL_API_KEY={firecrawl_key}

# Routing configuration
ANTHROPIC_SMART_SWITCH=1
OPUS_FOR_ANALYSIS=1
OPUS_FOR_LOW_CONF=1
LOW_CONF_THRESHOLD=0.5

# Performance settings
HTTP_CONNECT_TIMEOUT=10
HTTP_READ_TIMEOUT_SIMPLE=60
HTTP_READ_TIMEOUT_COMPLEX=180
PROACTIVE_RESEARCH_ENABLED=0
CIRCUIT_BREAKER_ENABLED=1
"""
    
    # Fill in actual keys
    env_template = env_template.format(
        openai_key=os.getenv("OPENAI_API_KEY", "your_key_here"),
        anthropic_key=os.getenv("ANTHROPIC_API_KEY", "your_key_here"),
        firecrawl_key=os.getenv("FIRECRAWL_API_KEY", "your_key_here")
    )
    
    # Save to file
    backup_path = Path(".secrets/.env.local.backup")
    env_path = Path(".secrets/.env.local")
    
    if env_path.exists():
        # Create backup
        import shutil
        shutil.copy(env_path, backup_path)
        print_info(f"Backed up existing config to {backup_path}")
    
    with open(env_path, 'w') as f:
        f.write(env_template)
    
    print_success(f"Generated new configuration at {env_path}")
    return env_path

def main():
    """Main diagnostic routine"""
    print(f"\n{Colors.BOLD}🔬 API CONNECTION DIAGNOSTIC TOOL{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    # Load environment
    print_info("Loading environment variables...")
    env_vars = load_env_file()
    
    if not env_vars:
        print_error("Failed to load environment variables")
        sys.exit(1)
    
    print_success(f"Loaded {len(env_vars)} environment variables")
    
    # Run tests
    results = {}
    
    results['lmstudio'] = test_lmstudio_connection()
    results['openai'] = test_openai_api()
    results['anthropic'] = test_anthropic_api()
    
    # Summary
    print_header("📊 DIAGNOSTIC SUMMARY")
    
    print("\nAPI Status:")
    for api, status in results.items():
        if status:
            print_success(f"{api.upper()}: Working")
        else:
            print_error(f"{api.upper()}: Not Working")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not results['lmstudio']:
        print_warning("LM Studio: Start LM Studio and load a model")
    
    if not results['openai']:
        print_warning("OpenAI: Check your API key and billing status at https://platform.openai.com/account/billing")
        print_info("  - Verify the API key is correct")
        print_info("  - Ensure you have credits/billing set up")
        print_info("  - Try model 'gpt-4o-mini' instead of 'gpt-5'")
    
    if not results['anthropic']:
        print_warning("Anthropic: Verify your API key at https://console.anthropic.com/")
        print_info("  - Check the API key is correct")
        print_info("  - Ensure billing is set up")
        print_info("  - Use correct model name: claude-3-5-sonnet-20241022")
    
    # Generate fixed config
    if not results['openai'] or not results['anthropic']:
        print("\n🔧 Generating optimized configuration...")
        generate_fixed_env_file(results['openai'], results['anthropic'])
    
    # Final advice
    print_header("🎯 NEXT STEPS")
    
    if results['lmstudio']:
        print_success("LM Studio is working - server can use local models")
    
    if not results['openai']:
        print("\n📝 To fix OpenAI:")
        print("1. Go to https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Update OPENAI_API_KEY in .secrets/.env.local")
        print("4. Ensure billing is set up at https://platform.openai.com/account/billing")
        print("5. Change OPENAI_MODEL to 'gpt-4o-mini' (more reliable than gpt-5)")
    
    if not results['anthropic']:
        print("\n📝 To fix Anthropic:")
        print("1. Go to https://console.anthropic.com/settings/keys")
        print("2. Create a new API key")
        print("3. Update ANTHROPIC_API_KEY in .secrets/.env.local")
        print("4. Ensure billing is set up")
        print("5. Use model: claude-3-5-sonnet-20241022")
    
    print("\n" + "="*70)
    
    if any(results.values()):
        print_success("At least one API is working - server can function")
    else:
        print_error("No APIs are working - please fix at least one")

if __name__ == "__main__":
    main()
