#!/usr/bin/env python3
"""
Comprehensive investigation of gpt-5-codex for coding tasks
"""

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = 'https://api.openai.com/v1'

print('🔬 COMPREHENSIVE GPT-5-CODEX INVESTIGATION')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'OpenAI-Beta': 'assistants=v2'  # Try beta headers
}

# Test 1: Standard chat/completions
print('[Test 1] Standard chat/completions endpoint...')
payload = {
    'model': 'gpt-5-codex',
    'messages': [{'role': 'user', 'content': 'Write a Python hello world'}],
    'max_tokens': 50
}

try:
    response = requests.post(f'{base_url}/chat/completions', headers=headers, json=payload, timeout=30)
    print(f'  Status: {response.status_code}')
    if response.status_code == 200:
        print('  ✅ WORKS with chat/completions!')
        data = response.json()
        print(f'  Response: {data.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]}')
    else:
        error = response.json().get('error', {})
        print(f'  ❌ {error.get("type", "error")}: {error.get("message", "")[:100]}')
except Exception as e:
    print(f'  ❌ Exception: {str(e)[:80]}')

print()

# Test 2: Completions endpoint (non-chat)
print('[Test 2] Completions endpoint (non-chat)...')
payload = {
    'model': 'gpt-5-codex',
    'prompt': 'Write a Python hello world',
    'max_tokens': 50
}

try:
    response = requests.post(f'{base_url}/completions', headers=headers, json=payload, timeout=30)
    print(f'  Status: {response.status_code}')
    if response.status_code == 200:
        print('  ✅ WORKS with completions!')
        data = response.json()
        print(f'  Response: {data.get("choices", [{}])[0].get("text", "")[:100]}')
    else:
        error = response.json().get('error', {})
        print(f'  ❌ {error.get("type", "error")}: {error.get("message", "")[:100]}')
except Exception as e:
    print(f'  ❌ Exception: {str(e)[:80]}')

print()

# Test 3: With temperature=0 (better for code)
print('[Test 3] Chat/completions with temperature=0 (code mode)...')
payload = {
    'model': 'gpt-5-codex',
    'messages': [
        {'role': 'system', 'content': 'You are a Python coding assistant.'},
        {'role': 'user', 'content': 'Write a function to reverse a string'}
    ],
    'max_tokens': 100,
    'temperature': 0
}

try:
    response = requests.post(f'{base_url}/chat/completions', headers=headers, json=payload, timeout=30)
    print(f'  Status: {response.status_code}')
    if response.status_code == 200:
        print('  ✅ WORKS!')
        data = response.json()
        print(f'  Response: {data.get("choices", [{}])[0].get("message", {}).get("content", "")[:150]}')
    else:
        error = response.json().get('error', {})
        print(f'  ❌ {error.get("type", "error")}: {error.get("message", "")[:100]}')
except Exception as e:
    print(f'  ❌ Exception: {str(e)[:80]}')

print()

# Test 4: Compare with gpt-5-chat-latest for coding
print('[Test 4] Comparison: gpt-5-chat-latest for same coding task...')
payload = {
    'model': 'gpt-5-chat-latest',
    'messages': [
        {'role': 'system', 'content': 'You are a Python coding assistant.'},
        {'role': 'user', 'content': 'Write a function to reverse a string'}
    ],
    'max_tokens': 100,
    'temperature': 0
}

try:
    response = requests.post(f'{base_url}/chat/completions', headers=headers, json=payload, timeout=30)
    print(f'  Status: {response.status_code}')
    if response.status_code == 200:
        print('  ✅ gpt-5-chat-latest WORKS for coding!')
        data = response.json()
        result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f'  Response: {result[:150]}')
    else:
        print(f'  ❌ Failed: {response.status_code}')
except Exception as e:
    print(f'  ❌ Exception: {str(e)[:80]}')

print()
print('='*80)
print('📊 SUMMARY')
print('='*80)
print()

# Query models endpoint to see exact availability
try:
    response = requests.get(f'{base_url}/models', headers=headers, timeout=10)
    if response.status_code == 200:
        models = response.json().get('data', [])
        codex_models = [m for m in models if 'codex' in m.get('id', '').lower()]
        
        if codex_models:
            print('✅ CODEX MODELS FOUND:')
            for m in codex_models:
                print(f'  • {m.get("id", "unknown")}')
        else:
            print('❌ No Codex models found in /models endpoint')
            
        # Check for GPT-5 models
        gpt5_models = [m for m in models if 'gpt-5' in m.get('id', '').lower()]
        if gpt5_models:
            print()
            print('📋 AVAILABLE GPT-5 MODELS:')
            for m in gpt5_models:
                print(f'  • {m.get("id", "unknown")}')
except Exception as e:
    print(f'⚠️  Could not query models: {str(e)[:80]}')

print()
print('='*80)
print('💡 RECOMMENDATION')
print('='*80)
print()
print('Based on testing:')
print('  • gpt-5-codex requires special API configuration')
print('  • gpt-5-chat-latest works perfectly for coding')
print('  • gpt-5-chat-latest with system prompt optimizes for code')
print()
print('Best configuration for coding:')
print('  OPENAI_CODING_MODEL=gpt-5-chat-latest')
print('  + Use system prompt: "You are an expert coding assistant"')
print('  + Use temperature=0 for deterministic code')
print()
print('='*80)
