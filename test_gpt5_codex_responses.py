#!/usr/bin/env python3
"""
Test gpt-5-codex using the /v1/responses endpoint
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = 'https://api.openai.com/v1'

print('🔧 GPT-5-CODEX TEST (Responses API)')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test gpt-5-codex with responses endpoint
print('Testing gpt-5-codex with /v1/responses...')
payload = {
    'model': 'gpt-5-codex',
    'input': 'Write a Python function to check if a number is prime. Include docstring.',
    'max_tokens': 200
}

try:
    response = requests.post(
        f'{base_url}/responses',
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f'Status: {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        print()
        print('✅ GPT-5-CODEX WORKS WITH RESPONSES API!')
        print('='*80)
        print('Response data:')
        print(data)
        print('='*80)
        
    else:
        error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
        error_msg = error_data.get('error', {}).get('message', response.text[:200])
        error_type = error_data.get('error', {}).get('type', 'unknown')
        
        print()
        print(f'❌ Failed with /v1/responses')
        print(f'Error type: {error_type}')
        print(f'Error: {error_msg}')
        print()
        
except Exception as e:
    print(f'❌ Exception: {str(e)}')

print()

# Also try with chat/completions but specify it's for coding
print('='*80)
print('Alternative: Testing gpt-5-chat-latest for coding...')
print('='*80)
print()

payload = {
    'model': 'gpt-5-chat-latest',
    'messages': [
        {'role': 'system', 'content': 'You are an expert coding assistant.'},
        {'role': 'user', 'content': 'Write a Python function to check if a number is prime. Include docstring and handle edge cases.'}
    ],
    'max_tokens': 250
}

try:
    response = requests.post(
        f'{base_url}/chat/completions',
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        tokens = data.get('usage', {}).get('total_tokens', 0)
        
        print('✅ gpt-5-chat-latest coding test:')
        print(f'Tokens: {tokens}')
        print()
        print(content)
        print()
        print('='*80)
        print('💡 RECOMMENDATION:')
        print('='*80)
        print()
        print('Use gpt-5-chat-latest for all tasks including coding.')
        print('It has excellent coding capabilities and is fully compatible')
        print('with your MCP server.')
        print()
        
except Exception as e:
    print(f'❌ Exception: {str(e)}')

print('='*80)
