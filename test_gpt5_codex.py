#!/usr/bin/env python3
"""
Test gpt-5-codex for coding tasks
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = 'https://api.openai.com/v1'

print('🔧 GPT-5-CODEX TEST')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test gpt-5-codex
print('Testing gpt-5-codex...')
payload = {
    'model': 'gpt-5-codex',
    'messages': [
        {'role': 'user', 'content': 'Write a Python function to check if a number is prime. Include docstring.'}
    ],
    'max_tokens': 200
}

try:
    response = requests.post(
        f'{base_url}/chat/completions',
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f'Status: {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        model_returned = data.get('model', 'unknown')
        tokens = data.get('usage', {}).get('total_tokens', 0)
        
        print()
        print('✅ GPT-5-CODEX IS WORKING!')
        print('='*80)
        print(f'Model returned: {model_returned}')
        print(f'Tokens used: {tokens}')
        print()
        print('Response:')
        print(content)
        print('='*80)
        print()
        print('✅ gpt-5-codex can be used for coding tasks!')
        
    else:
        error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
        error_msg = error_data.get('error', {}).get('message', 'Unknown')
        error_type = error_data.get('error', {}).get('type', 'unknown')
        
        print()
        print(f'❌ gpt-5-codex is NOT available')
        print(f'Error type: {error_type}')
        print(f'Error message: {error_msg}')
        print()
        
        if 'does not exist' in error_msg.lower() or 'model' in error_msg.lower():
            print('⚠️  Model does not exist in your account')
            print()
            print('Alternative: Use gpt-5-chat-latest for coding')
            print('It has excellent coding capabilities!')
        
except Exception as e:
    print(f'❌ Exception: {str(e)}')

print()
print('='*80)
