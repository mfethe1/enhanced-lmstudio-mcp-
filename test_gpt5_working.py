#!/usr/bin/env python3
"""
Test GPT-5 with the correct working API key
"""

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

print('🚀 GPT-5 COMPREHENSIVE TEST')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test all GPT-5 variants
gpt5_models = [
    'gpt-5',
    'gpt-5-2025-08-07',
    'gpt-5-chat-latest',
    'gpt-5-codex',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-pro',
    'gpt-5-pro-2025-10-06',
]

print(f'Testing {len(gpt5_models)} GPT-5 models...')
print()

working_models = []

for model_id in gpt5_models:
    print(f'Testing: {model_id:35s} ', end='', flush=True)
    
    payload = {
        'model': model_id,
        'messages': [
            {'role': 'user', 'content': 'What is 2+2? Answer with just the number.'}
        ],
        'max_tokens': 10,
        'temperature': 0
    }
    
    try:
        response = requests.post(
            f'{base_url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            model_returned = data.get('model', 'unknown')
            tokens = data.get('usage', {}).get('total_tokens', 0)
            
            print(f'✅ WORKS!')
            print(f'     → Returned as: {model_returned}')
            print(f'     → Response: "{content}" (tokens: {tokens})')
            
            working_models.append({
                'id': model_id,
                'returned': model_returned,
                'response': content,
                'tokens': tokens
            })
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', {}).get('message', 'Unknown')
            
            if 'does not exist' in error_msg or 'model' in error_msg.lower():
                print(f'❌ Not available')
            else:
                print(f'❌ Error: {error_msg[:50]}')
    except Exception as e:
        print(f'❌ Exception: {str(e)[:50]}')

print()
print('='*80)
print('📊 RESULTS')
print('='*80)
print()

if working_models:
    print(f'✅ FOUND {len(working_models)} WORKING GPT-5 MODELS:')
    print()
    for model in working_models:
        print(f'  ✓ {model["id"]:35s} → {model["returned"]}')
    
    print()
    print('='*80)
    print('🏆 RECOMMENDED CONFIGURATION')
    print('='*80)
    print()
    
    # Find the best model
    if any(m['id'] == 'gpt-5-pro-2025-10-06' for m in working_models):
        recommended = 'gpt-5-pro-2025-10-06'
        print(f'🥇 PRIMARY: {recommended} (Latest GPT-5 Pro)')
    elif any(m['id'] == 'gpt-5-pro' for m in working_models):
        recommended = 'gpt-5-pro'
        print(f'🥇 PRIMARY: {recommended} (GPT-5 Pro)')
    elif any(m['id'] == 'gpt-5' for m in working_models):
        recommended = 'gpt-5'
        print(f'🥇 PRIMARY: {recommended} (Standard GPT-5)')
    elif any(m['id'] == 'gpt-5-chat-latest' for m in working_models):
        recommended = 'gpt-5-chat-latest'
        print(f'🥇 PRIMARY: {recommended} (Latest GPT-5 Chat)')
    else:
        recommended = working_models[0]['id']
        print(f'🥇 PRIMARY: {recommended}')
    
    print(f'🥈 FALLBACK: gpt-4o (already working)')
    print()
    print('Update your .env.local:')
    print(f'  OPENAI_MODEL={recommended}')
    print('  OPENAI_FALLBACK_MODEL=gpt-4o')
    
else:
    print('❌ NO GPT-5 MODELS ARE WORKING')
    print()
    print('Current situation:')
    print('  • gpt-4o-mini: ✅ Working')
    print('  • gpt-4o: Should work')
    print('  • GPT-5: Not available yet or requires special access')
    print()
    print('Recommendation: Use gpt-4o as primary model')

print()
print('='*80)
