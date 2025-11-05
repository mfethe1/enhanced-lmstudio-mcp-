#!/usr/bin/env python3
"""
Direct OpenAI API test with multiple model fallbacks
"""

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

print('🔍 DIRECT OPENAI API TEST')
print('='*80)
print()
print(f'Base URL: {base_url}')
print(f'API Key (first 30 chars): {api_key[:30] if api_key else "NOT SET"}...')
print(f'API Key length: {len(api_key) if api_key else 0} characters')
print()

if not api_key:
    print('❌ ERROR: No API key found!')
    exit(1)

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Try multiple models in order of preference
models_to_try = [
    ('gpt-4o-mini', 'GPT-4o Mini (cheapest, most reliable)'),
    ('gpt-4o', 'GPT-4o'),
    ('gpt-4', 'GPT-4'),
    ('gpt-3.5-turbo', 'GPT-3.5 Turbo'),
    ('gpt-5', 'GPT-5'),
]

print('='*80)
print('🧪 TESTING MODELS (cheapest first to verify billing)')
print('='*80)
print()

working_model = None

for model_id, model_name in models_to_try:
    print(f'Testing: {model_name} ({model_id})...')
    
    payload = {
        'model': model_id,
        'messages': [
            {'role': 'user', 'content': 'Say "Hello, I am working!" in exactly 5 words.'}
        ],
        'max_tokens': 20,
        'temperature': 0.5
    }
    
    try:
        response = requests.post(
            f'{base_url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f'  Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            model_returned = data.get('model', 'unknown')
            usage = data.get('usage', {})
            
            print(f'  ✅ SUCCESS!')
            print(f'  Model returned: {model_returned}')
            print(f'  Response: {content}')
            print(f'  Tokens used: {usage.get("total_tokens", "unknown")}')
            
            working_model = model_id
            break
        else:
            # Parse error
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                error_type = error_data.get('error', {}).get('type', 'unknown')
                error_code = error_data.get('error', {}).get('code', 'unknown')
                
                print(f'  ❌ FAILED: {error_type}')
                print(f'     Error: {error_msg[:150]}')
                
                # If it's not a model availability issue, stop trying
                if 'model' not in error_msg.lower() and 'does not exist' not in error_msg.lower():
                    if 'quota' in error_msg.lower() or 'insufficient' in error_msg.lower():
                        print()
                        print('  ⚠️  QUOTA/BILLING ISSUE DETECTED')
                        print('     This API key appears to have insufficient quota.')
                        print()
                    elif 'invalid' in error_msg.lower() or 'authentication' in error_msg.lower():
                        print()
                        print('  ⚠️  AUTHENTICATION ISSUE DETECTED')
                        print('     This API key may be invalid or expired.')
                        print()
                        break
            except:
                print(f'  ❌ Error: {response.text[:200]}')
    except Exception as e:
        print(f'  ❌ Exception: {str(e)[:100]}')
    
    print()

print('='*80)
print('📊 RESULTS')
print('='*80)
print()

if working_model:
    print(f'✅ SUCCESS! Working model found: {working_model}')
    print()
    print('Your OpenAI API is working correctly!')
    print()
    print('Next steps:')
    print('  1. Update OPENAI_MODEL in .env.local if needed')
    print('  2. Test GPT-5 once basic connectivity is confirmed')
else:
    print('❌ NO WORKING MODELS FOUND')
    print()
    print('Possible issues:')
    print('  1. API key is invalid or expired')
    print('  2. Billing/quota issue persists')
    print('  3. Network connectivity problem')
    print()
    print('Action required:')
    print('  1. Verify your API key at: https://platform.openai.com/api-keys')
    print('  2. Check billing at: https://platform.openai.com/account/billing')
    print('  3. Generate a new API key if needed')

print()
print('='*80)
