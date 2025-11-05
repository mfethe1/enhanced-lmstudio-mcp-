#!/usr/bin/env python3
"""
Test O-series reasoning models (o1, o3) for complex tasks
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = 'https://api.openai.com/v1'

print('🧠 TESTING O-SERIES REASONING MODELS')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Test reasoning models
reasoning_models = [
    ('o3-mini', 'O3 Mini (Fast reasoning)'),
    ('o3', 'O3 (Advanced reasoning)'),
    ('o3-pro', 'O3 Pro (Maximum reasoning)'),
    ('o1', 'O1 (Original reasoning)'),
    ('o1-pro', 'O1 Pro (Advanced O1)'),
]

print(f'Testing {len(reasoning_models)} reasoning models...')
print()

working_models = []

for model_id, model_name in reasoning_models:
    print(f'Testing: {model_name} ({model_id})...')
    
    # Use a reasoning task
    payload = {
        'model': model_id,
        'messages': [
            {'role': 'user', 'content': 'If a train travels 120 miles in 2 hours, what is its speed in mph? Just the number.'}
        ],
        'max_tokens': 20
    }
    
    try:
        response = requests.post(
            f'{base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=45
        )
        
        print(f'  Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            model_returned = data.get('model', 'unknown')
            tokens = data.get('usage', {}).get('total_tokens', 0)
            
            print(f'  ✅ WORKS!')
            print(f'     Model returned: {model_returned}')
            print(f'     Response: {content}')
            print(f'     Tokens: {tokens}')
            
            working_models.append({
                'id': model_id,
                'name': model_name,
                'returned': model_returned
            })
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', {}).get('message', 'Unknown')
            error_type = error_data.get('error', {}).get('type', 'unknown')
            
            if 'does not exist' in error_msg.lower() or 'model' in error_msg.lower():
                print(f'  ❌ Not available')
            else:
                print(f'  ❌ Error: {error_type}')
                print(f'     {error_msg[:100]}')
    except Exception as e:
        print(f'  ❌ Exception: {str(e)[:80]}')
    
    print()

print('='*80)
print('📊 RESULTS')
print('='*80)
print()

if working_models:
    print(f'✅ FOUND {len(working_models)} WORKING REASONING MODELS:')
    print()
    for model in working_models:
        print(f'  ✓ {model["id"]:20s} - {model["name"]}')
    
    print()
    print('='*80)
    print('🎯 RECOMMENDED CONFIGURATION')
    print('='*80)
    print()
    
    # Recommend the best reasoning model
    if any(m['id'] == 'o3-mini' for m in working_models):
        reasoning = 'o3-mini'
        print(f'🧠 REASONING MODEL: {reasoning} (Fast, cost-effective)')
    elif any(m['id'] == 'o3' for m in working_models):
        reasoning = 'o3'
        print(f'🧠 REASONING MODEL: {reasoning} (Advanced)')
    elif any(m['id'] == 'o1' for m in working_models):
        reasoning = 'o1'
        print(f'🧠 REASONING MODEL: {reasoning} (Original)')
    else:
        reasoning = working_models[0]['id']
        print(f'🧠 REASONING MODEL: {reasoning}')
    
    print()
    print('Suggested configuration:')
    print(f'  OPENAI_MODEL=gpt-5-chat-latest           # General tasks')
    print(f'  OPENAI_REASONING_MODEL={reasoning}       # Complex reasoning')
    print(f'  OPENAI_FALLBACK_MODEL=gpt-4o             # Fallback')
    print(f'  LMSTUDIO_MODEL=openai/gpt-oss-20b       # Simple tasks (local)')
    
else:
    print('❌ NO REASONING MODELS AVAILABLE')
    print()
    print('Recommendation: Use gpt-5-chat-latest for all tasks')

print()
print('='*80)
