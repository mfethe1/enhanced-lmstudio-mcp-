#!/usr/bin/env python3
"""
Comprehensive OpenAI GPT-5 Configuration Verification
"""

import os
import requests
from dotenv import load_dotenv
import json

# Load environment
load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

print('🔍 OPENAI GPT-5 CONFIGURATION VERIFICATION')
print('='*80)
print()

# Get configuration
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
primary_model = os.getenv('OPENAI_MODEL')
fallback_model = os.getenv('OPENAI_FALLBACK_MODEL')

print('📋 CURRENT CONFIGURATION:')
print(f'  Base URL:       {base_url}')
print(f'  Primary Model:  {primary_model}')
print(f'  Fallback Model: {fallback_model}')
print(f'  API Key:        {api_key[:20] if api_key else "NOT SET"}...')
print()

if not api_key:
    print('❌ ERROR: OPENAI_API_KEY not set!')
    exit(1)

# Step 1: Get all available models from OpenAI
print('='*80)
print('📡 QUERYING OPENAI MODELS API')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

try:
    response = requests.get(
        f'{base_url.rstrip("/")}/models',
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        models = data.get('data', [])
        print(f'✅ Successfully retrieved {len(models)} models')
        print()
        
        # Filter GPT-5 models
        gpt5_models = [m for m in models if 'gpt-5' in m.get('id', '').lower()]
        
        if gpt5_models:
            print(f'🏆 GPT-5 MODELS FOUND ({len(gpt5_models)}):')
            for model in sorted(gpt5_models, key=lambda x: x.get('id', '')):
                model_id = model.get('id', 'unknown')
                created = model.get('created', 'unknown')
                owned_by = model.get('owned_by', 'unknown')
                print(f'  • {model_id:45s} (owned by: {owned_by})')
            print()
        else:
            print('⚠️  NO GPT-5 MODELS FOUND')
            print('    Let me check for GPT-4 models instead...')
            print()
            
            gpt4_models = [m for m in models if 'gpt-4' in m.get('id', '').lower()]
            if gpt4_models:
                print(f'📊 LATEST GPT-4 MODELS ({len(gpt4_models)}):')
                for model in sorted(gpt4_models, key=lambda x: x.get('id', ''))[:10]:
                    print(f'  • {model.get("id", "unknown")}')
                print()
        
        # Check O-series models (reasoning models)
        o_models = [m for m in models if m.get('id', '').startswith('o1') or m.get('id', '').startswith('o3')]
        if o_models:
            print(f'🧠 O-SERIES REASONING MODELS ({len(o_models)}):')
            for model in sorted(o_models, key=lambda x: x.get('id', '')):
                print(f'  • {model.get("id", "unknown")}')
            print()
            
    else:
        print(f'❌ Failed to get models: {response.status_code}')
        print(f'   Response: {response.text[:200]}')
        models = []
        
except Exception as e:
    print(f'❌ Error querying models: {str(e)}')
    models = []

# Step 2: Test configured models
print('='*80)
print('🧪 TESTING CONFIGURED MODELS')
print('='*80)
print()

test_models = [
    (primary_model, 'Primary'),
    (fallback_model, 'Fallback'),
]

working_models = []
failed_models = []

for model_id, label in test_models:
    if not model_id:
        print(f'⚠️  {label} model not configured')
        continue
        
    print(f'Testing {label}: {model_id}...')
    
    payload = {
        'model': model_id,
        'messages': [{'role': 'user', 'content': 'What is your model name and version?'}],
        'max_tokens': 100
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
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            model_used = data.get('model', 'unknown')
            print(f'  ✅ WORKS! (returned as: {model_used})')
            print(f'     Response: {content[:100]}...')
            working_models.append((model_id, model_used, content))
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', {}).get('message', response.text[:100])
            error_type = error_data.get('error', {}).get('type', 'unknown')
            print(f'  ❌ FAILED: {response.status_code} - {error_type}')
            print(f'     Error: {error_msg[:150]}')
            failed_models.append((model_id, error_msg))
    except Exception as e:
        print(f'  ❌ Exception: {str(e)[:100]}')
        failed_models.append((model_id, str(e)))
    
    print()

# Step 3: Test specific GPT-5 variants
print('='*80)
print('🔬 TESTING SPECIFIC GPT-5 VARIANTS')
print('='*80)
print()

gpt5_variants = [
    'gpt-5',
    'gpt-5-2025-08-07',
    'gpt-5-chat-latest',
    'gpt-5-mini',
    'gpt-5-nano',
]

print(f'Testing {len(gpt5_variants)} GPT-5 model variants...')
print()

gpt5_working = []

for variant in gpt5_variants:
    print(f'[{gpt5_variants.index(variant)+1}/{len(gpt5_variants)}] {variant:35s} ', end='', flush=True)
    
    payload = {
        'model': variant,
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'max_tokens': 10
    }
    
    try:
        response = requests.post(
            f'{base_url.rstrip("/")}/chat/completions',
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            model_returned = data.get('model', 'unknown')
            print(f'✅ WORKS (as: {model_returned})')
            gpt5_working.append(variant)
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', {}).get('message', '')
            if 'does not exist' in error_msg or 'model' in error_msg.lower():
                print(f'❌ Not available')
            else:
                print(f'❌ {response.status_code}')
    except Exception as e:
        print(f'❌ Error')

print()

# Summary
print('='*80)
print('📊 SUMMARY')
print('='*80)
print()

if working_models:
    print(f'✅ WORKING MODELS ({len(working_models)}):')
    for model_id, returned_as, _ in working_models:
        status = '✅' if model_id == returned_as else '⚠️'
        print(f'   {status} {model_id}' + (f' (returned as {returned_as})' if model_id != returned_as else ''))
    print()

if failed_models:
    print(f'❌ FAILED MODELS ({len(failed_models)}):')
    for model_id, error in failed_models:
        print(f'   ✗ {model_id}')
        print(f'      Error: {error[:100]}')
    print()

if gpt5_working:
    print(f'🏆 GPT-5 MODELS WORKING ({len(gpt5_working)}):')
    for model in gpt5_working:
        print(f'   • {model}')
    print()

# Recommendations
print('='*80)
print('💡 RECOMMENDATIONS')
print('='*80)
print()

if primary_model in [m[0] for m in working_models]:
    print(f'✅ Your primary model "{primary_model}" is working correctly!')
else:
    print(f'⚠️  Your primary model "{primary_model}" is NOT working!')
    if gpt5_working:
        print(f'   Recommended: Switch to "{gpt5_working[0]}"')
    elif any('gpt-4o' in m.get('id', '') for m in models):
        print(f'   Recommended: Switch to "gpt-4o" (GPT-5 not available)')

print()
print('='*80)
