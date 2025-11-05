#!/usr/bin/env python3
"""
Check OpenAI account billing and quota status
"""

import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

api_key = os.getenv('OPENAI_API_KEY')
base_url = 'https://api.openai.com/v1'

print('💳 OPENAI BILLING & QUOTA CHECK')
print('='*80)
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Check subscription/billing
print('📊 Checking subscription status...')
try:
    response = requests.get(
        'https://api.openai.com/v1/dashboard/billing/subscription',
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print('✅ Subscription data retrieved:')
        print(f'   Plan: {data.get("plan", {}).get("title", "Unknown")}')
        print(f'   Has payment method: {data.get("has_payment_method", False)}')
        print(f'   Soft limit (USD): ${data.get("soft_limit_usd", 0)}')
        print(f'   Hard limit (USD): ${data.get("hard_limit_usd", 0)}')
    else:
        print(f'⚠️  Status {response.status_code}: {response.text[:200]}')
except Exception as e:
    print(f'⚠️  Could not check subscription: {str(e)[:100]}')

print()

# Check usage
print('📈 Checking usage...')
try:
    # Get current month usage
    today = datetime.now()
    start_date = today.replace(day=1).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    response = requests.get(
        f'https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}',
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        total_usage = data.get('total_usage', 0) / 100  # Convert from cents
        print(f'✅ Usage this month: ${total_usage:.2f}')
        
        # Daily breakdown
        daily = data.get('daily_costs', [])
        if daily and len(daily) > 0:
            latest = daily[-1]
            latest_cost = sum(item.get('cost', 0) for item in latest.get('line_items', [])) / 100
            print(f'   Latest day usage: ${latest_cost:.2f} ({latest.get("timestamp", "unknown")})')
    else:
        print(f'⚠️  Status {response.status_code}: {response.text[:200]}')
except Exception as e:
    print(f'⚠️  Could not check usage: {str(e)[:100]}')

print()

# Check credit grants
print('💰 Checking credits...')
try:
    response = requests.get(
        'https://api.openai.com/v1/dashboard/billing/credit_grants',
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        grants = data.get('data', [])
        if grants:
            print(f'✅ Found {len(grants)} credit grant(s):')
            for grant in grants:
                amount = grant.get('grant_amount', 0)
                used = grant.get('used_amount', 0)
                remaining = amount - used
                expires = grant.get('expires_at', 'N/A')
                print(f'   • Grant: ${amount:.2f}, Used: ${used:.2f}, Remaining: ${remaining:.2f}')
                print(f'     Expires: {expires}')
        else:
            print('⚠️  No credit grants found')
    else:
        print(f'⚠️  Status {response.status_code}: {response.text[:200]}')
except Exception as e:
    print(f'⚠️  Could not check credits: {str(e)[:100]}')

print()
print('='*80)
print('🔍 DIAGNOSIS')
print('='*80)
print()

print('Your OpenAI API key appears to have QUOTA ISSUES.')
print()
print('Possible causes:')
print('  1. ❌ No payment method on file')
print('  2. ❌ Billing limit reached (soft or hard limit)')
print('  3. ❌ Free trial credits expired')
print('  4. ❌ Account needs funding')
print()
print('📝 ACTION REQUIRED:')
print('  1. Visit: https://platform.openai.com/account/billing')
print('  2. Add a payment method if missing')
print('  3. Check your billing limits')
print('  4. Add credits to your account')
print()
print('Once billing is resolved, GPT-5 models will be accessible.')
print('='*80)
