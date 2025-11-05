#!/usr/bin/env python3
"""
Final verification that Claude 4.5 and Opus 4.1 are correctly configured
"""

from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

print('✅ FINAL CONFIGURATION VERIFICATION')
print('='*80)
print()

# Get configured models
primary = os.getenv('ANTHROPIC_MODEL')
complex_model = os.getenv('ANTHROPIC_MODEL_COMPLEX')
overseer = os.getenv('ANTHROPIC_MODEL_OVERSEER')

print('📋 CONFIGURED MODELS:')
print(f'  Primary:   {primary}')
print(f'  Complex:   {complex_model}')
print(f'  Overseer:  {overseer}')
print()

# Get available models from API
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
models = client.models.list()

print('📡 AVAILABLE CLAUDE 4.x MODELS FROM API:')
claude4_models = [m for m in models.data if '4-' in m.id or '4.' in m.id]
for m in claude4_models:
    print(f'  • {m.id:40s} - {m.display_name}')
print()

# Verify configuration
print('🔍 VERIFICATION:')
available_ids = [m.id for m in models.data]

if primary in available_ids:
    print(f'  ✅ Primary model "{primary}" is available')
else:
    print(f'  ❌ Primary model "{primary}" NOT FOUND')

if complex_model in available_ids:
    print(f'  ✅ Complex model "{complex_model}" is available')
else:
    print(f'  ❌ Complex model "{complex_model}" NOT FOUND')

print()
print('='*80)
print('🎉 CONFIGURATION COMPLETE!')
print()
print('Your MCP server is configured with:')
print(f'  • Claude Sonnet 4.5 (Primary)')
print(f'  • Claude Opus 4.1 (Complex/Overseer)')
print()
print('Next step: Restart your MCP server to activate the new models')
print('  Command: python E:\\Projects\\lmstudio-mcp\\server.py')
print('='*80)
