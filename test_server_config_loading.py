#!/usr/bin/env python3
"""
Test that server.py loads the correct configuration
"""

import os
import sys

# Add server path
sys.path.insert(0, 'E:\\Projects\\lmstudio-mcp')

# Import server to trigger env loading
import server

print('✅ SERVER CONFIGURATION TEST')
print('='*80)
print()
print('ANTHROPIC MODELS (from os.environ after server.py import):')
print(f'  Primary Model:       {os.getenv("ANTHROPIC_MODEL")}')
print(f'  Complex Model:       {os.getenv("ANTHROPIC_MODEL_COMPLEX")}')
print(f'  Overseer Model:      {os.getenv("ANTHROPIC_MODEL_OVERSEER")}')
print()
print('OPENAI MODELS:')
print(f'  Primary Model:       {os.getenv("OPENAI_MODEL")}')
print(f'  Fallback Model:      {os.getenv("OPENAI_FALLBACK_MODEL")}')
print()
print('='*80)

expected = {
    'ANTHROPIC_MODEL': 'claude-sonnet-4-5-20250929',
    'ANTHROPIC_MODEL_COMPLEX': 'claude-opus-4-1-20250805',
    'ANTHROPIC_MODEL_OVERSEER': 'claude-opus-4-1-20250805',
}

all_correct = True
for key, expected_value in expected.items():
    actual = os.getenv(key)
    if actual == expected_value:
        print(f'✅ {key}: CORRECT')
    else:
        print(f'❌ {key}: WRONG - expected "{expected_value}", got "{actual}"')
        all_correct = False

print()
if all_correct:
    print('🎉 ALL CONFIGURATIONS CORRECT!')
else:
    print('⚠️  SOME CONFIGURATIONS ARE WRONG')
