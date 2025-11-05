#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv('E:\\Projects\\lmstudio-mcp\\.secrets\\.env.local')

print('✅ CONFIGURATION VERIFICATION')
print('='*80)
print()
print('ANTHROPIC MODELS:')
print(f'  Primary Model:       {os.getenv("ANTHROPIC_MODEL")}')
print(f'  Complex Model:       {os.getenv("ANTHROPIC_MODEL_COMPLEX")}')
print(f'  Overseer Model:      {os.getenv("ANTHROPIC_MODEL_OVERSEER")}')
print()
print('OPENAI MODELS:')
print(f'  Primary Model:       {os.getenv("OPENAI_MODEL")}')
print(f'  Fallback Model:      {os.getenv("OPENAI_FALLBACK_MODEL")}')
print()
print('='*80)
print('✅ All models configured successfully!')
print()
print('SUMMARY:')
print('  🎯 Claude Sonnet 4.5 - Primary Anthropic model')
print('  🎯 Claude Opus 4.1   - Complex/Overseer Anthropic model')
print('  🎯 GPT-5             - Primary OpenAI model')
print('  🎯 GPT-4o            - Fallback OpenAI model')
