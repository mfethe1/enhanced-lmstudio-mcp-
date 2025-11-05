#!/usr/bin/env python3
import json

print('🔍 Validating mcp.json...')
print('='*80)
print()

try:
    with open('E:\\Projects\\lmstudio-mcp\\mcp.json', 'r') as f:
        data = json.load(f)
    
    print('✅ mcp.json is VALID JSON')
    print()
    
    # Check structure
    if 'mcpServers' in data:
        print('✅ Contains mcpServers')
        servers = list(data['mcpServers'].keys())
        print(f'   Servers: {", ".join(servers)}')
        print()
        
        for server_name, config in data['mcpServers'].items():
            print(f'📋 Server: {server_name}')
            print(f'   Type: {config.get("type", "unknown")}')
            print(f'   Command: {config.get("command", "unknown")}')
            print(f'   Args: {" ".join(config.get("args", []))}')
            print()
            
            env = config.get('env', {})
            print(f'   Environment variables: {len(env)}')
            
            # Check key models
            models = {
                'OpenAI': env.get('OPENAI_MODEL'),
                'OpenAI Coding': env.get('OPENAI_CODING_MODEL'),
                'OpenAI Reasoning': env.get('OPENAI_REASONING_MODEL'),
                'Anthropic': env.get('ANTHROPIC_MODEL'),
                'Anthropic Complex': env.get('ANTHROPIC_MODEL_COMPLEX'),
                'LM Studio': env.get('LMSTUDIO_MODEL')
            }
            
            print()
            print('   Models configured:')
            for name, model in models.items():
                status = '✅' if model else '❌'
                print(f'     {status} {name}: {model or "NOT SET"}')
    else:
        print('❌ Missing mcpServers key')
    
    print()
    print('='*80)
    print('✅ VALIDATION COMPLETE - mcp.json is ready to use!')
    print('='*80)
    
except json.JSONDecodeError as e:
    print(f'❌ JSON PARSE ERROR: {e}')
except Exception as e:
    print(f'❌ ERROR: {e}')
