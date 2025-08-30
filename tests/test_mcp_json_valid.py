import json, os, importlib

def test_mcp_json_is_valid():
    with open(os.path.join('recommendations','mcp.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'mcpServers' in data and isinstance(data['mcpServers'], dict)
    assert 'lmstudio-mcp' in data['mcpServers']
    srv = data['mcpServers']['lmstudio-mcp']
    assert srv['type'] == 'stdio'
    assert srv['command'] in ('python', 'py')
    assert isinstance(srv['args'], list) and srv['args']
    # Basic env validations
    env = srv.get('env', {})
    assert env.get('MODEL_NAME') == 'openai/gpt-oss-20b'
    assert env.get('OPENAI_MODEL') == 'gpt5'
    assert env.get('ANTHROPIC_MODEL') == 'claude-4-sonnet'

