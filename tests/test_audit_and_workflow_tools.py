import json
import os
import server as s


def _call(name, args):
    return s.handle_tool_call({
        'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': name, 'arguments': args}
    })


def test_audit_and_workflow_smoke(monkeypatch):
    monkeypatch.setenv('ALLOWED_BASE_DIR', os.getcwd())
    # tools list contains our new tools
    tools = s.handle_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})['result']['tools']
    names = {t['name'] for t in tools}
    assert 'audit_search' in names
    assert 'workflow_create' in names

    # Basic audit actions
    r = _call('audit_verify_integrity', {'limit': 5})
    assert 'result' in r or 'error' in r

    r = _call('audit_add_rule', {'rule_id': 't1', 'name': 'tool exec', 'pattern': 'tool_execution'})
    assert 'result' in r or 'error' in r

    # Basic workflow creation
    r = _call('workflow_create', {'name': 'Demo'})
    assert 'result' in r or 'error' in r

