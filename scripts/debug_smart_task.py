import importlib, json, sys, os
sys.path.insert(0, os.getcwd())
srv_mod = importlib.import_module('server')

class DummyServer:
    async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
        return json.dumps({
            'tool': 'agent_team_review_and_test',
            'arguments': {'apply_fixes': True},
            'confidence': 0.61
        })

out = srv_mod.handle_smart_task({
    'instruction': 'review and test this change', 'context': 'safety-critical', 'dry_run': True
}, DummyServer())
print('OUT_RAW=', out)
try:
    data = json.loads(out)
    print('PARSED_KEYS=', sorted(list(data.keys())))
    print('SELECTED=', data.get('selected'))
    print('INVOKED=', data.get('invoked'))
    print('ARGS_KEYS=', list((data.get('arguments') or {}).keys()))
except Exception as e:
    print('JSON_ERROR=', e)

