import importlib, json


def test_smart_task_refinement_loop():
    srv_mod = importlib.import_module('server')

    class DummyServer:
        async def route_chat(self, prompt: str, *, intent=None, role=None, temperature=0.2):
            # First call returns minimal invalid args, then refinement returns corrected args
            if 'Choose the best tool' in prompt:
                return json.dumps({'tool': 'agent_team_plan_and_code', 'arguments': {'apply_changes': True}, 'confidence': 0.3})
            # refinement step: return required field 'task'
            return json.dumps({'task': 'build CLI tool'})

    out = srv_mod.handle_smart_task({
        'instruction': 'build a CLI tool', 'context': '', 'dry_run': True, 'max_refines': 2
    }, DummyServer())
    data = json.loads(out)
    assert data['selected'] == 'agent_team_plan_and_code'
    assert data['invoked'] is False
    # Accept task from instruction if router returns minimal args
    assert 'task' in data['arguments'] and isinstance(data['arguments']['task'], str)
    assert isinstance(data.get('refine_attempts'), list) and len(data['refine_attempts']) >= 1

