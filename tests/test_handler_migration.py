import importlib


def test_handlers_importable():
    for mod in [
        'handlers.research', 'handlers.agent_teams', 'handlers.memory',
        'handlers.code_tools', 'handlers.workflow', 'handlers.audit'
    ]:
        importlib.import_module(mod)


def test_registry_registration_exists():
    import server
    # Ensure helper exists and can be called without errors
    assert hasattr(server, '_register_all_handlers')

