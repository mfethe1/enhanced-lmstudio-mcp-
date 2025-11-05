import json
import os
from pathlib import Path
import importlib


def test_safe_path_allows_cwd(tmp_path):
    # Ensure CWD is allowed by default
    cwd = Path(os.getcwd()).resolve()
    server = importlib.import_module('server')
    p = server._safe_path(str(cwd / 'server.py'))  # file exists in repo
    assert cwd in [Path(d) for d in server._allowed_paths_manager.list()]
    assert str(p).startswith(str(cwd))


def test_grant_and_deny(tmp_path):
    server = importlib.import_module('server')
    # Create a temp dir to grant
    target = tmp_path / 'ext'
    target.mkdir(parents=True, exist_ok=True)

    # Grant access
    r = server.handle_grant_directory_access({'directory': str(target)})
    assert str(target.resolve()) in r['allowed_directories']

    # Now safe_path should allow
    p = server._safe_path(str(target / 'file.txt'))
    assert str(p).startswith(str(target.resolve()))

    # Revoke access
    r2 = server.handle_deny_directory_access({'directory': str(target)})
    assert str(target.resolve()) not in r2['allowed_directories']

    # Now safe_path should raise PATH_NOT_ALLOWED
    try:
        server._safe_path(str(target / 'file.txt'))
    except Exception as e:
        msg = str(e)
        assert 'PATH_NOT_ALLOWED' in msg
        data = json.loads(msg)
        assert data['code'] == 'PATH_NOT_ALLOWED'
        assert str(target.resolve()) not in data['allowed_roots']
    else:
        raise AssertionError('Expected PATH_NOT_ALLOWED')

