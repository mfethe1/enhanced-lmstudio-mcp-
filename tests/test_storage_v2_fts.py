import importlib
import os
import tempfile


def test_v2_storage_fts_like_fallback(tmp_path, monkeypatch):
    # Use a temp sqlite db path
    db_path = tmp_path / 'mem.db'
    monkeypatch.setenv('ENHANCED_STORAGE', '1')
    # Force v2 by importing the module and initializing
    storage_mod = importlib.import_module('enhanced_mcp_storage_v2')
    Storage = storage_mod.EnhancedMCPStorageV2
    st = Storage(str(db_path))

    # Store a couple of entries
    assert st.store_memory('k1', 'Router learns from analytics', 'general') is True
    assert st.store_memory('k2', 'Collaborator writes tests', 'general') is True

    # Search term; if FTS exists we use MATCH, else fallback to LIKE should still find some
    res = st.retrieve_memory(search_term='Router', limit=10)
    assert isinstance(res, list)
    # At least one result should match either via FTS or LIKE
    assert any('Router' in (r.get('value') or '') for r in res)

