from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Import the enhanced schema from exported assets (Windows path with dash directory)
import importlib.util, sys, pathlib
_asset_path = pathlib.Path(__file__).parent / 'recommendations' / 'exported-assets' / 'enhanced_mcp_storage.py'
spec = importlib.util.spec_from_file_location('enhanced_mcp_storage_assets', str(_asset_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore
_EnhancedSQLite = mod.EnhancedMCPStorage

try:
    # Optional Postgres parity could be added here in the future
    from storage_postgres import MCPStoragePostgres  # type: ignore
except Exception:  # pragma: no cover
    MCPStoragePostgres = None  # type: ignore


class StorageSelector:
    """Select between legacy storage and the enhanced V2 schema via env flags.
    - ENHANCED_STORAGE=1 enables the enhanced V2 SQLite storage
    - STORAGE_BACKEND=postgres reserved for future parity with V2 schema
    """

    def __init__(self, legacy_impl):
        self.legacy_impl = legacy_impl
        use_v2 = os.getenv("ENHANCED_STORAGE", "0").strip() in ("1", "true", "yes")
        backend = os.getenv("STORAGE_BACKEND", "sqlite").strip().lower()
        if use_v2 and backend == "sqlite":
            db_path = os.getenv("SQLITE_PATH", "enhanced_mcp_storage.db")
            self.impl = _EnhancedSQLite(db_path=db_path)
        else:
            # Fall back to the legacy storage implementation (SQLite or Postgres)
            self.impl = legacy_impl

    # Delegate existing public API expected by the server
    def store_memory(self, *a, **kw) -> bool: return self.impl.store_memory(*a, **kw)
    def retrieve_memory(self, *a, **kw) -> List[Dict]: return self.impl.retrieve_memory(*a, **kw)
    def delete_memory(self, *a, **kw) -> bool: return self.impl.delete_memory(*a, **kw)
    def store_thinking_step(self, *a, **kw) -> bool: return getattr(self.impl, "store_thinking_step", lambda *a, **kw: True)(*a, **kw)
    def get_thinking_session(self, *a, **kw) -> List[Dict]: return getattr(self.impl, "get_thinking_session", lambda *a, **kw: [])(*a, **kw)
    def log_performance(self, *a, **kw) -> bool: return getattr(self.impl, "log_performance", lambda *a, **kw: True)(*a, **kw)
    def get_performance_stats(self, *a, **kw) -> Dict[str, Any]: return getattr(self.impl, "get_performance_stats", lambda *a, **kw: {})(*a, **kw)
    def log_error(self, *a, **kw) -> bool: return getattr(self.impl, "log_error", lambda *a, **kw: True)(*a, **kw)
    def get_error_patterns(self, *a, **kw) -> List[Dict]: return getattr(self.impl, "get_error_patterns", lambda *a, **kw: [])(*a, **kw)
    def cleanup_old_data(self, *a, **kw) -> Dict[str, int]: return getattr(self.impl, "cleanup_old_data", lambda *a, **kw: {})(*a, **kw)
    def get_storage_stats(self, *a, **kw) -> Dict[str, Any]: return getattr(self.impl, "get_storage_stats", lambda *a, **kw: {})(*a, **kw)

    # Expose V2-only convenience APIs when available
    def __getattr__(self, name: str):
        # If the underlying impl has advanced methods (context_envelopes, artifacts, etc.), expose them
        return getattr(self.impl, name)

