from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from storage import MCPStorage as _SQLiteStorage

try:
    # Optional Postgres
    from storage_postgres import MCPStoragePostgres as _PgStorage  # type: ignore
except Exception:  # pragma: no cover
    _PgStorage = None  # type: ignore


class EnhancedMCPStorage:
    """Facade that selects SQLite or Postgres based on env, keeping the same API.
    Adds convenience helpers for envelopes and artifacts when needed later.
    """

    def __init__(self) -> None:
        backend = os.getenv("STORAGE_BACKEND", "sqlite").strip().lower()
        if backend == "postgres" and _PgStorage is not None:
            dsn = os.getenv("POSTGRES_DSN", "")
            self._impl = _PgStorage(dsn=dsn)
        else:
            self._impl = _SQLiteStorage(os.getenv("SQLITE_PATH", "mcp_data.db"))

    # Delegate existing public API
    def store_memory(self, *a, **kw) -> bool: return self._impl.store_memory(*a, **kw)
    def retrieve_memory(self, *a, **kw) -> List[Dict]: return self._impl.retrieve_memory(*a, **kw)
    def delete_memory(self, *a, **kw) -> bool: return self._impl.delete_memory(*a, **kw)
    def store_thinking_step(self, *a, **kw) -> bool: return self._impl.store_thinking_step(*a, **kw)
    def get_thinking_session(self, *a, **kw) -> List[Dict]: return self._impl.get_thinking_session(*a, **kw)
    def log_performance(self, *a, **kw) -> bool: return getattr(self._impl, "log_performance", lambda *a, **kw: True)(*a, **kw)
    def get_performance_stats(self, *a, **kw) -> Dict[str, Any]: return getattr(self._impl, "get_performance_stats", lambda *a, **kw: {})(*a, **kw)
    def log_error(self, *a, **kw) -> bool: return getattr(self._impl, "log_error", lambda *a, **kw: True)(*a, **kw)
    def get_error_patterns(self, *a, **kw) -> List[Dict]: return getattr(self._impl, "get_error_patterns", lambda *a, **kw: [])(*a, **kw)
    def cleanup_old_data(self, *a, **kw) -> Dict[str, int]: return getattr(self._impl, "cleanup_old_data", lambda *a, **kw: {})(*a, **kw)
    def get_storage_stats(self, *a, **kw) -> Dict[str, Any]: return getattr(self._impl, "get_storage_stats", lambda *a, **kw: {})(*a, **kw)

