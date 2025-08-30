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


# Backwards-compat shim for tests that import EnhancedMCPStorageV2
class EnhancedMCPStorageV2:
    def __init__(self, db_path: str | None = None):
        # mirror the SQLite enhanced implementation interface
        self.db_path = db_path or os.getenv("SQLITE_PATH", "enhanced_mcp_storage.db")
        self._ops = _SQLiteV2Ops(type("NS", (), {"db_path": self.db_path})())
    def store_memory(self, *a, **kw):
        return self._ops.store_memory(*a, **kw)
    def retrieve_memory(self, *a, **kw):
        return self._ops.retrieve_memory(*a, **kw)
    def delete_memory(self, *a, **kw):
        return self._ops.delete_memory(*a, **kw)

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
    def store_memory(self, *a, **kw) -> bool:
        if hasattr(self.impl, 'db_path'):
            return _SQLiteV2Ops(self.impl).store_memory(*a, **kw)
        return self.impl.store_memory(*a, **kw)

    def retrieve_memory(self, *a, **kw) -> List[Dict]:
        if hasattr(self.impl, 'db_path'):
            return _SQLiteV2Ops(self.impl).retrieve_memory(*a, **kw)
        return self.impl.retrieve_memory(*a, **kw)

    def delete_memory(self, *a, **kw) -> bool:
        if hasattr(self.impl, 'db_path'):
            return _SQLiteV2Ops(self.impl).delete_memory(*a, **kw)
        return self.impl.delete_memory(*a, **kw)

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


# --- V2 convenience wrappers when Enhanced SQLite is active ---
import sqlite3, json, time, uuid
from typing import List, Dict, Any, Optional

class _SQLiteV2Ops:
    def __init__(self, impl):
        self.impl = impl
        self.db_path = getattr(impl, 'db_path', None)
    def _conn(self):
        if not self.db_path:
            raise RuntimeError('V2 storage backend not available')
        return sqlite3.connect(self.db_path)

    # Legacy memory API compatibility on V2
    def _ensure_legacy_table(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legacy_memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )

    def _ensure_fts(self, conn):
        # Lightweight FTS5 index for better search over legacy memories
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS legacy_memories_fts USING fts5(key, value, category)")
        except Exception:
            # FTS5 may not be available in some builds; ignore
            pass

    def store_memory(self, key: str, value: str, category: str = "general") -> bool:
        with self._conn() as conn:
            self._ensure_legacy_table(conn)
            self._ensure_fts(conn)
            conn.execute(
                """INSERT OR REPLACE INTO legacy_memories (key, value, category, created_at) VALUES (?, ?, ?, ?)""",
                (str(key), str(value), str(category or "general"), time.time()),
            )
            try:
                # Update FTS shadow table
                conn.execute("INSERT INTO legacy_memories_fts(rowid, key, value, category) VALUES ((SELECT rowid FROM legacy_memories WHERE key=?), ?, ?, ?)"
                             , (str(key), str(key), str(value), str(category or "general")))
            except Exception:
                pass
        return True

    def retrieve_memory(self, key: Optional[str] = None, category: Optional[str] = None, search_term: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            self._ensure_legacy_table(conn)
            self._ensure_fts(conn)
            conn.row_factory = sqlite3.Row
            if key:
                cur = conn.execute("SELECT key, value, category, created_at FROM legacy_memories WHERE key = ?", (str(key),))
                row = cur.fetchone()
                if row:
                    return [{"key": row["key"], "value": row["value"], "category": row["category"], "timestamp": row["created_at"]}]
                return []
            # Prefer FTS when there's a search term
            if search_term:
                try:
                    # Basic FTS5 search by value/key; still filter by category if provided
                    if category:
                        cur = conn.execute(
                            """
                            SELECT lm.key, lm.value, lm.category, lm.created_at
                            FROM legacy_memories_fts f
                            JOIN legacy_memories lm ON lm.rowid = f.rowid
                            WHERE f.value MATCH ? AND lm.category = ?
                            ORDER BY lm.created_at DESC LIMIT ?
                            """,
                            (search_term, str(category), int(limit))
                        )
                    else:
                        cur = conn.execute(
                            """
                            SELECT lm.key, lm.value, lm.category, lm.created_at
                            FROM legacy_memories_fts f
                            JOIN legacy_memories lm ON lm.rowid = f.rowid
                            WHERE f.value MATCH ?
                            ORDER BY lm.created_at DESC LIMIT ?
                            """,
                            (search_term, int(limit))
                        )
                    return [
                        {"key": r["key"], "value": r["value"], "category": r["category"], "timestamp": r["created_at"]}
                        for r in cur.fetchall()
                    ]
                except Exception:
                    # Fallback to LIKE if FTS not available
                    pass
            # dynamic query fallback
            query = "SELECT key, value, category, created_at FROM legacy_memories WHERE 1=1"
            params: list[Any] = []
            if category:
                query += " AND category = ?"
                params.append(str(category))
            if search_term:
                like = f"%{search_term}%"
                query += " AND (key LIKE ? OR value LIKE ?)"
                params.extend([like, like])
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            cur = conn.execute(query, params)
            return [
                {"key": r["key"], "value": r["value"], "category": r["category"], "timestamp": r["created_at"]}
                for r in cur.fetchall()
            ]

    def delete_memory(self, key: str) -> bool:
        with self._conn() as conn:
            self._ensure_legacy_table(conn)
            conn.execute("""DELETE FROM legacy_memories WHERE key = ?""", (str(key),))
        return True

    def create_collaboration_session(self, session_id: str, session_type: str, participants: List[str], initial_task: str, complexity_analysis: Optional[Dict[str, Any]] = None, status: str = 'active') -> str:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO collaboration_sessions
                (session_id, session_type, participants, initial_task, complexity_analysis, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, session_type, json.dumps(participants), initial_task, json.dumps(complexity_analysis or {}), status, time.time(), time.time()))
        return session_id

    def log_tool_usage(self, tool_name: str, agent_role: str, session_id: str, parameters: Dict[str, Any], result_type: str, execution_time_ms: int, success: bool, error_details: Optional[str] = None) -> str:
        usage_id = f"use_{uuid.uuid4().hex[:16]}"
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO mcp_tool_usage
                (usage_id, tool_name, agent_role, session_id, parameters, result_type, execution_time_ms, success, error_details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (usage_id, tool_name, agent_role, session_id, json.dumps(parameters), result_type, int(execution_time_ms), 1 if success else 0, error_details, time.time()))
        return usage_id

    def list_artifacts(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM collaborative_artifacts
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, int(limit)))
            rows = cur.fetchall()
            out = []
            for r in rows:
                out.append({
                    'artifact_id': r['artifact_id'],
                    'artifact_type': r['artifact_type'],
                    'title': r['title'],
                    'content': r['content'],
                    'contributors': json.loads(r['contributors']),
                    'version': r['version'],
                    'status': r['status'],
                    'tags': json.loads(r['tags']),
                    'created_at': r['created_at'],
                    'updated_at': r['updated_at'],
                })
            return out

    def list_context_envelopes(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT envelope_id, context_type, content, metadata, created_at, updated_at
                FROM context_envelopes
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, int(limit)))
            return [
                {
                    'envelope_id': r['envelope_id'],
                    'context_type': r['context_type'],
                    'content': json.loads(r['content']),
                    'metadata': json.loads(r['metadata']),
                    'created_at': r['created_at'],
                    'updated_at': r['updated_at'],
                }
                for r in cur.fetchall()
            ]


# Extend the selector with V2 helpers
class StorageSelector(StorageSelector):  # type: ignore
    def _v2ops(self) -> _SQLiteV2Ops:
        return _SQLiteV2Ops(self.impl)

    # Safe helpers (no-op when V2 not active)
    def create_collaboration_session(self, *a, **kw):
        if hasattr(self.impl, 'db_path'):
            return self._v2ops().create_collaboration_session(*a, **kw)
        return kw.get('session_id') or (a[0] if a else 'sess_legacy')

    def log_tool_usage(self, *a, **kw):
        if hasattr(self.impl, 'db_path'):
            return self._v2ops().log_tool_usage(*a, **kw)
        return "use_legacy"

    def list_artifacts(self, *a, **kw):
        if hasattr(self.impl, 'db_path'):
            return self._v2ops().list_artifacts(*a, **kw)
        return []

    def list_context_envelopes(self, *a, **kw):
        if hasattr(self.impl, 'db_path'):
            return self._v2ops().list_context_envelopes(*a, **kw)
        return []
