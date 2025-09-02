import os
import time
import json
import logging
from typing import Dict, List, Optional, Any

import psycopg

logger = logging.getLogger(__name__)

class MCPStoragePostgres:
    """Postgres implementation of MCP storage interface compatible with MCPStorage."""
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "")
        if not self.dsn:
            raise RuntimeError("POSTGRES_DSN is required for MCPStoragePostgres")
        self._init_db()

    def _conn(self):
        return psycopg.connect(self.dsn)

    def _init_db(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        category TEXT DEFAULT 'general',
                        timestamp DOUBLE PRECISION NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS thinking_sessions (
                        session_id TEXT NOT NULL,
                        thought_number INTEGER NOT NULL,
                        thought TEXT NOT NULL,
                        is_revision BOOLEAN DEFAULT FALSE,
                        revises_thought INTEGER,
                        branch_id TEXT,
                        timestamp DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (session_id, thought_number)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        execution_time DOUBLE PRECISION NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        timestamp DOUBLE PRECISION NOT NULL,
                        request_size INTEGER,
                        response_size INTEGER
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS error_log (
                        id SERIAL PRIMARY KEY,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        context TEXT,
                        tool_name TEXT,
                        stack_trace TEXT,
                        timestamp DOUBLE PRECISION NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE
                    );
                    """
                )
            conn.commit()

    # Memory Management
    def store_memory(self, key: str, value: str, category: str = "general", metadata: Dict | None = None) -> bool:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory (key, value, category, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        category = EXCLUDED.category,
                        timestamp = EXCLUDED.timestamp,
                        metadata = EXCLUDED.metadata
                    """,
                    (key, value, category, time.time(), json.dumps(metadata or {}))
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store memory {key}: {e}")
            return False

    def retrieve_memory(self, key: Optional[str] = None, category: Optional[str] = None,
                        search_term: Optional[str] = None, limit: int = 50) -> List[Dict]:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                if key:
                    cur.execute("SELECT key, value, category, timestamp, metadata FROM memory WHERE key = %s", (key,))
                    row = cur.fetchone()
                    if row:
                        k, v, c, ts, md = row
                        return [{"key": k, "value": v, "category": c, "timestamp": ts, "metadata": md}]
                    return []
                query = "SELECT key, value, category, timestamp, metadata FROM memory WHERE 1=1"
                params: list[Any] = []
                if category:
                    query += " AND category = %s"
                    params.append(category)
                if search_term:
                    query += " AND (key ILIKE %s OR value ILIKE %s)"
                    like = f"%{search_term}%"
                    params.extend([like, like])
                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)
                cur.execute(query, params)
                rows = cur.fetchall()
                out = []
                for k, v, c, ts, md in rows:
                    out.append({"key": k, "value": v, "category": c, "timestamp": ts, "metadata": md})
                return out
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return []

    def delete_memory(self, key: str) -> bool:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM memory WHERE key = %s", (key,))
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete memory {key}: {e}")
            return False

    # Thinking Sessions
    def store_thinking_step(self, session_id: str, thought_number: int, thought: str,
                            is_revision: bool = False, revises_thought: Optional[int] = None,
                            branch_id: Optional[str] = None) -> bool:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO thinking_sessions (session_id, thought_number, thought, is_revision, revises_thought, branch_id, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, thought_number) DO UPDATE SET
                        thought = EXCLUDED.thought,
                        is_revision = EXCLUDED.is_revision,
                        revises_thought = EXCLUDED.revises_thought,
                        branch_id = EXCLUDED.branch_id,
                        timestamp = EXCLUDED.timestamp
                    """,
                    (session_id, thought_number, thought, is_revision, revises_thought, branch_id, time.time())
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store thinking step: {e}")
            return False

    def get_thinking_session(self, session_id: str) -> List[Dict]:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, thought_number, thought, is_revision, revises_thought, branch_id, timestamp
                    FROM thinking_sessions WHERE session_id = %s ORDER BY thought_number
                    """,
                    (session_id,)
                )
                rows = cur.fetchall()
                return [
                    {
                        "session_id": r[0],
                        "thought_number": r[1],
                        "thought": r[2],
                        "is_revision": r[3],
                        "revises_thought": r[4],
                        "branch_id": r[5],
                        "timestamp": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get thinking session {session_id}: {e}")
            return []

    # Performance Metrics and Error Log can be implemented similarly if needed for remote scale.

