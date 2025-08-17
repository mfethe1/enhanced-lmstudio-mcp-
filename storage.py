import sqlite3
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MCPStorage:
    """Persistent storage layer for MCP server with SQLite backend"""
    
    def __init__(self, db_path: str = "mcp_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            # Memory storage table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    timestamp REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Thinking sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thinking_sessions (
                    session_id TEXT NOT NULL,
                    thought_number INTEGER NOT NULL,
                    thought TEXT NOT NULL,
                    is_revision BOOLEAN DEFAULT FALSE,
                    revises_thought INTEGER,
                    branch_id TEXT,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (session_id, thought_number)
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    timestamp REAL NOT NULL,
                    request_size INTEGER,
                    response_size INTEGER
                )
            """)
            
            # Error tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    context TEXT,
                    tool_name TEXT,
                    stack_trace TEXT,
                    timestamp REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thinking_session ON thinking_sessions(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_tool ON performance_metrics(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
            
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # Memory Management Methods
    def store_memory(self, key: str, value: str, category: str = "general", metadata: Dict = None) -> bool:
        """Store memory entry with metadata"""
        try:
            metadata_json = json.dumps(metadata or {})
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory 
                    (key, value, category, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, value, category, time.time(), metadata_json))
            return True
        except Exception as e:
            logger.error(f"Failed to store memory {key}: {e}")
            return False
    
    def retrieve_memory(self, key: Optional[str] = None, category: Optional[str] = None, 
                       search_term: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Retrieve memory entries with flexible filtering"""
        try:
            with self._get_connection() as conn:
                if key:
                    cursor = conn.execute("SELECT * FROM memory WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    if row:
                        return [dict(row)]
                    return []
                
                # Build dynamic query
                query = "SELECT * FROM memory WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if search_term:
                    query += " AND (key LIKE ? OR value LIKE ?)"
                    params.extend([f"%{search_term}%", f"%{search_term}%"])
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return []
    
    def delete_memory(self, key: str) -> bool:
        """Delete memory entry by key"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete memory {key}: {e}")
            return False
    
    # Thinking Sessions Management
    def store_thinking_step(self, session_id: str, thought_number: int, thought: str,
                           is_revision: bool = False, revises_thought: Optional[int] = None,
                           branch_id: Optional[str] = None) -> bool:
        """Store a thinking step in a session"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO thinking_sessions
                    (session_id, thought_number, thought, is_revision, revises_thought, branch_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, thought_number, thought, is_revision, revises_thought, branch_id, time.time()))
            return True
        except Exception as e:
            logger.error(f"Failed to store thinking step: {e}")
            return False
    
    def get_thinking_session(self, session_id: str) -> List[Dict]:
        """Get all thoughts in a session"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM thinking_sessions 
                    WHERE session_id = ? 
                    ORDER BY thought_number
                """, (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get thinking session {session_id}: {e}")
            return []
    
    # Performance Metrics
    def log_performance(self, tool_name: str, execution_time: float, success: bool,
                       error_message: Optional[str] = None, request_size: int = 0,
                       response_size: int = 0) -> bool:
        """Log performance metrics for a tool execution"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_metrics
                    (tool_name, execution_time, success, error_message, timestamp, request_size, response_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (tool_name, execution_time, success, error_message, time.time(), request_size, response_size))
            return True
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")
            return False
    
    def get_performance_stats(self, tool_name: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for tools"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self._get_connection() as conn:
                query = """
                    SELECT 
                        tool_name,
                        COUNT(*) as total_calls,
                        AVG(execution_time) as avg_time,
                        MAX(execution_time) as max_time,
                        MIN(execution_time) as min_time,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                        COUNT(*) - SUM(CASE WHEN success THEN 1 ELSE 0 END) as error_count
                    FROM performance_metrics 
                    WHERE timestamp > ?
                """
                params = [cutoff_time]
                
                if tool_name:
                    query += " AND tool_name = ?"
                    params.append(tool_name)
                
                query += " GROUP BY tool_name"
                
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Calculate success rates
                for result in results:
                    total = result['total_calls']
                    result['success_rate'] = result['success_count'] / total if total > 0 else 0
                    result['error_rate'] = result['error_count'] / total if total > 0 else 0
                
                return {
                    'stats': results,
                    'period_hours': hours,
                    'total_tools': len(results)
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    # Error Tracking
    def log_error(self, error_type: str, error_message: str, context: Optional[str] = None,
                  tool_name: Optional[str] = None, stack_trace: Optional[str] = None) -> bool:
        """Log an error for tracking and analysis"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO error_log
                    (error_type, error_message, context, tool_name, stack_trace, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (error_type, error_message, context, tool_name, stack_trace, time.time()))
            return True
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
            return False
    
    def get_error_patterns(self, hours: int = 24) -> List[Dict]:
        """Get error patterns for analysis"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        error_type,
                        tool_name,
                        COUNT(*) as occurrence_count,
                        MAX(timestamp) as last_occurrence,
                        MIN(timestamp) as first_occurrence
                    FROM error_log 
                    WHERE timestamp > ?
                    GROUP BY error_type, tool_name
                    ORDER BY occurrence_count DESC
                """, (cutoff_time,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get error patterns: {e}")
            return []
    
    # Maintenance and Cleanup
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Clean up old data to maintain performance"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            deleted_counts = {}
            
            with self._get_connection() as conn:
                # Clean old performance metrics
                cursor = conn.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
                deleted_counts['performance_metrics'] = cursor.rowcount
                
                # Clean old thinking sessions (keep recent ones)
                cursor = conn.execute("DELETE FROM thinking_sessions WHERE timestamp < ?", (cutoff_time,))
                deleted_counts['thinking_sessions'] = cursor.rowcount
                
                # Clean resolved errors older than cutoff
                cursor = conn.execute("""
                    DELETE FROM error_log 
                    WHERE timestamp < ? AND resolved = TRUE
                """, (cutoff_time,))
                deleted_counts['error_log'] = cursor.rowcount
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                
            logger.info(f"Cleaned up old data: {deleted_counts}")
            return deleted_counts
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for monitoring"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Table sizes
                for table in ['memory', 'thinking_sessions', 'performance_metrics', 'error_log']:
                    cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()['count']
                
                # Database file size
                stats['db_file_size'] = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                # Recent activity (last 24 hours)
                cutoff_time = time.time() - (24 * 3600)
                cursor = conn.execute("SELECT COUNT(*) as count FROM performance_metrics WHERE timestamp > ?", (cutoff_time,))
                stats['recent_performance_logs'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {} 