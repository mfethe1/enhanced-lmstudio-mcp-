# Enhanced Storage Layer for Agent Collaboration and MCP Integration
import sqlite3
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import uuid

@dataclass
class ContextEnvelope:
    """MCP-compatible context envelope for persistent context"""
    envelope_id: str
    session_id: str
    context_type: str  # 'research', 'collaboration', 'task', 'file'
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    parent_envelope_id: Optional[str]
    created_at: float
    updated_at: float
    access_count: int = 0
    ttl_seconds: Optional[int] = None

@dataclass
class AgentInteraction:
    """Record of agent-to-agent interactions"""
    interaction_id: str
    session_id: str
    source_agent: str
    target_agent: str
    interaction_type: str  # 'delegation', 'question', 'review', 'synthesis'
    content: str
    context_envelope_id: Optional[str]
    response: Optional[str]
    success: bool
    created_at: float
    duration_ms: int

@dataclass
class CollaborativeArtifact:
    """Shared artifacts created during collaboration"""
    artifact_id: str
    session_id: str
    artifact_type: str  # 'research_summary', 'code_patch', 'analysis', 'plan'
    title: str
    content: str
    contributors: List[str]
    version: int
    parent_artifact_id: Optional[str]
    status: str  # 'draft', 'review', 'approved', 'deprecated'
    tags: List[str]
    created_at: float
    updated_at: float

class EnhancedMCPStorage:
    """Enhanced storage layer optimized for MCP and agent collaboration"""
    
    def __init__(self, db_path: str = "enhanced_mcp_storage.db"):
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._init_database()
        self._create_indexes()

    def _init_database(self):
        """Initialize database with enhanced schemas"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Context envelopes for MCP compatibility
                CREATE TABLE IF NOT EXISTS context_envelopes (
                    envelope_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    content TEXT NOT NULL,  -- JSON
                    metadata TEXT NOT NULL, -- JSON
                    parent_envelope_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    FOREIGN KEY (parent_envelope_id) REFERENCES context_envelopes(envelope_id)
                );

                -- Agent interactions tracking
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    target_agent TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context_envelope_id TEXT,
                    response TEXT,
                    success BOOLEAN DEFAULT FALSE,
                    created_at REAL NOT NULL,
                    duration_ms INTEGER,
                    FOREIGN KEY (context_envelope_id) REFERENCES context_envelopes(envelope_id)
                );

                -- Collaborative artifacts
                CREATE TABLE IF NOT EXISTS collaborative_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    contributors TEXT NOT NULL, -- JSON array
                    version INTEGER DEFAULT 1,
                    parent_artifact_id TEXT,
                    status TEXT DEFAULT 'draft',
                    tags TEXT DEFAULT '[]', -- JSON array
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (parent_artifact_id) REFERENCES collaborative_artifacts(artifact_id)
                );

                -- Session metadata
                CREATE TABLE IF NOT EXISTS collaboration_sessions (
                    session_id TEXT PRIMARY KEY,
                    session_type TEXT NOT NULL,
                    participants TEXT NOT NULL, -- JSON array
                    initial_task TEXT NOT NULL,
                    complexity_analysis TEXT, -- JSON
                    status TEXT DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    total_interactions INTEGER DEFAULT 0,
                    total_artifacts INTEGER DEFAULT 0
                );

                -- Agent performance metrics
                CREATE TABLE IF NOT EXISTS agent_performance (
                    metric_id TEXT PRIMARY KEY,
                    agent_role TEXT NOT NULL,
                    backend_used TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    complexity_level TEXT NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_type TEXT,
                    quality_score REAL, -- 0.0 to 1.0
                    collaboration_rating REAL, -- 0.0 to 1.0
                    created_at REAL NOT NULL
                );

                -- Context lineage for Augment Code integration
                CREATE TABLE IF NOT EXISTS context_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    change_type TEXT NOT NULL, -- 'create', 'modify', 'delete'
                    agent_responsible TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    context_before TEXT, -- JSON snapshot
                    context_after TEXT,  -- JSON snapshot
                    reasoning TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                -- MCP tool usage tracking
                CREATE TABLE IF NOT EXISTS mcp_tool_usage (
                    usage_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    parameters TEXT NOT NULL, -- JSON
                    result_type TEXT NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_details TEXT,
                    created_at REAL NOT NULL
                );
            """)

    def _create_indexes(self):
        """Create indexes for better query performance"""
        with sqlite3.connect(self.db_path) as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_context_session ON context_envelopes(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_context_type ON context_envelopes(context_type);",
                "CREATE INDEX IF NOT EXISTS idx_context_created ON context_envelopes(created_at);",
                "CREATE INDEX IF NOT EXISTS idx_interactions_session ON agent_interactions(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_interactions_agents ON agent_interactions(source_agent, target_agent);",
                "CREATE INDEX IF NOT EXISTS idx_artifacts_session ON collaborative_artifacts(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON collaborative_artifacts(artifact_type);",
                "CREATE INDEX IF NOT EXISTS idx_sessions_status ON collaboration_sessions(status);",
                "CREATE INDEX IF NOT EXISTS idx_performance_agent ON agent_performance(agent_role);",
                "CREATE INDEX IF NOT EXISTS idx_performance_created ON agent_performance(created_at);",
                "CREATE INDEX IF NOT EXISTS idx_lineage_file ON context_lineage(file_path);",
                "CREATE INDEX IF NOT EXISTS idx_lineage_session ON context_lineage(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON mcp_tool_usage(tool_name);",
                "CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON mcp_tool_usage(session_id);"
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)

    # Context Envelope Management (MCP Compatible)
    def create_context_envelope(self, session_id: str, context_type: str, 
                              content: Dict[str, Any], metadata: Dict[str, Any] = None,
                              parent_envelope_id: str = None, ttl_seconds: int = None) -> str:
        """Create a new context envelope with MCP compatibility"""
        envelope_id = f"env_{uuid.uuid4().hex[:16]}"
        metadata = metadata or {}
        now = time.time()
        
        envelope = ContextEnvelope(
            envelope_id=envelope_id,
            session_id=session_id,
            context_type=context_type,
            content=content,
            metadata=metadata,
            parent_envelope_id=parent_envelope_id,
            created_at=now,
            updated_at=now,
            ttl_seconds=ttl_seconds
        )
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_envelopes 
                    (envelope_id, session_id, context_type, content, metadata, 
                     parent_envelope_id, created_at, updated_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    envelope.envelope_id,
                    envelope.session_id,
                    envelope.context_type,
                    json.dumps(envelope.content),
                    json.dumps(envelope.metadata),
                    envelope.parent_envelope_id,
                    envelope.created_at,
                    envelope.updated_at,
                    envelope.ttl_seconds
                ))
        
        return envelope_id

    def get_context_envelope(self, envelope_id: str) -> Optional[ContextEnvelope]:
        """Retrieve a context envelope by ID"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM context_envelopes 
                    WHERE envelope_id = ?
                """, (envelope_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Update access count
                conn.execute("""
                    UPDATE context_envelopes 
                    SET access_count = access_count + 1 
                    WHERE envelope_id = ?
                """, (envelope_id,))
                
                return ContextEnvelope(
                    envelope_id=row['envelope_id'],
                    session_id=row['session_id'],
                    context_type=row['context_type'],
                    content=json.loads(row['content']),
                    metadata=json.loads(row['metadata']),
                    parent_envelope_id=row['parent_envelope_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    access_count=row['access_count'] + 1,
                    ttl_seconds=row['ttl_seconds']
                )

    def get_session_context_chain(self, session_id: str) -> List[ContextEnvelope]:
        """Get all context envelopes for a session in chronological order"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM context_envelopes 
                    WHERE session_id = ? 
                    ORDER BY created_at ASC
                """, (session_id,))
                
                envelopes = []
                for row in cursor.fetchall():
                    envelopes.append(ContextEnvelope(
                        envelope_id=row['envelope_id'],
                        session_id=row['session_id'],
                        context_type=row['context_type'],
                        content=json.loads(row['content']),
                        metadata=json.loads(row['metadata']),
                        parent_envelope_id=row['parent_envelope_id'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        access_count=row['access_count'],
                        ttl_seconds=row['ttl_seconds']
                    ))
                
                return envelopes

    # Agent Interaction Tracking
    def log_agent_interaction(self, session_id: str, source_agent: str, target_agent: str,
                            interaction_type: str, content: str, context_envelope_id: str = None,
                            response: str = None, success: bool = True, duration_ms: int = 0) -> str:
        """Log an interaction between agents"""
        interaction_id = f"int_{uuid.uuid4().hex[:16]}"
        now = time.time()
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            session_id=session_id,
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            content=content,
            context_envelope_id=context_envelope_id,
            response=response,
            success=success,
            created_at=now,
            duration_ms=duration_ms
        )
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO agent_interactions 
                    (interaction_id, session_id, source_agent, target_agent, 
                     interaction_type, content, context_envelope_id, response, 
                     success, created_at, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.interaction_id,
                    interaction.session_id,
                    interaction.source_agent,
                    interaction.target_agent,
                    interaction.interaction_type,
                    interaction.content,
                    interaction.context_envelope_id,
                    interaction.response,
                    interaction.success,
                    interaction.created_at,
                    interaction.duration_ms
                ))
        
        return interaction_id

    # Collaborative Artifacts Management
    def create_artifact(self, session_id: str, artifact_type: str, title: str,
                       content: str, contributors: List[str], tags: List[str] = None,
                       parent_artifact_id: str = None) -> str:
        """Create a new collaborative artifact"""
        artifact_id = f"art_{uuid.uuid4().hex[:16]}"
        now = time.time()
        tags = tags or []
        
        artifact = CollaborativeArtifact(
            artifact_id=artifact_id,
            session_id=session_id,
            artifact_type=artifact_type,
            title=title,
            content=content,
            contributors=contributors,
            version=1,
            parent_artifact_id=parent_artifact_id,
            status='draft',
            tags=tags,
            created_at=now,
            updated_at=now
        )
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO collaborative_artifacts 
                    (artifact_id, session_id, artifact_type, title, content, 
                     contributors, version, parent_artifact_id, status, tags, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    artifact.artifact_id,
                    artifact.session_id,
                    artifact.artifact_type,
                    artifact.title,
                    artifact.content,
                    json.dumps(artifact.contributors),
                    artifact.version,
                    artifact.parent_artifact_id,
                    artifact.status,
                    json.dumps(artifact.tags),
                    artifact.created_at,
                    artifact.updated_at
                ))
        
        return artifact_id

    def update_artifact(self, artifact_id: str, content: str = None, contributors: List[str] = None,
                       status: str = None, tags: List[str] = None) -> bool:
        """Update an existing artifact"""
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if contributors is not None:
            updates.append("contributors = ?")
            params.append(json.dumps(contributors))
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        updates.append("version = version + 1")
        params.append(time.time())
        params.append(artifact_id)
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"""
                    UPDATE collaborative_artifacts 
                    SET {', '.join(updates)}
                    WHERE artifact_id = ?
                """, params)
                
                return conn.rowcount > 0

    # Context Lineage for Augment Code Integration
    def log_context_change(self, file_path: str, change_type: str, agent_responsible: str,
                          session_id: str, context_before: Dict[str, Any] = None,
                          context_after: Dict[str, Any] = None, reasoning: str = "") -> str:
        """Log context changes for Augment Code lineage tracking"""
        lineage_id = f"lin_{uuid.uuid4().hex[:16]}"
        now = time.time()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_lineage 
                    (lineage_id, file_path, change_type, agent_responsible, 
                     session_id, context_before, context_after, reasoning, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    lineage_id,
                    file_path,
                    change_type,
                    agent_responsible,
                    session_id,
                    json.dumps(context_before) if context_before else None,
                    json.dumps(context_after) if context_after else None,
                    reasoning,
                    now
                ))
        
        return lineage_id

    # Performance Analytics
    def log_agent_performance(self, agent_role: str, backend_used: str, task_type: str,
                            complexity_level: str, execution_time_ms: int, success: bool,
                            error_type: str = None, quality_score: float = None,
                            collaboration_rating: float = None) -> str:
        """Log agent performance metrics"""
        metric_id = f"perf_{uuid.uuid4().hex[:16]}"
        now = time.time()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO agent_performance 
                    (metric_id, agent_role, backend_used, task_type, complexity_level, 
                     execution_time_ms, success, error_type, quality_score, 
                     collaboration_rating, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    agent_role,
                    backend_used,
                    task_type,
                    complexity_level,
                    execution_time_ms,
                    success,
                    error_type,
                    quality_score,
                    collaboration_rating,
                    now
                ))
        
        return metric_id

    # Cleanup and Maintenance
    def cleanup_expired_contexts(self) -> int:
        """Clean up expired context envelopes"""
        now = time.time()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM context_envelopes 
                    WHERE ttl_seconds IS NOT NULL 
                    AND (created_at + ttl_seconds) < ?
                """, (now,))
                
                return cursor.rowcount

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a collaboration session"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Session overview
                session_cursor = conn.execute("""
                    SELECT * FROM collaboration_sessions WHERE session_id = ?
                """, (session_id,))
                session_row = session_cursor.fetchone()
                
                if not session_row:
                    return {}
                
                # Interaction stats
                interaction_cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(duration_ms) as avg_duration,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_interactions
                    FROM agent_interactions 
                    WHERE session_id = ?
                """, (session_id,))
                interaction_stats = interaction_cursor.fetchone()
                
                # Artifact stats
                artifact_cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_artifacts,
                        COUNT(DISTINCT artifact_type) as unique_types,
                        MAX(version) as max_version
                    FROM collaborative_artifacts 
                    WHERE session_id = ?
                """, (session_id,))
                artifact_stats = artifact_cursor.fetchone()
                
                return {
                    'session_id': session_id,
                    'status': session_row['status'],
                    'participants': json.loads(session_row['participants']),
                    'duration_minutes': (time.time() - session_row['created_at']) / 60,
                    'interactions': {
                        'total': interaction_stats['total_interactions'],
                        'average_duration_ms': interaction_stats['avg_duration'],
                        'success_rate': interaction_stats['successful_interactions'] / max(interaction_stats['total_interactions'], 1)
                    },
                    'artifacts': {
                        'total': artifact_stats['total_artifacts'],
                        'unique_types': artifact_stats['unique_types'],
                        'max_version': artifact_stats['max_version']
                    }
                }