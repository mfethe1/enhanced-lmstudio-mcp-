"""
Enhanced Agent Memory System with Context Persistence and Learning
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import sqlite3
import threading
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    entry_id: str
    agent_id: str
    content: str
    memory_type: str  # episodic, semantic, procedural, working
    importance: float
    timestamp: float
    access_count: int
    last_accessed: float
    tags: List[str]
    embedding: Optional[List[float]] = None
    related_entries: List[str] = None

@dataclass
class ContextWindow:
    window_id: str
    agent_id: str
    context_type: str
    content: Dict[str, Any]
    active_span: Tuple[float, float]  # start_time, end_time
    importance_decay: float
    retrieval_triggers: List[str]

class EnhancedAgentMemory:
    """Advanced memory system with episodic, semantic, and procedural memory"""
    
    def __init__(self, agent_id: str, db_path: str = ":memory:"):
        self.agent_id = agent_id
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Working memory (immediate context)
        self.working_memory = deque(maxlen=20)
        
        # Context windows for different time horizons
        self.context_windows: Dict[str, ContextWindow] = {}
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.forgetting_curve_factor = 0.1
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for persistent memory"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL,
                    tags TEXT,
                    embedding BLOB
                );
                
                CREATE TABLE IF NOT EXISTS memory_relations (
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    strength REAL,
                    FOREIGN KEY (source_id) REFERENCES memory_entries(entry_id),
                    FOREIGN KEY (target_id) REFERENCES memory_entries(entry_id)
                );
                
                CREATE TABLE IF NOT EXISTS context_windows (
                    window_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    importance_decay REAL DEFAULT 0.1,
                    retrieval_triggers TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_agent_time ON memory_entries(agent_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance DESC);
                CREATE INDEX IF NOT EXISTS idx_context_windows_agent ON context_windows(agent_id, context_type);
            """)
    
    async def store_memory(self, content: str, memory_type: str = "episodic", 
                          importance: float = 0.5, tags: List[str] = None) -> str:
        """Store a new memory entry with importance weighting"""
        
        entry_id = f"mem_{int(time.time() * 1000)}_{hash(content) % 10000}"
        current_time = time.time()
        
        # Calculate importance based on recency, content, and context
        adjusted_importance = await self._calculate_importance(content, importance)
        
        memory_entry = MemoryEntry(
            entry_id=entry_id,
            agent_id=self.agent_id,
            content=content,
            memory_type=memory_type,
            importance=adjusted_importance,
            timestamp=current_time,
            access_count=0,
            last_accessed=current_time,
            tags=tags or [],
            related_entries=[]
        )
        
        with self.lock:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memory_entries 
                    (entry_id, agent_id, content, memory_type, importance, timestamp, 
                     last_accessed, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id, self.agent_id, content, memory_type, 
                    adjusted_importance, current_time, current_time, 
                    json.dumps(tags or [])
                ))
            
            # Add to working memory if important enough
            if adjusted_importance > 0.6:
                self.working_memory.append(memory_entry)
        
        # Trigger consolidation if needed
        await self._consolidate_memories()
        
        return entry_id
    
    async def retrieve_memories(self, query: str, memory_types: List[str] = None, 
                              limit: int = 10, min_importance: float = 0.1) -> List[MemoryEntry]:
        """Retrieve relevant memories based on query and importance"""
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Build query
                where_conditions = ["agent_id = ?"]
                params = [self.agent_id]
                
                if memory_types:
                    where_conditions.append(f"memory_type IN ({','.join(['?'] * len(memory_types))})")
                    params.extend(memory_types)
                
                if min_importance > 0:
                    where_conditions.append("importance >= ?")
                    params.append(min_importance)
                
                # Simple text search (in production, use vector similarity)
                where_conditions.append("content LIKE ?")
                params.append(f"%{query}%")
                
                query_sql = f"""
                    SELECT entry_id, agent_id, content, memory_type, importance, 
                           timestamp, access_count, last_accessed, tags
                    FROM memory_entries 
                    WHERE {' AND '.join(where_conditions)}
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(query_sql, params)
                rows = cursor.fetchall()
                
                memories = []
                current_time = time.time()
                
                for row in rows:
                    # Apply forgetting curve
                    time_factor = np.exp(-self.forgetting_curve_factor * (current_time - row[5]))
                    adjusted_importance = row[4] * time_factor
                    
                    if adjusted_importance > min_importance:
                        memory = MemoryEntry(
                            entry_id=row[0],
                            agent_id=row[1],
                            content=row[2],
                            memory_type=row[3],
                            importance=adjusted_importance,
                            timestamp=row[5],
                            access_count=row[6],
                            last_accessed=row[7],
                            tags=json.loads(row[8] or "[]")
                        )
                        memories.append(memory)
                        
                        # Update access count
                        conn.execute("""
                            UPDATE memory_entries 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE entry_id = ?
                        """, (current_time, row[0]))
                
                return memories
    
    async def create_context_window(self, context_type: str, content: Dict[str, Any], 
                                  duration: float = 3600.0) -> str:
        """Create a context window for maintaining relevant information"""
        
        window_id = f"ctx_{int(time.time() * 1000)}_{context_type}"
        current_time = time.time()
        
        context_window = ContextWindow(
            window_id=window_id,
            agent_id=self.agent_id,
            context_type=context_type,
            content=content,
            active_span=(current_time, current_time + duration),
            importance_decay=0.1,
            retrieval_triggers=list(content.keys())
        )
        
        self.context_windows[window_id] = context_window
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO context_windows 
                (window_id, agent_id, context_type, content, start_time, end_time, 
                 importance_decay, retrieval_triggers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                window_id, self.agent_id, context_type, 
                json.dumps(content), current_time, current_time + duration,
                0.1, json.dumps(list(content.keys()))
            ))
        
        return window_id
    
    async def get_active_context(self, context_types: List[str] = None) -> Dict[str, Any]:
        """Get currently active context information"""
        
        active_context = {}
        current_time = time.time()
        
        # Check active context windows
        for window in self.context_windows.values():
            if (window.active_span[0] <= current_time <= window.active_span[1] and 
                (context_types is None or window.context_type in context_types)):
                
                # Apply time-based decay
                time_factor = 1.0 - ((current_time - window.active_span[0]) / 
                                   (window.active_span[1] - window.active_span[0]))
                
                for key, value in window.content.items():
                    active_context[f"{window.context_type}_{key}"] = {
                        "value": value,
                        "importance": time_factor,
                        "source": window.window_id
                    }
        
        # Add working memory context
        working_context = {}
        for memory in list(self.working_memory)[-5:]:  # Last 5 working memories
            working_context[f"recent_{memory.entry_id}"] = {
                "content": memory.content,
                "importance": memory.importance,
                "type": memory.memory_type
            }
        
        active_context["working_memory"] = working_context
        
        return active_context
    
    async def _calculate_importance(self, content: str, base_importance: float) -> float:
        """Calculate memory importance based on content and context"""
        
        # Factors that increase importance
        importance_factors = {
            "error": 0.3,
            "success": 0.2,
            "failure": 0.3,
            "learn": 0.2,
            "improve": 0.2,
            "critical": 0.4,
            "important": 0.3,
        }
        
        content_lower = content.lower()
        importance_boost = 0.0
        
        for keyword, boost in importance_factors.items():
            if keyword in content_lower:
                importance_boost += boost
        
        # Consider recency (recent memories slightly more important)
        recency_boost = 0.1
        
        # Check if content relates to current working memory
        context_boost = 0.0
        for memory in list(self.working_memory):
            if any(word in content_lower for word in memory.content.lower().split()[-10:]):
                context_boost += 0.1
        
        final_importance = min(1.0, base_importance + importance_boost + recency_boost + context_boost)
        
        return final_importance
    
    async def _consolidate_memories(self):
        """Consolidate memories based on importance and relations"""
        
        current_time = time.time()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Find memories that should be consolidated
                cursor = conn.execute("""
                    SELECT entry_id, content, importance, timestamp 
                    FROM memory_entries 
                    WHERE agent_id = ? AND importance > ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (self.agent_id, self.consolidation_threshold))
                
                memories = cursor.fetchall()
                
                # Simple consolidation: group similar recent memories
                consolidated_groups = defaultdict(list)
                
                for memory in memories:
                    # Group by common keywords (simple approach)
                    keywords = set(word.lower() for word in memory[1].split() if len(word) > 4)
                    group_key = tuple(sorted(keywords)[:3]) if keywords else "general"
                    consolidated_groups[group_key].append(memory)
                
                # Create consolidated memories for large groups
                for group_key, group_memories in consolidated_groups.items():
                    if len(group_memories) > 3:
                        consolidated_content = f"Consolidated memory group: {', '.join(str(group_key))}\n"
                        consolidated_content += "\n".join([mem[1][:100] + "..." for mem in group_memories[:3]])
                        
                        avg_importance = sum(mem[2] for mem in group_memories) / len(group_memories)
                        
                        # Store consolidated memory
                        consolidated_id = f"cons_{int(current_time)}_{hash(consolidated_content) % 1000}"
                        conn.execute("""
                            INSERT INTO memory_entries 
                            (entry_id, agent_id, content, memory_type, importance, 
                             timestamp, last_accessed, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            consolidated_id, self.agent_id, consolidated_content, 
                            "consolidated", avg_importance, current_time, current_time,
                            json.dumps(["consolidated", str(group_key)])
                        ))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count memories by type
            cursor = conn.execute("""
                SELECT memory_type, COUNT(*), AVG(importance), AVG(access_count)
                FROM memory_entries 
                WHERE agent_id = ?
                GROUP BY memory_type
            """, (self.agent_id,))
            
            stats["by_type"] = {}
            for row in cursor.fetchall():
                stats["by_type"][row[0]] = {
                    "count": row[1],
                    "avg_importance": row[2],
                    "avg_access": row[3]
                }
            
            # Active context windows
            active_windows = sum(1 for w in self.context_windows.values() 
                               if w.active_span[1] > time.time())
            stats["active_context_windows"] = active_windows
            
            # Working memory size
            stats["working_memory_size"] = len(self.working_memory)
            
            return stats


class ContextAwarePromptBuilder:
    """Build context-aware prompts using agent memory and active context"""
    
    def __init__(self, memory_system: EnhancedAgentMemory):
        self.memory = memory_system
        
    async def build_prompt(self, base_prompt: str, context_types: List[str] = None,
                          include_memory: bool = True, max_context_length: int = 2000) -> str:
        """Build an enhanced prompt with relevant context and memory"""
        
        enhanced_prompt = base_prompt
        context_sections = []
        
        if include_memory:
            # Get relevant memories
            memories = await self.memory.retrieve_memories(
                base_prompt, memory_types=["episodic", "semantic"], limit=5
            )
            
            if memories:
                memory_context = "Relevant past experiences:\n"
                for memory in memories[:3]:
                    memory_context += f"- {memory.content[:150]}...\n"
                context_sections.append(memory_context)
        
        # Get active context
        active_context = await self.memory.get_active_context(context_types)
        
        if active_context:
            context_info = "Current context:\n"
            for key, context_item in active_context.items():
                if key != "working_memory":  # Handle working memory separately
                    context_info += f"- {key}: {str(context_item.get('value', ''))[:100]}\n"
                    
            if len(context_info) > 50:  # Only add if we have meaningful context
                context_sections.append(context_info)
        
        # Add working memory if available
        if "working_memory" in active_context:
            working_mem = active_context["working_memory"]
            if working_mem:
                recent_context = "Recent thoughts:\n"
                for mem_key, mem_item in list(working_mem.items())[-3:]:
                    recent_context += f"- {mem_item['content'][:100]}...\n"
                context_sections.append(recent_context)
        
        # Combine all context sections
        full_context = "\n\n".join(context_sections)
        
        # Trim context if too long
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "...\n[Context truncated]"
        
        # Build final prompt
        if full_context:
            enhanced_prompt = f"{full_context}\n\n{'='*50}\n\n{base_prompt}"
        
        return enhanced_prompt
    
    async def build_reflection_prompt(self, original_response: str, 
                                    context: Dict[str, Any]) -> str:
        """Build a prompt for self-reflection"""
        
        # Get memories related to similar situations
        reflection_memories = await self.memory.retrieve_memories(
            original_response, memory_types=["episodic"], limit=3
        )
        
        similar_experiences = ""
        if reflection_memories:
            similar_experiences = "Similar past experiences to consider:\n"
            for memory in reflection_memories:
                similar_experiences += f"- {memory.content[:100]}...\n"
        
        reflection_prompt = f"""
        {similar_experiences}
        
        Current situation:
        Context: {json.dumps(context, indent=2)[:500]}
        Your response: {original_response}
        
        Reflect on your response:
        1. What assumptions did you make?
        2. What information might you have missed?
        3. How could you improve this response?
        4. What would you do differently next time?
        5. What patterns do you notice compared to similar past experiences?
        
        Provide specific, actionable feedback:
        """
        
        return reflection_prompt


# Integration helper
def enhance_agent_with_memory(agent, db_path: str = None):
    """Enhance an existing agent with advanced memory capabilities"""
    
    if db_path is None:
        db_path = f"agent_memory_{agent.agent_id}.db"
    
    agent.memory_system = EnhancedAgentMemory(agent.agent_id, db_path)
    agent.prompt_builder = ContextAwarePromptBuilder(agent.memory_system)
    
    # Enhance the agent's process_message method
    original_process_message = agent.process_message
    
    async def enhanced_process_message(message: str, context: Dict[str, Any], 
                                     collaboration_session = None):
        # Store the incoming message in memory
        await agent.memory_system.store_memory(
            f"User request: {message}", "episodic", importance=0.6,
            tags=["user_input", "request"]
        )
        
        # Create context window for this interaction
        interaction_context = {
            "user_message": message,
            "timestamp": time.time(),
            "context_keys": list(context.keys())
        }
        
        await agent.memory_system.create_context_window(
            "interaction", interaction_context, duration=300  # 5 minutes
        )
        
        # Build enhanced prompt with memory context
        enhanced_message = await agent.prompt_builder.build_prompt(message)
        
        # Process with original method
        result = await original_process_message(enhanced_message, context, collaboration_session)
        
        # Store the agent's response in memory
        await agent.memory_system.store_memory(
            f"Agent response: {result.content}", "episodic", 
            importance=result.confidence, tags=["agent_output", "response"]
        )
        
        return result
    
    agent.process_message = enhanced_process_message
    
    return agent