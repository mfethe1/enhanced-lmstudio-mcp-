"""
Ingestion utilities for converting TeamMemory items and Firecrawl results into KG nodes/edges.
Supports vector embeddings when providers are enabled.
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .kg_store import KGNode, KGEdge, get_store
from .team_memory import TeamMemory

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    nodes_created: int
    edges_created: int
    errors: List[str]
    processing_time_ms: float

class KGIngestionEngine:
    """Converts various data sources into KG nodes and edges."""
    
    def __init__(self, server=None):
        self.server = server
        self.kg_store = get_store()
        self.embeddings_enabled = self._check_embeddings_available()
        
    def _check_embeddings_available(self) -> bool:
        """Check if embeddings are available through server providers."""
        if not self.server:
            return False
        
        # Check if any provider supports embeddings
        try:
            providers = getattr(self.server, 'providers', {})
            for provider_name, provider in providers.items():
                if hasattr(provider, 'embeddings') or 'embed' in provider_name.lower():
                    return True
        except Exception as e:
            logger.debug(f"Could not check embeddings availability: {e}")
        
        return False
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text if available."""
        if not self.embeddings_enabled or not self.server:
            return None
            
        try:
            # Try to use server's embedding capability
            if hasattr(self.server, 'generate_embedding'):
                return self.server.generate_embedding(text)
            
            # Fallback: use simple text-based similarity (lexical)
            # This is a placeholder - in production you'd use proper embeddings
            return self._simple_text_vector(text)
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def _simple_text_vector(self, text: str, dim: int = 384) -> List[float]:
        """Generate a simple hash-based vector for text (fallback)."""
        # Create a deterministic hash-based vector
        hash_obj = hashlib.md5(text.lower().encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        vector = []
        for i in range(0, min(len(hash_bytes), dim // 8)):
            byte_val = hash_bytes[i]
            # Normalize to [-1, 1] range
            vector.append((byte_val - 127.5) / 127.5)
        
        # Pad to desired dimension
        while len(vector) < dim:
            vector.append(0.0)
            
        return vector[:dim]
    
    def ingest_team_memory(self, team_memory: TeamMemory, project_id: str) -> IngestionResult:
        """Convert TeamMemory items into KG nodes and edges."""
        start_time = datetime.now()
        nodes_created = 0
        edges_created = 0
        errors = []
        
        try:
            memories = team_memory.get_memories(project_id)
            
            for memory in memories:
                try:
                    # Create memory node
                    memory_text = f"{memory.get('content', '')} {memory.get('context', '')}"
                    memory_id = f"memory_{memory.get('timestamp', 'unknown')}_{hash(memory_text) % 10000}"
                    
                    embedding = self._generate_embedding(memory_text)
                    
                    memory_node = KGNode(
                        id=memory_id,
                        label="memory",
                        props={
                            "content": memory.get('content', ''),
                            "context": memory.get('context', ''),
                            "author": memory.get('author', 'unknown'),
                            "timestamp": memory.get('timestamp'),
                            "project_id": project_id,
                            "type": "team_memory"
                        },
                        vector=embedding
                    )
                    
                    self.kg_store.upsert_node(memory_node)
                    nodes_created += 1
                    
                    # Create project relationship edge
                    project_edge = KGEdge(
                        source_id=memory_id,
                        target_id=f"project_{project_id}",
                        relationship="belongs_to",
                        props={"created_at": memory.get('timestamp')}
                    )
                    
                    self.kg_store.upsert_edge_obj(project_edge)
                    edges_created += 1
                    
                except Exception as e:
                    errors.append(f"Failed to ingest memory: {str(e)}")
                    
        except Exception as e:
            errors.append(f"Failed to access team memory: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            nodes_created=nodes_created,
            edges_created=edges_created,
            errors=errors,
            processing_time_ms=processing_time
        )
    
    def ingest_firecrawl_results(self, firecrawl_data: Dict[str, Any], query_context: str = "") -> IngestionResult:
        """Convert Firecrawl search/scrape results into KG nodes and edges."""
        start_time = datetime.now()
        nodes_created = 0
        edges_created = 0
        errors = []
        
        try:
            # Handle different Firecrawl result formats
            if 'results' in firecrawl_data:
                # Search results format
                results = firecrawl_data['results']
            elif 'content' in firecrawl_data:
                # Single scrape result format
                results = [firecrawl_data]
            else:
                results = [firecrawl_data]
            
            for result in results:
                try:
                    url = result.get('url', 'unknown')
                    title = result.get('title', 'Untitled')
                    content = result.get('content', result.get('markdown', ''))
                    
                    # Create document node
                    doc_id = f"doc_{hashlib.md5(url.encode()).hexdigest()[:12]}"
                    doc_text = f"{title} {content}"
                    
                    embedding = self._generate_embedding(doc_text)
                    
                    doc_node = KGNode(
                        id=doc_id,
                        label="document",
                        props={
                            "url": url,
                            "title": title,
                            "content": content[:2000],  # Truncate for storage
                            "source": "firecrawl",
                            "query_context": query_context,
                            "ingested_at": datetime.now().isoformat(),
                            "type": "web_document"
                        },
                        vector=embedding
                    )
                    
                    self.kg_store.upsert_node(doc_node)
                    nodes_created += 1
                    
                    # Create query context edge if provided
                    if query_context:
                        query_id = f"query_{hashlib.md5(query_context.encode()).hexdigest()[:12]}"
                        
                        # Create query node if it doesn't exist
                        query_node = KGNode(
                            id=query_id,
                            label="query",
                            props={
                                "text": query_context,
                                "type": "search_query",
                                "created_at": datetime.now().isoformat()
                            },
                            vector=self._generate_embedding(query_context)
                        )
                        
                        self.kg_store.upsert_node(query_node)
                        
                        # Link document to query
                        query_edge = KGEdge(
                            source_id=doc_id,
                            target_id=query_id,
                            relationship="found_by",
                            props={"relevance_score": result.get('score', 1.0)}
                        )
                        
                        self.kg_store.upsert_edge_obj(query_edge)
                        edges_created += 1
                    
                except Exception as e:
                    errors.append(f"Failed to ingest Firecrawl result: {str(e)}")
                    
        except Exception as e:
            errors.append(f"Failed to process Firecrawl data: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            nodes_created=nodes_created,
            edges_created=edges_created,
            errors=errors,
            processing_time_ms=processing_time
        )
    
    def ingest_agent_interaction(self, agent_role: str, task: str, result: str, project_id: str) -> IngestionResult:
        """Ingest agent interaction results into KG."""
        start_time = datetime.now()
        
        try:
            # Create interaction node
            interaction_id = f"interaction_{datetime.now().timestamp()}_{hash(task) % 10000}"
            interaction_text = f"{agent_role}: {task} -> {result}"
            
            embedding = self._generate_embedding(interaction_text)
            
            interaction_node = KGNode(
                id=interaction_id,
                label="interaction",
                props={
                    "agent_role": agent_role,
                    "task": task,
                    "result": result[:1000],  # Truncate
                    "project_id": project_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "agent_interaction"
                },
                vector=embedding
            )
            
            self.kg_store.upsert_node(interaction_node)
            
            # Link to project
            project_edge = KGEdge(
                source_id=interaction_id,
                target_id=f"project_{project_id}",
                relationship="part_of",
                props={"agent_role": agent_role}
            )
            
            self.kg_store.upsert_edge_obj(project_edge)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                nodes_created=1,
                edges_created=1,
                errors=[],
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return IngestionResult(
                nodes_created=0,
                edges_created=0,
                errors=[f"Failed to ingest agent interaction: {str(e)}"],
                processing_time_ms=processing_time
            )
