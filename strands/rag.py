"""
Hybrid RAG system combining Knowledge Graph queries with memory search and Firecrawl augmentation.
Provides intelligent retrieval routing and context augmentation for multi-agent workflows.
"""
from __future__ import annotations
import os
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional

from .kg_store import get_store
from .selection_router import SelectionRouter, RetrievalStrategy, RetrievalMetrics

logger = logging.getLogger(__name__)


def _cheap_embed(text: str, dim: int = 64) -> List[float]:
    """Very small, deterministic embedding substitute to avoid deps in tests."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat to fill dim
    arr = list(h) * ((dim + len(h) - 1) // len(h))
    return [x / 255.0 for x in arr[:dim]]


class HybridRetriever:
    """Combines KG queries, memory search, and optional Firecrawl augmentation with intelligent routing."""

    def __init__(self, server):
        self.server = server
        self.kg = get_store()
        self.router = SelectionRouter(server)

    def retrieve(self, query: str, top_k: int = 5, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant information using intelligent hybrid approach."""
        start_time = time.time()
        context = context or {}

        # Classify query to determine optimal strategy
        classification = self.router.classify_query(query, context)

        # Execute retrieval based on strategy
        if classification.strategy == RetrievalStrategy.VECTOR_ONLY:
            results = self._vector_retrieval(query, top_k)
        elif classification.strategy == RetrievalStrategy.KG_ONLY:
            results = self._kg_retrieval(query, top_k)
        elif classification.strategy == RetrievalStrategy.MEMORY_ONLY:
            results = self._memory_retrieval(query, top_k, context)
        else:  # HYBRID
            results = self._hybrid_retrieval(query, top_k, context)

        # Record metrics
        response_time = (time.time() - start_time) * 1000
        precision_at_k = self.router.calculate_precision_at_k(results.get('combined', []), top_k)

        metrics = RetrievalMetrics(
            query=query,
            strategy=classification.strategy,
            results_count=len(results.get('combined', [])),
            response_time_ms=response_time,
            precision_at_k=precision_at_k
        )

        self.router.record_metrics(metrics)

        # Add metadata
        results['strategy_used'] = classification.strategy.value
        results['confidence'] = classification.confidence
        results['reasoning'] = classification.reasoning

        return results

    def _vector_retrieval(self, query: str, top_k: int) -> Dict[str, Any]:
        """Vector-only retrieval using embeddings."""
        try:
            # Use KG's vector search capability
            kg_hits = self.kg.hybrid_query(query, top_k=top_k)

            return {
                "kg": kg_hits,
                "memory": [],
                "web": [],
                "combined": kg_hits,
                "web_used": False
            }
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return {"kg": [], "memory": [], "web": [], "combined": [], "web_used": False}

    def _kg_retrieval(self, query: str, top_k: int) -> Dict[str, Any]:
        """KG-only retrieval using graph traversal."""
        try:
            # Focus on relationship-based queries
            kg_hits = self.kg.hybrid_query(query, top_k=top_k)

            return {
                "kg": kg_hits,
                "memory": [],
                "web": [],
                "combined": kg_hits,
                "web_used": False
            }
        except Exception as e:
            logger.warning(f"KG retrieval failed: {e}")
            return {"kg": [], "memory": [], "web": [], "combined": [], "web_used": False}

    def _memory_retrieval(self, query: str, top_k: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Memory-only retrieval from team memory."""
        memory_hits = []

        try:
            if self.server and hasattr(self.server, 'memory_retrieve_semantic'):
                # Use server's semantic memory retrieval
                memory_result = self.server.memory_retrieve_semantic(query, limit=top_k)
                if memory_result and 'memories' in memory_result:
                    memory_hits = memory_result['memories']
            else:
                # Fallback to storage-based retrieval
                items_json = self.server.storage.get("team_memory:global")
                if items_json:
                    memory_hits = [i for i in (items_json or [])][:top_k]

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

        return {
            "kg": [],
            "memory": memory_hits,
            "web": [],
            "combined": memory_hits,
            "web_used": False
        }

    def _hybrid_retrieval(self, query: str, top_k: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Full hybrid retrieval combining all sources."""
        kg_hits = []
        memory_hits = []
        web_hits = []
        web_used = False

        # KG retrieval
        try:
            kg_hits = self.kg.hybrid_query(query, top_k=max(2, top_k // 2))
        except Exception as e:
            logger.warning(f"KG retrieval in hybrid failed: {e}")

        # Memory retrieval
        try:
            if self.server and hasattr(self.server, 'memory_retrieve_semantic'):
                memory_result = self.server.memory_retrieve_semantic(query, limit=max(2, top_k // 2))
                if memory_result and 'memories' in memory_result:
                    memory_hits = memory_result['memories']
            else:
                # Fallback to storage-based retrieval
                items_json = self.server.storage.get("team_memory:global")
                if items_json:
                    memory_hits = [i for i in (items_json or [])][:max(2, top_k // 2)]
        except Exception as e:
            logger.warning(f"Memory retrieval in hybrid failed: {e}")

        # Optional web augmentation via Firecrawl
        if os.getenv("ENABLE_WEB_AUGMENTATION", "0") == "1":
            try:
                web_hits = self._firecrawl_augmentation(query, max(1, top_k // 3))
                web_used = len(web_hits) > 0
            except Exception as e:
                logger.warning(f"Web augmentation failed: {e}")

        # Combine and rank results
        combined = self._combine_and_rank_results(kg_hits, memory_hits, web_hits, top_k)

        return {
            "kg": kg_hits,
            "memory": memory_hits,
            "web": web_hits,
            "combined": combined,
            "web_used": web_used
        }

    def _firecrawl_augmentation(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Augment results with Firecrawl web search."""
        if not self.server:
            return []

        try:
            # Use server's web search capability
            if hasattr(self.server, 'web_search'):
                search_result = self.server.web_search(query, max_depth=1, time_limit=30)
                if search_result and 'finalAnalysis' in search_result:
                    return [{
                        'content': search_result['finalAnalysis'],
                        'source': 'web_search',
                        'query': query,
                        'type': 'web_analysis'
                    }]

        except Exception as e:
            logger.debug(f"Firecrawl augmentation failed: {e}")

        return []

    def _combine_and_rank_results(self, kg_hits: List, memory_hits: List,
                                web_hits: List, top_k: int) -> List[Dict[str, Any]]:
        """Combine and rank results from different sources."""
        combined = []

        # Add KG results with source tagging
        for hit in kg_hits:
            if isinstance(hit, dict):
                hit['source'] = 'knowledge_graph'
                hit['score'] = hit.get('score', 0.8)
                combined.append(hit)

        # Add memory results
        for hit in memory_hits:
            if isinstance(hit, dict):
                hit['source'] = 'team_memory'
                hit['score'] = hit.get('score', 0.7)
                combined.append(hit)

        # Add web results
        for hit in web_hits:
            if isinstance(hit, dict):
                hit['source'] = 'web_search'
                hit['score'] = hit.get('score', 0.6)
                combined.append(hit)

        # Sort by score and return top_k
        combined.sort(key=lambda x: x.get('score', 0), reverse=True)
        return combined[:top_k]

