"""
Selection router that classifies queries to choose between vector-only, KG-only, or hybrid approaches.
Includes metrics tracking and feedback loop for continuous improvement.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    KG_ONLY = "kg_only" 
    HYBRID = "hybrid"
    MEMORY_ONLY = "memory_only"

@dataclass
class QueryClassification:
    """Result of query classification."""
    strategy: RetrievalStrategy
    confidence: float
    reasoning: str
    features: Dict[str, Any]

@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""
    query: str
    strategy: RetrievalStrategy
    results_count: int
    response_time_ms: float
    user_feedback: Optional[float] = None  # 0-1 relevance score
    precision_at_k: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class SelectionRouter:
    """Routes queries to optimal retrieval strategy based on query characteristics."""
    
    def __init__(self, server=None):
        self.server = server
        self.metrics_history: List[RetrievalMetrics] = []
        self.strategy_performance: Dict[RetrievalStrategy, Dict[str, float]] = {}
        self._load_metrics_history()
        self._initialize_strategy_performance()
        
    def _load_metrics_history(self):
        """Load historical metrics from storage."""
        try:
            if self.server and hasattr(self.server, 'storage'):
                metrics_data = self.server.storage.get("selection_router_metrics", {})
                if metrics_data:
                    self.metrics_history = [
                        RetrievalMetrics(**m) for m in metrics_data.get('history', [])
                    ]
                    self.strategy_performance = metrics_data.get('performance', {})
                    # Convert string keys back to enum
                    self.strategy_performance = {
                        RetrievalStrategy(k): v for k, v in self.strategy_performance.items()
                    }
        except Exception as e:
            logger.warning(f"Could not load metrics history: {e}")
            
    def _save_metrics_history(self):
        """Save metrics history to storage."""
        try:
            if self.server and hasattr(self.server, 'storage'):
                # Convert enum keys to strings for JSON serialization
                performance_data = {
                    k.value: v for k, v in self.strategy_performance.items()
                }
                
                metrics_data = {
                    'history': [asdict(m) for m in self.metrics_history[-1000:]],  # Keep last 1000
                    'performance': performance_data,
                    'last_updated': datetime.now().isoformat()
                }
                
                self.server.storage.set("selection_router_metrics", metrics_data)
        except Exception as e:
            logger.warning(f"Could not save metrics history: {e}")
    
    def _initialize_strategy_performance(self):
        """Initialize strategy performance tracking."""
        if not self.strategy_performance:
            for strategy in RetrievalStrategy:
                self.strategy_performance[strategy] = {
                    'avg_response_time': 0.0,
                    'avg_precision': 0.0,
                    'usage_count': 0,
                    'success_rate': 0.0
                }
    
    def classify_query(self, query: str, context: Dict[str, Any] = None) -> QueryClassification:
        """Classify query to determine optimal retrieval strategy."""
        context = context or {}
        
        # Extract query features
        features = self._extract_query_features(query, context)
        
        # Apply classification rules
        strategy, confidence, reasoning = self._apply_classification_rules(features)
        
        # Adjust based on historical performance
        strategy, confidence = self._adjust_for_performance(strategy, confidence, features)
        
        return QueryClassification(
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            features=features
        )
    
    def _extract_query_features(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from query for classification."""
        query_lower = query.lower()
        
        features = {
            'length': len(query.split()),
            'has_entities': any(word.isupper() for word in query.split()),
            'has_numbers': any(char.isdigit() for char in query),
            'has_dates': any(word in query_lower for word in ['today', 'yesterday', 'last', 'recent', 'ago']),
            'is_factual': any(word in query_lower for word in ['what', 'who', 'when', 'where', 'how']),
            'is_analytical': any(word in query_lower for word in ['analyze', 'compare', 'evaluate', 'assess']),
            'is_relational': any(word in query_lower for word in ['related', 'connected', 'similar', 'like']),
            'is_temporal': any(word in query_lower for word in ['before', 'after', 'during', 'timeline']),
            'domain_specific': self._detect_domain(query_lower),
            'complexity_score': self._calculate_complexity(query),
            'context_available': bool(context.get('project_id') or context.get('agent_role'))
        }
        
        return features
    
    def _detect_domain(self, query_lower: str) -> str:
        """Detect domain/topic of query."""
        domains = {
            'scientific': ['research', 'study', 'experiment', 'hypothesis', 'data', 'analysis'],
            'financial': ['investment', 'market', 'trading', 'portfolio', 'risk', 'return'],
            'technical': ['code', 'software', 'system', 'architecture', 'implementation'],
            'business': ['strategy', 'operations', 'management', 'process', 'workflow']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        factors = [
            len(query.split()) / 20,  # Length factor
            len([w for w in query.split() if len(w) > 6]) / len(query.split()),  # Complex words
            query.count('?') + query.count(',') + query.count(';'),  # Punctuation complexity
            len(set(query.lower().split())) / len(query.split())  # Vocabulary diversity
        ]
        
        return min(sum(factors) / len(factors), 1.0)
    
    def _apply_classification_rules(self, features: Dict[str, Any]) -> Tuple[RetrievalStrategy, float, str]:
        """Apply rule-based classification logic."""
        
        # Rule 1: Simple factual queries -> Vector search
        if features['is_factual'] and features['length'] < 8 and not features['is_relational']:
            return RetrievalStrategy.VECTOR_ONLY, 0.8, "Simple factual query best served by vector search"
        
        # Rule 2: Relational/connected queries -> KG search
        if features['is_relational'] or features['is_temporal']:
            return RetrievalStrategy.KG_ONLY, 0.9, "Relational query requires graph traversal"
        
        # Rule 3: Complex analytical queries -> Hybrid
        if features['is_analytical'] and features['complexity_score'] > 0.6:
            return RetrievalStrategy.HYBRID, 0.85, "Complex analytical query benefits from hybrid approach"
        
        # Rule 4: Domain-specific with context -> Hybrid
        if features['domain_specific'] != 'general' and features['context_available']:
            return RetrievalStrategy.HYBRID, 0.75, "Domain-specific query with context uses hybrid retrieval"
        
        # Rule 5: Recent/temporal queries with context -> Memory + KG
        if features['has_dates'] and features['context_available']:
            return RetrievalStrategy.HYBRID, 0.7, "Temporal query with context uses hybrid approach"
        
        # Default: Vector search for general queries
        return RetrievalStrategy.VECTOR_ONLY, 0.6, "Default vector search for general query"
    
    def _adjust_for_performance(self, strategy: RetrievalStrategy, confidence: float, 
                              features: Dict[str, Any]) -> Tuple[RetrievalStrategy, float]:
        """Adjust strategy based on historical performance."""
        
        # If we have enough data, consider switching strategies
        if self.strategy_performance[strategy]['usage_count'] > 10:
            current_performance = self.strategy_performance[strategy]['avg_precision']
            
            # Find best performing strategy for similar queries
            best_strategy = strategy
            best_performance = current_performance
            
            for alt_strategy, perf_data in self.strategy_performance.items():
                if (perf_data['usage_count'] > 5 and 
                    perf_data['avg_precision'] > best_performance + 0.1):  # 10% improvement threshold
                    best_strategy = alt_strategy
                    best_performance = perf_data['avg_precision']
            
            # Switch if significantly better performance available
            if best_strategy != strategy and best_performance > current_performance + 0.15:
                return best_strategy, confidence * 0.9, "Switched based on performance history"
        
        return strategy, confidence
    
    def record_metrics(self, metrics: RetrievalMetrics):
        """Record retrieval metrics for performance tracking."""
        self.metrics_history.append(metrics)
        
        # Update strategy performance
        strategy_perf = self.strategy_performance[metrics.strategy]
        count = strategy_perf['usage_count']
        
        # Update running averages
        strategy_perf['avg_response_time'] = (
            (strategy_perf['avg_response_time'] * count + metrics.response_time_ms) / (count + 1)
        )
        
        if metrics.precision_at_k is not None:
            strategy_perf['avg_precision'] = (
                (strategy_perf['avg_precision'] * count + metrics.precision_at_k) / (count + 1)
            )
        
        strategy_perf['usage_count'] = count + 1
        
        # Calculate success rate (results > 0)
        success_count = sum(1 for m in self.metrics_history 
                          if m.strategy == metrics.strategy and m.results_count > 0)
        strategy_perf['success_rate'] = success_count / strategy_perf['usage_count']
        
        # Save periodically
        if len(self.metrics_history) % 10 == 0:
            self._save_metrics_history()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        summary = {
            'total_queries': len(self.metrics_history),
            'strategies': {}
        }
        
        for strategy, perf in self.strategy_performance.items():
            summary['strategies'][strategy.value] = {
                'usage_count': perf['usage_count'],
                'avg_response_time_ms': round(perf['avg_response_time'], 2),
                'avg_precision': round(perf['avg_precision'], 3),
                'success_rate': round(perf['success_rate'], 3)
            }
        
        # Recent performance (last 50 queries)
        recent_metrics = self.metrics_history[-50:]
        if recent_metrics:
            recent_avg_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
            recent_precision = [m.precision_at_k for m in recent_metrics if m.precision_at_k is not None]
            recent_avg_precision = sum(recent_precision) / len(recent_precision) if recent_precision else 0
            
            summary['recent_performance'] = {
                'avg_response_time_ms': round(recent_avg_time, 2),
                'avg_precision': round(recent_avg_precision, 3),
                'query_count': len(recent_metrics)
            }
        
        return summary
    
    def calculate_precision_at_k(self, results: List[Dict[str, Any]], k: int = 5) -> float:
        """Calculate precision@k for retrieval results."""
        if not results or k <= 0:
            return 0.0
        
        # Simple relevance scoring based on result properties
        relevant_count = 0
        for i, result in enumerate(results[:k]):
            # Score based on available metadata
            relevance_score = 0.0
            
            if result.get('score', 0) > 0.5:  # High similarity score
                relevance_score += 0.4
            
            if result.get('content') and len(result['content']) > 100:  # Substantial content
                relevance_score += 0.3
            
            if result.get('timestamp'):  # Recent content
                relevance_score += 0.2
            
            if result.get('type') in ['memory', 'document']:  # Structured content
                relevance_score += 0.1
            
            if relevance_score >= 0.5:  # Threshold for relevance
                relevant_count += 1
        
        return relevant_count / min(k, len(results))
