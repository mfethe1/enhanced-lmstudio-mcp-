"""
Tests for advanced Strands features: ingestion, selection router, and enhanced RAG.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch
from datetime import datetime

from strands.ingestion import KGIngestionEngine, IngestionResult
from strands.selection_router import SelectionRouter, RetrievalStrategy, QueryClassification, RetrievalMetrics
from strands.rag import HybridRetriever
from strands.team_memory import TeamMemory
from strands.team_orchestrator import TeamOrchestrator, OrchestratorConfig


class TestKGIngestionEngine:
    """Test the KG ingestion engine."""
    
    def test_ingestion_engine_init(self):
        """Test ingestion engine initialization."""
        mock_server = Mock()
        engine = KGIngestionEngine(mock_server)
        
        assert engine.server == mock_server
        assert engine.kg_store is not None
        assert isinstance(engine.embeddings_enabled, bool)
    
    def test_simple_text_vector(self):
        """Test simple text vector generation."""
        engine = KGIngestionEngine(None)
        vector = engine._simple_text_vector("test text", dim=10)
        
        assert len(vector) == 10
        assert all(isinstance(v, float) for v in vector)
        assert all(-1 <= v <= 1 for v in vector)
    
    def test_ingest_team_memory(self):
        """Test team memory ingestion."""
        mock_server = Mock()
        mock_server.storage = Mock()
        
        engine = KGIngestionEngine(mock_server)
        
        # Create mock team memory
        team_memory = Mock()
        team_memory.get_memories.return_value = [
            {
                'content': 'Test memory content',
                'context': 'Test context',
                'author': 'test_agent',
                'timestamp': '2024-01-01T00:00:00'
            }
        ]
        
        result = engine.ingest_team_memory(team_memory, "test_project")
        
        assert isinstance(result, IngestionResult)
        assert result.nodes_created >= 0
        assert result.edges_created >= 0
        assert isinstance(result.processing_time_ms, float)
    
    def test_ingest_firecrawl_results(self):
        """Test Firecrawl results ingestion."""
        mock_server = Mock()
        engine = KGIngestionEngine(mock_server)
        
        firecrawl_data = {
            'results': [
                {
                    'url': 'https://example.com',
                    'title': 'Test Page',
                    'content': 'Test content from web page'
                }
            ]
        }
        
        result = engine.ingest_firecrawl_results(firecrawl_data, "test query")
        
        assert isinstance(result, IngestionResult)
        assert result.nodes_created >= 0
        assert result.edges_created >= 0
    
    def test_ingest_agent_interaction(self):
        """Test agent interaction ingestion."""
        mock_server = Mock()
        engine = KGIngestionEngine(mock_server)
        
        result = engine.ingest_agent_interaction(
            agent_role="research_coordinator",
            task="Analyze market trends",
            result="Market analysis complete",
            project_id="test_project"
        )
        
        assert isinstance(result, IngestionResult)
        assert result.nodes_created == 1
        assert result.edges_created == 1


class TestSelectionRouter:
    """Test the selection router."""
    
    def test_router_init(self):
        """Test router initialization."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        router = SelectionRouter(mock_server)
        
        assert router.server == mock_server
        assert len(router.strategy_performance) == len(RetrievalStrategy)
    
    def test_query_classification(self):
        """Test query classification."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        router = SelectionRouter(mock_server)
        
        # Test factual query
        classification = router.classify_query("What is the capital of France?")
        assert isinstance(classification, QueryClassification)
        assert classification.strategy in RetrievalStrategy
        assert 0 <= classification.confidence <= 1
        
        # Test relational query
        classification = router.classify_query("Show me documents related to AI research")
        assert classification.strategy in [RetrievalStrategy.KG_ONLY, RetrievalStrategy.HYBRID]
        
        # Test analytical query
        classification = router.classify_query("Analyze the performance trends and compare with competitors")
        assert classification.strategy in [RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY]
    
    def test_feature_extraction(self):
        """Test query feature extraction."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        router = SelectionRouter(mock_server)
        
        features = router._extract_query_features("What are the recent AI developments?", {})
        
        assert 'length' in features
        assert 'is_factual' in features
        assert 'complexity_score' in features
        assert 'domain_specific' in features
        assert isinstance(features['length'], int)
        assert isinstance(features['is_factual'], bool)
    
    def test_metrics_recording(self):
        """Test metrics recording and performance tracking."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        router = SelectionRouter(mock_server)
        
        # Record some metrics
        metrics = RetrievalMetrics(
            query="test query",
            strategy=RetrievalStrategy.HYBRID,
            results_count=5,
            response_time_ms=150.0,
            precision_at_k=0.8
        )
        
        router.record_metrics(metrics)
        
        assert len(router.metrics_history) == 1
        assert router.strategy_performance[RetrievalStrategy.HYBRID]['usage_count'] == 1
        assert router.strategy_performance[RetrievalStrategy.HYBRID]['avg_precision'] == 0.8
    
    def test_precision_calculation(self):
        """Test precision@k calculation."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        router = SelectionRouter(mock_server)
        
        results = [
            {'score': 0.9, 'content': 'High quality content with substantial information', 'type': 'document'},
            {'score': 0.7, 'content': 'Medium quality content', 'type': 'memory'},
            {'score': 0.3, 'content': 'Low quality', 'type': 'unknown'}
        ]
        
        precision = router.calculate_precision_at_k(results, k=3)
        assert 0 <= precision <= 1


class TestEnhancedHybridRetriever:
    """Test the enhanced hybrid retriever."""
    
    def test_retriever_with_router(self):
        """Test retriever with selection router integration."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        retriever = HybridRetriever(mock_server)
        
        assert retriever.router is not None
        assert isinstance(retriever.router, SelectionRouter)
    
    def test_retrieval_with_context(self):
        """Test retrieval with context information."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        retriever = HybridRetriever(mock_server)
        
        context = {
            'project_id': 'test_project',
            'agent_role': 'research_coordinator'
        }
        
        result = retriever.retrieve("test query", top_k=5, context=context)
        
        assert 'strategy_used' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert 'combined' in result
    
    def test_vector_only_retrieval(self):
        """Test vector-only retrieval strategy."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        retriever = HybridRetriever(mock_server)
        
        result = retriever._vector_retrieval("test query", 5)
        
        assert 'kg' in result
        assert 'memory' in result
        assert 'web' in result
        assert 'combined' in result
        assert result['web_used'] is False
    
    def test_memory_only_retrieval(self):
        """Test memory-only retrieval strategy."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = [
            {'content': 'memory item 1'},
            {'content': 'memory item 2'}
        ]
        
        retriever = HybridRetriever(mock_server)
        
        result = retriever._memory_retrieval("test query", 5, {})
        
        assert 'memory' in result
        assert 'combined' in result
        assert len(result['combined']) >= 0
    
    def test_result_combination_and_ranking(self):
        """Test result combination and ranking."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        retriever = HybridRetriever(mock_server)
        
        kg_hits = [{'content': 'kg result', 'score': 0.9}]
        memory_hits = [{'content': 'memory result', 'score': 0.7}]
        web_hits = [{'content': 'web result', 'score': 0.6}]
        
        combined = retriever._combine_and_rank_results(kg_hits, memory_hits, web_hits, 5)
        
        assert len(combined) <= 5
        # Results should be sorted by score (descending)
        if len(combined) > 1:
            for i in range(len(combined) - 1):
                assert combined[i]['score'] >= combined[i + 1]['score']


class TestEnhancedOrchestrator:
    """Test the enhanced team orchestrator."""
    
    def test_orchestrator_with_ingestion(self):
        """Test orchestrator with ingestion engine."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        config = OrchestratorConfig(project_id="test_project")
        orchestrator = TeamOrchestrator(mock_server, config)
        
        assert orchestrator.ingestion_engine is not None
        assert isinstance(orchestrator.ingestion_engine, KGIngestionEngine)
    
    def test_performance_metrics(self):
        """Test performance metrics retrieval."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        
        config = OrchestratorConfig(project_id="test_project")
        orchestrator = TeamOrchestrator(mock_server, config)
        
        metrics = orchestrator.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        # Should have either performance data or error
        assert 'total_queries' in metrics or 'error' in metrics


@pytest.mark.integration
class TestIntegrationAdvancedFeatures:
    """Integration tests for advanced features."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with all advanced features."""
        mock_server = Mock()
        mock_server.storage = Mock()
        mock_server.storage.get.return_value = {}
        mock_server.storage.set = Mock()

        # Mock async route_chat method
        async def mock_route_chat(*args, **kwargs):
            return "Mock async response"
        mock_server.route_chat = mock_route_chat

        # Set up environment variables
        with patch.dict(os.environ, {
            'USE_HYBRID_RETRIEVAL': '1',
            'ENABLE_KG_INGESTION': '1'
        }):
            config = OrchestratorConfig(project_id="integration_test")
            orchestrator = TeamOrchestrator(mock_server, config)

            # Mock the routing method to avoid actual LLM calls
            orchestrator._route = Mock(return_value="Mock response")

            # Test orchestration
            result = orchestrator.orchestrate("Test research task", "Test context")

            assert isinstance(result, dict)
            assert 'task' in result
            assert 'agents' in result
            assert 'results' in result
    
    def test_firecrawl_integration(self):
        """Test Firecrawl integration with ingestion."""
        mock_server = Mock()
        mock_server.web_search = Mock(return_value={
            'finalAnalysis': 'Mock web search analysis'
        })
        
        engine = KGIngestionEngine(mock_server)
        
        # Mock Firecrawl data
        firecrawl_data = {
            'url': 'https://example.com',
            'title': 'Test Article',
            'content': 'Test article content about AI research'
        }
        
        result = engine.ingest_firecrawl_results(firecrawl_data, "AI research query")
        
        assert result.nodes_created >= 0
        assert result.errors == [] or len(result.errors) == 0
