#!/usr/bin/env python3
"""
Advanced Strands Multi-Agent System Demo

This example demonstrates the enhanced capabilities including:
- Intelligent query routing (vector-only, KG-only, hybrid)
- Knowledge graph ingestion from multiple sources
- Performance metrics and feedback loops
- Vector embeddings integration
- Business acumen integration patterns

Run with: python examples/advanced_strands_demo.py
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strands import TeamOrchestrator, OrchestratorConfig
from strands.ingestion import KGIngestionEngine
from strands.selection_router import SelectionRouter, RetrievalMetrics
from strands.rag import HybridRetriever


class MockEnhancedServer:
    """Mock server for demonstration purposes."""
    
    def __init__(self):
        self.storage = MockStorage()
        self.providers = {}
        
    def route_chat(self, messages, **kwargs):
        """Mock chat routing."""
        return {"content": "Mock response for demonstration"}
    
    def memory_retrieve_semantic(self, query, limit=5):
        """Mock semantic memory retrieval."""
        return {
            "memories": [
                {"content": f"Memory result for: {query}", "score": 0.8},
                {"content": f"Related memory: {query}", "score": 0.6}
            ]
        }
    
    def web_search(self, query, max_depth=1, time_limit=30):
        """Mock web search."""
        return {
            "finalAnalysis": f"Web analysis for query: {query}. This demonstrates integration with Firecrawl MCP for real-time web research."
        }


class MockStorage:
    """Mock storage for demonstration."""
    
    def __init__(self):
        self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value


def demonstrate_query_classification():
    """Demonstrate intelligent query classification."""
    print("\n=== Query Classification Demo ===")
    
    server = MockEnhancedServer()
    router = SelectionRouter(server)
    
    test_queries = [
        "What is the capital of France?",  # Should be vector-only
        "Show me documents related to AI research",  # Should be KG-only
        "Analyze the performance trends and compare with competitors over time",  # Should be hybrid
        "Find recent memories about project Alpha",  # Should be memory-focused
        "How do McKinsey consultants approach market analysis frameworks?"  # Business acumen
    ]
    
    for query in test_queries:
        classification = router.classify_query(query)
        print(f"Query: {query}")
        print(f"  Strategy: {classification.strategy.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Reasoning: {classification.reasoning}")
        print()


def demonstrate_kg_ingestion():
    """Demonstrate knowledge graph ingestion."""
    print("\n=== Knowledge Graph Ingestion Demo ===")
    
    server = MockEnhancedServer()
    engine = KGIngestionEngine(server)
    
    # Simulate Firecrawl results from business research
    firecrawl_data = {
        "results": [
            {
                "url": "https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/how-to-master-the-seven-step-problem-solving-process",
                "title": "McKinsey Problem Solving Process",
                "content": "The hypothesis-led approach emphasizes forming and testing hypotheses to arrive at impactful solutions efficiently. The seven steps include: 1) Define problem, 2) Structure problem, 3) Prioritize issues, 4) Develop analysis plan, 5) Conduct analyses, 6) Synthesize findings, 7) Develop recommendations."
            },
            {
                "url": "https://www.bcg.com/about/our-history/growth-share-matrix",
                "title": "BCG Growth-Share Matrix",
                "content": "The BCG matrix categorizes products into four quadrants: Stars (high growth, high market share), Cash Cows (low growth, high market share), Question Marks (high growth, low market share), and Dogs (low growth, low market share)."
            }
        ]
    }
    
    # Ingest the business research
    result = engine.ingest_firecrawl_results(firecrawl_data, "business consulting frameworks")
    print(f"Ingested Firecrawl results:")
    print(f"  Nodes created: {result.nodes_created}")
    print(f"  Edges created: {result.edges_created}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    
    # Simulate agent interaction
    agent_result = engine.ingest_agent_interaction(
        agent_role="business_analyst",
        task="Analyze market positioning using BCG matrix",
        result="Completed BCG matrix analysis. Identified 3 Stars, 2 Cash Cows, 4 Question Marks, and 1 Dog in the product portfolio.",
        project_id="market_analysis_2024"
    )
    
    print(f"\nIngested agent interaction:")
    print(f"  Nodes created: {agent_result.nodes_created}")
    print(f"  Edges created: {agent_result.edges_created}")


def demonstrate_hybrid_retrieval():
    """Demonstrate hybrid retrieval with routing."""
    print("\n=== Hybrid Retrieval Demo ===")
    
    server = MockEnhancedServer()
    retriever = HybridRetriever(server)
    
    # Test different types of queries
    queries = [
        ("What are the key principles of strategic consulting?", {"project_id": "consulting_research"}),
        ("Show me related documents about McKinsey frameworks", {"agent_role": "business_analyst"}),
        ("Analyze the competitive landscape using Porter's Five Forces", {"project_id": "market_analysis"})
    ]
    
    for query, context in queries:
        print(f"Query: {query}")
        result = retriever.retrieve(query, top_k=3, context=context)
        
        print(f"  Strategy used: {result.get('strategy_used')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Results found: {len(result.get('combined', []))}")
        print(f"  Sources: KG={len(result.get('kg', []))}, Memory={len(result.get('memory', []))}, Web={result.get('web_used', False)}")
        print()


def demonstrate_performance_tracking():
    """Demonstrate performance metrics and feedback loops."""
    print("\n=== Performance Tracking Demo ===")
    
    server = MockEnhancedServer()
    router = SelectionRouter(server)
    
    # Simulate some retrieval operations with metrics
    from strands.selection_router import RetrievalStrategy, RetrievalMetrics
    
    metrics_data = [
        RetrievalMetrics("business strategy analysis", RetrievalStrategy.HYBRID, 5, 150.0, 0.8),
        RetrievalMetrics("simple fact lookup", RetrievalStrategy.VECTOR_ONLY, 3, 50.0, 0.9),
        RetrievalMetrics("relationship mapping", RetrievalStrategy.KG_ONLY, 4, 200.0, 0.7),
        RetrievalMetrics("complex research query", RetrievalStrategy.HYBRID, 8, 300.0, 0.85),
    ]
    
    for metrics in metrics_data:
        router.record_metrics(metrics)
    
    # Get performance summary
    summary = router.get_performance_summary()
    print("Performance Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    
    for strategy, perf in summary['strategies'].items():
        print(f"  {strategy}:")
        print(f"    Usage count: {perf['usage_count']}")
        print(f"    Avg response time: {perf['avg_response_time_ms']}ms")
        print(f"    Avg precision: {perf['avg_precision']}")
        print(f"    Success rate: {perf['success_rate']}")


def demonstrate_business_acumen_integration():
    """Demonstrate business acumen integration patterns."""
    print("\n=== Business Acumen Integration Demo ===")
    
    server = MockEnhancedServer()
    config = OrchestratorConfig(project_id="business_strategy_2024")
    orchestrator = TeamOrchestrator(server, config)
    
    # Mock the routing to avoid actual LLM calls
    orchestrator._route = lambda prompt, **kwargs: f"Business analysis response for: {prompt[:100]}..."
    
    # Set environment variables for advanced features
    os.environ["USE_HYBRID_RETRIEVAL"] = "1"
    os.environ["ENABLE_KG_INGESTION"] = "1"
    
    # Demonstrate business strategy orchestration
    business_task = """
    Conduct a comprehensive market analysis for our SaaS platform expansion into the European market.
    Use McKinsey's 7-S framework and BCG growth-share matrix to evaluate our strategic position.
    Include competitive analysis using Porter's Five Forces and recommend go-to-market strategies.
    """
    
    context = """
    Current market: North America (established)
    Target market: Europe (new)
    Product: B2B SaaS platform for project management
    Team size: 50 employees
    Revenue: $10M ARR
    """
    
    print("Orchestrating business strategy analysis...")
    result = orchestrator.orchestrate(business_task, context)
    
    print(f"Task: {result['task'][:100]}...")
    print(f"Agents assembled: {len(result['agents'])}")
    print(f"Workflow stages completed: {len(result['results'])}")
    
    # Show agent roles
    agent_roles = [agent['role'] for agent in result['agents']]
    print(f"Agent roles: {', '.join(agent_roles)}")
    
    # Get performance metrics
    metrics = orchestrator.get_performance_metrics()
    if 'total_queries' in metrics:
        print(f"Retrieval performance: {metrics['total_queries']} queries processed")


def main():
    """Run all demonstrations."""
    print("🚀 Advanced Strands Multi-Agent System Demo")
    print("=" * 50)
    
    try:
        demonstrate_query_classification()
        demonstrate_kg_ingestion()
        demonstrate_hybrid_retrieval()
        demonstrate_performance_tracking()
        demonstrate_business_acumen_integration()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Intelligent query routing based on content analysis")
        print("- Knowledge graph ingestion from multiple sources")
        print("- Hybrid retrieval combining vector, KG, and memory search")
        print("- Performance metrics and continuous improvement")
        print("- Business consulting framework integration")
        print("- Multi-agent orchestration with context awareness")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
