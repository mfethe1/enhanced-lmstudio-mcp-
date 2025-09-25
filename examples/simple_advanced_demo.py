#!/usr/bin/env python3
"""
Simple Advanced Strands Demo - Core Features Only

Demonstrates the key advanced features without complex orchestration:
- Query classification and routing
- Knowledge graph ingestion
- Performance metrics tracking
- Hybrid retrieval strategies
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strands.selection_router import SelectionRouter, RetrievalStrategy, RetrievalMetrics
from strands.ingestion import KGIngestionEngine
from strands.rag import HybridRetriever


class SimpleStorage:
    def __init__(self):
        self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value


class SimpleServer:
    def __init__(self):
        self.storage = SimpleStorage()


def main():
    print("🚀 Advanced Strands Features Demo")
    print("=" * 40)
    
    server = SimpleServer()
    
    # 1. Query Classification Demo
    print("\n1. Query Classification & Routing")
    print("-" * 30)
    
    router = SelectionRouter(server)
    
    test_queries = [
        "What is machine learning?",
        "Show me documents related to AI research",
        "Analyze market trends and compare with competitors",
        "Find recent project memories"
    ]
    
    for query in test_queries:
        classification = router.classify_query(query)
        print(f"'{query[:40]}...'")
        print(f"  → Strategy: {classification.strategy.value}")
        print(f"  → Confidence: {classification.confidence:.2f}")
        print()
    
    # 2. Knowledge Graph Ingestion Demo
    print("2. Knowledge Graph Ingestion")
    print("-" * 30)
    
    engine = KGIngestionEngine(server)
    
    # Simulate business research data
    firecrawl_data = {
        "url": "https://example.com/consulting-frameworks",
        "title": "Strategic Consulting Frameworks",
        "content": "McKinsey's problem-solving approach uses hypothesis-driven analysis. BCG's growth-share matrix categorizes products into Stars, Cash Cows, Question Marks, and Dogs."
    }
    
    result = engine.ingest_firecrawl_results(firecrawl_data, "consulting frameworks")
    print(f"Ingested web research:")
    print(f"  → Nodes created: {result.nodes_created}")
    print(f"  → Edges created: {result.edges_created}")
    print(f"  → Processing time: {result.processing_time_ms:.1f}ms")
    
    # Simulate agent interaction
    agent_result = engine.ingest_agent_interaction(
        agent_role="business_analyst",
        task="Apply BCG matrix to product portfolio",
        result="Identified 2 Stars, 3 Cash Cows, 1 Question Mark, 0 Dogs",
        project_id="portfolio_analysis"
    )
    
    print(f"\nIngested agent interaction:")
    print(f"  → Nodes created: {agent_result.nodes_created}")
    print(f"  → Edges created: {agent_result.edges_created}")
    
    # 3. Performance Metrics Demo
    print("\n3. Performance Metrics & Learning")
    print("-" * 30)
    
    # Simulate retrieval operations
    metrics_examples = [
        RetrievalMetrics("business strategy", RetrievalStrategy.HYBRID, 5, 150.0, 0.85),
        RetrievalMetrics("simple lookup", RetrievalStrategy.VECTOR_ONLY, 2, 45.0, 0.95),
        RetrievalMetrics("relationship query", RetrievalStrategy.KG_ONLY, 4, 120.0, 0.75),
        RetrievalMetrics("complex analysis", RetrievalStrategy.HYBRID, 7, 280.0, 0.80),
    ]
    
    for metrics in metrics_examples:
        router.record_metrics(metrics)
    
    summary = router.get_performance_summary()
    print(f"Performance Summary ({summary['total_queries']} queries):")
    
    for strategy, perf in summary['strategies'].items():
        if perf['usage_count'] > 0:
            print(f"  {strategy}:")
            print(f"    → Usage: {perf['usage_count']} queries")
            print(f"    → Avg time: {perf['avg_response_time_ms']:.0f}ms")
            print(f"    → Avg precision: {perf['avg_precision']:.2f}")
    
    # 4. Hybrid Retrieval Demo
    print("\n4. Hybrid Retrieval Strategies")
    print("-" * 30)
    
    retriever = HybridRetriever(server)
    
    test_retrievals = [
        ("McKinsey problem solving framework", {"project_id": "consulting"}),
        ("competitive analysis methods", {"agent_role": "strategist"}),
        ("recent team discussions", {"project_id": "current_project"})
    ]
    
    for query, context in test_retrievals:
        result = retriever.retrieve(query, top_k=3, context=context)
        print(f"'{query}'")
        print(f"  → Strategy: {result.get('strategy_used', 'unknown')}")
        print(f"  → Confidence: {result.get('confidence', 0):.2f}")
        print(f"  → Sources: KG={len(result.get('kg', []))}, Memory={len(result.get('memory', []))}")
        print()
    
    # 5. Summary
    print("5. Key Capabilities Demonstrated")
    print("-" * 30)
    print("✅ Intelligent query routing based on content analysis")
    print("✅ Knowledge graph ingestion from web research")
    print("✅ Agent interaction capture and storage")
    print("✅ Performance metrics and continuous learning")
    print("✅ Hybrid retrieval combining multiple strategies")
    print("✅ Context-aware retrieval optimization")
    
    print(f"\n🎯 System processed {summary['total_queries']} queries with intelligent routing")
    print("🔄 Ready for production deployment with your LM Studio MCP server")


if __name__ == "__main__":
    main()
