#!/usr/bin/env python3
import os
os.environ.setdefault("PROACTIVE_RESEARCH_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def test_hybrid_retriever_inmemory():
    import server
    from strands.rag import HybridRetriever
    from strands.kg_store import KGNode

    srv = server.EnhancedLMStudioMCPServer()
    retriever = HybridRetriever(srv)

    # Populate in-memory KG with a couple of nodes
    retriever.kg.upsert_node(KGNode(id="n1", label="Paper", props={"title": "Multi-Agent Planning"}))
    retriever.kg.upsert_node(KGNode(id="n2", label="Paper", props={"title": "GraphRAG with Neo4j"}))

    out = retriever.retrieve("multi-agent")
    assert isinstance(out, dict)
    assert "kg" in out
    # Some result should come back (naive lexical scoring)
    assert len(out["kg"]) >= 1

