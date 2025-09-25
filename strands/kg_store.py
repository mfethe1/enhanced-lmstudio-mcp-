"""
Knowledge Graph + Vector hybrid store interfaces with safe fallbacks.
Adapters: Memgraph, Neo4j (placeholders) with in-memory fallback so imports/tests pass
without external services. Designed to integrate with server.storage and router.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os


@dataclass
class KGNode:
    id: str
    label: str
    props: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relationship: str
    props: Dict[str, Any] = None

    def __post_init__(self):
        if self.props is None:
            self.props = {}


class BaseKGStore:
    def upsert_node(self, node: KGNode) -> None: ...
    def upsert_edge(self, src_id: str, rel: str, dst_id: str, props: Dict[str, Any] | None = None) -> None: ...
    def hybrid_query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]: ...


class InMemoryKGStore(BaseKGStore):
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[Dict[str, Any]] = []

    def upsert_node(self, node: KGNode) -> None:
        self.nodes[node.id] = node

    def upsert_edge(self, src_id: str, rel: str, dst_id: str, props: Dict[str, Any] | None = None) -> None:
        self.edges.append({"src": src_id, "rel": rel, "dst": dst_id, "props": props or {}})

    def upsert_edge_obj(self, edge: KGEdge) -> None:
        """Upsert edge using KGEdge object."""
        self.upsert_edge(edge.source_id, edge.relationship, edge.target_id, edge.props)

    def hybrid_query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Very naive lexical search over label/props; placeholder
        q = text.lower()
        scored = []
        for n in self.nodes.values():
            hay = f"{n.label} {n.props}".lower()
            score = hay.count(q) if q else 0
            scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": n.id, "label": n.label, "props": n.props, "score": s} for s, n in scored[:top_k]]


class MemgraphAdapter(BaseKGStore):
    def __init__(self, url: Optional[str] = None, user: str = "", password: str = ""):
        self.url = url or os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
        self.user = user or os.getenv("MEMGRAPH_USER", "")
        self.password = password or os.getenv("MEMGRAPH_PASSWORD", "")
        # Lazy connect in real implementation

    def upsert_node(self, node: KGNode) -> None:
        # TODO: implement via memgraph client; placeholder no-op
        return

    def upsert_edge(self, src_id: str, rel: str, dst_id: str, props: Dict[str, Any] | None = None) -> None:
        # TODO: implement via memgraph client; placeholder
        return

    def hybrid_query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # TODO: implement hybrid vector + lexical using Memgraph; placeholder
        return []


class Neo4jAdapter(BaseKGStore):
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")

    def upsert_node(self, node: KGNode) -> None: return
    def upsert_edge(self, src_id: str, rel: str, dst_id: str, props: Dict[str, Any] | None = None) -> None: return
    def hybrid_query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]: return []


def get_store() -> BaseKGStore:
    backend = (os.getenv("KG_BACKEND") or "").lower()
    if backend == "memgraph":
        return MemgraphAdapter()
    if backend == "neo4j":
        return Neo4jAdapter()
    return InMemoryKGStore()

