"""
Knowledge Graph for Advanced MCP Cognitive Architecture

Persistent graph of code artifacts, their relationships, and learned patterns.
- Uses NetworkX when available; falls back to a minimal in-memory graph structure otherwise
- Embeddings via sentence-transformers when installed; otherwise lightweight hashing vector
- Graph persistence via server/storage with base64-encoded pickle for portability

Design goals:
- Safe to import/run without heavy deps
- Clear APIs for adding/querying artifacts and retrieving generation context
"""
from __future__ import annotations

import base64
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


# ------------------------------
# Minimal Graph Fallback (if no networkx)
# ------------------------------
class _MiniDiGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self._edges[(src, dst)] = dict(attrs)

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def predecessors(self, node_id: str) -> List[str]:
        return [a for (a, b) in self._edges.keys() if b == node_id]

    @property
    def edges(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        return self._edges

    @property
    def nodes_data(self) -> Dict[str, Dict[str, Any]]:
        return self._nodes


# ------------------------------
# Embedding Fallback
# ------------------------------
class _LiteEmbedder:
    """Lightweight hashing-based embedder to avoid heavy dependencies when unavailable.
    Produces fixed-size vector from token frequencies.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, text: str):
        vec = [0.0] * self.dim
        for tok in (text or "").lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        if np is not None:
            return np.array(vec, dtype=float)
        return vec


@dataclass
class _StoreKey:
    category: str = "knowledge_graph"
    key: str = "graph_pickle_b64"


class CodeKnowledgeGraph:
    """Persistent knowledge graph of code artifacts, their relationships, and patterns."""

    def __init__(self, storage: Any, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.storage = storage
        # Graph selection
        self.graph = nx.DiGraph() if nx is not None else _MiniDiGraph()
        # Embedder selection
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer(embedding_model)
            except Exception:
                self.embedder = _LiteEmbedder()
        else:
            self.embedder = _LiteEmbedder()
        self.pattern_cache: Dict[str, Any] = {}
        self._keys = _StoreKey()
        self.load_or_initialize()

    # ------------------------------
    # Persistence
    # ------------------------------
    def load_or_initialize(self) -> None:
        """Load existing graph from storage or create a new base graph."""
        try:
            # Try preferred category/key pair
            rows = self.storage.retrieve_memory(category=self._keys.category, limit=1000) or []
            blob_b64: Optional[str] = None
            # Primary key search
            for r in rows:
                if r.get("key") == self._keys.key:
                    blob_b64 = r.get("value")
                    break
            # Back-compat path: old category/key
            if blob_b64 is None:
                rows2 = self.storage.retrieve_memory(key="knowledge_graph", category="graph") or []
                if rows2:
                    blob_b64 = rows2[0].get("value")
            if blob_b64:
                try:
                    data = base64.b64decode(blob_b64)
                    obj = pickle.loads(data)
                    if nx is not None and isinstance(obj, nx.DiGraph):
                        self.graph = obj
                    elif isinstance(obj, dict) and isinstance(self.graph, _MiniDiGraph):
                        # Restore minimal form
                        self.graph._nodes = obj.get("nodes", {})
                        self.graph._edges = {tuple(k): v for k, v in obj.get("edges", {}).items()}
                        return
                    return
                except Exception:
                    pass
        except Exception:
            pass
        self._initialize_base_graph()
        self._save_graph()

    def _serialize_graph(self) -> bytes:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            return pickle.dumps(self.graph)
        # Minimal fallback serialization
        assert isinstance(self.graph, _MiniDiGraph)
        payload = {"nodes": self.graph.nodes_data, "edges": {f"{a}|{b}": v for (a, b), v in self.graph.edges.items()}}
        return pickle.dumps(payload)

    def _save_graph(self) -> None:
        try:
            data = self._serialize_graph()
            blob_b64 = base64.b64encode(data).decode("utf-8")
            self.storage.store_memory(key=self._keys.key, value=blob_b64, category=self._keys.category)
        except Exception:
            pass

    def _initialize_base_graph(self) -> None:
        categories = [
            ("root", {"type": "root", "description": "Graph root"}),
            ("patterns", {"type": "category", "description": "Code patterns"}),
            ("functions", {"type": "category", "description": "Functions"}),
            ("classes", {"type": "category", "description": "Classes"}),
            ("modules", {"type": "category", "description": "Modules"}),
            ("tests", {"type": "category", "description": "Tests"}),
            ("bugs", {"type": "category", "description": "Bug patterns"}),
            ("optimizations", {"type": "category", "description": "Optimizations"}),
        ]
        for node_id, attrs in categories:
            self._add_node(node_id, **attrs)
            if node_id != "root":
                self._add_edge("root", node_id, relationship="contains")

    # ------------------------------
    # Graph primitives (abstract over nx/minigraph)
    # ------------------------------
    def _add_node(self, node_id: str, **attrs: Any) -> None:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            self.graph.add_node(node_id, **attrs)
        else:
            self.graph.add_node(node_id, **attrs)  # type: ignore[attr-defined]

    def _add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            self.graph.add_edge(src, dst, **attrs)
        else:
            self.graph.add_edge(src, dst, **attrs)  # type: ignore[attr-defined]

    def _get_node(self, node_id: str) -> Dict[str, Any]:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            return self.graph.nodes[node_id]
        return self.graph.nodes_data.get(node_id, {})  # type: ignore[attr-defined]

    def _has_node(self, node_id: str) -> bool:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            return node_id in self.graph
        return node_id in self.graph.nodes_data  # type: ignore[attr-defined]

    def _predecessors(self, node_id: str) -> List[str]:
        if nx is not None and isinstance(self.graph, nx.DiGraph):
            return list(self.graph.predecessors(node_id))
        return list(self.graph.predecessors(node_id))  # type: ignore[attr-defined]

    # ------------------------------
    # Public API
    # ------------------------------
    def add_code_artifact(self, code: str, artifact_type: str, metadata: Dict[str, Any]) -> str:
        """Add a code artifact to the graph with embeddings and relationships."""
        artifact_id = hashlib.md5((code or "").encode()).hexdigest()[:16]
        emb = self.embedder.encode(code or "")
        # Normalize embedding to list for storage
        if np is not None and not isinstance(emb, list):
            emb_list = emb.tolist()
        else:
            emb_list = list(emb)
        # Similar nodes
        similar_nodes = self._find_similar_nodes(emb, threshold=0.8)
        # Persist node
        self._add_node(
            artifact_id,
            type=artifact_type,
            code=code,
            embedding=emb_list,
            metadata=metadata,
            created_at=time.time(),
        )
        # Category edge
        category_map = {"function": "functions", "class": "classes", "module": "modules", "test": "tests"}
        if artifact_type in category_map:
            self._add_edge(category_map[artifact_type], artifact_id, relationship="contains")
        # Similarity edges
        for sim_id, sim_score in similar_nodes:
            self._add_edge(artifact_id, sim_id, relationship="similar_to", weight=float(sim_score))
        # Pattern detection
        for pattern in self._detect_patterns(code or "", artifact_type):
            pattern_id = f"pattern_{pattern['name']}"
            if not self._has_node(pattern_id):
                self._add_node(pattern_id, type="pattern", name=pattern["name"], description=pattern["description"])
                self._add_edge("patterns", pattern_id, relationship="contains")
            self._add_edge(artifact_id, pattern_id, relationship="implements", confidence=float(pattern["confidence"]))
        self._save_graph()
        return artifact_id

    def _find_similar_nodes(self, embedding: Any, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find nodes with similar embeddings using cosine similarity."""
        results: List[Tuple[str, float]] = []
        # Iterate nodes, skip those without embeddings
        node_ids = self.graph.nodes() if nx is not None else self.graph.nodes()  # type: ignore
        for nid in node_ids:  # type: ignore
            data = self._get_node(nid)
            if "embedding" not in data:
                continue
            other = data["embedding"]
            try:
                sim = self._cosine_sim(embedding, other)
            except Exception:
                continue
            if sim > threshold:
                results.append((nid, float(sim)))
        return sorted(results, key=lambda x: -x[1])[:5]

    def _cosine_sim(self, a: Any, b: Any) -> float:
        if np is not None:
            va = a if isinstance(a, np.ndarray) else np.array(a, dtype=float)
            vb = b if isinstance(b, np.ndarray) else np.array(b, dtype=float)
            denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
            return float(np.dot(va, vb) / denom)
        # Pure python fallback
        va = list(a)
        vb = list(b)
        n = min(len(va), len(vb))
        dot = sum(float(va[i]) * float(vb[i]) for i in range(n))
        norm_a = sum(float(x) * float(x) for x in va) ** 0.5 or 1.0
        norm_b = sum(float(x) * float(x) for x in vb) ** 0.5 or 1.0
        return float(dot / (norm_a * norm_b))

    def _detect_patterns(self, code: str, artifact_type: str) -> List[Dict[str, Any]]:
        rules = {
            "singleton": ["instance", "getinstance", "_instance", "__new__"],
            "factory": ["create", "factory", "build", "make"],
            "observer": ["subscribe", "notify", "observer", "listener"],
            "decorator": ["@", "wrapper", "decorator"],
            "async": ["async", "await", "asyncio"],
            "error_handling": ["try", "except", "catch", "finally"],
            "validation": ["validate", "check", "verify", "assert"],
        }
        out: List[Dict[str, Any]] = []
        s = (code or "").lower()
        for name, kws in rules.items():
            matches = sum(1 for kw in kws if kw in s)
            if matches:
                conf = min(matches / max(1, len(kws)), 1.0)
                out.append({"name": name, "description": f"Detected {name} pattern", "confidence": conf})
        return out

    # ------------------------------
    # Queries
    # ------------------------------
    def query_graph(self, query: str, query_type: str = "similarity", limit: int = 10) -> List[Dict[str, Any]]:
        if query_type == "similarity":
            q_emb = self.embedder.encode(query or "")
            sims = self._find_similar_nodes(q_emb, threshold=0.6)
            results: List[Dict[str, Any]] = []
            for nid, score in sims[:limit]:
                ndata = self._get_node(nid)
                results.append({
                    "id": nid,
                    "type": ndata.get("type"),
                    "code": ndata.get("code"),
                    "similarity": float(score),
                    "metadata": ndata.get("metadata"),
                })
            return results
        if query_type == "pattern":
            pattern_id = f"pattern_{query}"
            if not self._has_node(pattern_id):
                return []
            out: List[Dict[str, Any]] = []
            for pred in self._predecessors(pattern_id):
                edge = self.graph.edges.get((pred, pattern_id), {}) if nx is None else self.graph.edges[pred, pattern_id]
                if edge.get("relationship") == "implements":
                    nd = self._get_node(pred)
                    out.append({
                        "id": pred,
                        "code": nd.get("code"),
                        "confidence": float(edge.get("confidence", 0.0)),
                        "metadata": nd.get("metadata"),
                    })
            return sorted(out, key=lambda x: -x.get("confidence", 0.0))[:limit]
        if query_type == "evolution":
            chain: List[Dict[str, Any]] = []
            cur = query
            while cur:
                if not self._has_node(cur):
                    break
                nd = self._get_node(cur)
                chain.append({"id": cur, "code": nd.get("code"), "timestamp": nd.get("created_at"), "metadata": nd.get("metadata")})
                preds = [p for p in self._predecessors(cur) if (self.graph.edges.get((p, cur), {}) if nx is None else self.graph.edges[p, cur]).get("relationship") == "evolved_to"]
                cur = preds[0] if preds else None
            return chain
        return []

    def get_context_for_generation(self, task_description: str, max_context_items: int = 5) -> List[Dict[str, Any]]:
        ctx: List[Dict[str, Any]] = []
        similar = self.query_graph(task_description, "similarity", limit=max_context_items)
        for item in similar:
            item["relevance_type"] = "similar_code"
            ctx.append(item)
        # Extract patterns and add their examples
        for pat in self._detect_patterns(task_description or "", "task"):
            examples = self.query_graph(pat["name"], "pattern", limit=2)
            for ex in examples:
                ex["relevance_type"] = "pattern_example"
                ex["pattern"] = pat["name"]
                ctx.append(ex)
        ctx.sort(key=lambda x: x.get("similarity", 0.0) + x.get("confidence", 0.0), reverse=True)
        return ctx[:max_context_items]

