from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResearchTopic:
    topic: str
    score: float
    source: str  # 'trend' | 'gap' | 'manual'
    discovered_at: float = field(default_factory=lambda: time.time())


class ProactiveResearchOrchestrator:
    """Continuously proposes and executes research on high-priority topics.

    Responsibilities
    - Discover trends (Firecrawl search when available; otherwise heuristics)
    - Detect knowledge gaps from stored artifacts
    - Prioritize a small set of topics and execute deep research
    - Persist outcomes using enhanced_mcp_storage_v2 (artifacts/contexts) or fallback memory
    - Run on an interval in the background; allow manual trigger and status queries
    """

    def __init__(self, storage, server, interval_seconds: int | None = None) -> None:
        self.storage = storage
        self.server = server  # for calling route_chat and deep_research handler
        self.interval_seconds = interval_seconds or int(
            (int((__import__('os').getenv('PROACTIVE_RESEARCH_INTERVAL_SEC', '0') or '0')) or 6 * 3600)
        )
        if self.interval_seconds < 900:
            # guard against noisy loops; minimum 15 minutes
            self.interval_seconds = 900

        self._queue: List[ResearchTopic] = []
        self._recent_history: List[Dict[str, Any]] = []
        self._last_run_at: float | None = None
        self._running = False
        self._bg_thread: Optional[threading.Thread] = None

        # Cache function proxies for Firecrawl when available
        try:
            from functions import firecrawl_search_firecrawl_mcp as _fc_search  # type: ignore
        except Exception:
            _fc_search = None
        try:
            from functions import firecrawl_deep_research_firecrawl_mcp as _fc_deep  # type: ignore
        except Exception:
            _fc_deep = None
        self._fc_search = _fc_search
        self._fc_deep = _fc_deep

    # --- Public control API ---
    def start_background(self) -> None:
        if self._running:
            return
        self._running = True
        # Prefer asyncio task if a loop is running; else use a thread
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._loop())
            logger.info("ProactiveResearchOrchestrator started on existing asyncio loop")
        except RuntimeError:
            # No running loop -> spawn thread that owns its own loop
            def _runner():
                asyncio.run(self._loop())
            self._bg_thread = threading.Thread(target=_runner, name="proactive-research", daemon=True)
            self._bg_thread.start()
            logger.info("ProactiveResearchOrchestrator started in background thread")

    def stop(self) -> None:
        self._running = False

    def enqueue(self, topic: str, score: float = 0.6, source: str = "manual") -> None:
        self._queue.append(ResearchTopic(topic=topic.strip(), score=score, source=source))
        # Keep queue small and sorted
        self._queue = sorted(self._queue, key=lambda t: t.score, reverse=True)[:50]

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "last_run_at": self._last_run_at,
            "queue_len": len(self._queue),
            "queue_preview": [t.topic for t in self._queue[:5]],
            "recent": self._recent_history[-5:],
        }

    async def run_once(self, max_topics: int = 3) -> Dict[str, Any]:
        """Execute one full research cycle now."""
        gaps = await self._identify_knowledge_gaps()
        trends = await self._discover_trends()
        priorities = self._prioritize_topics(gaps, trends, max_topics=max_topics)
        executed: List[Tuple[str, str]] = []  # (topic, research_id or note)
        for topic in priorities:
            try:
                summary = self._execute_research(topic)
                # Extract research id from summary when present
                rid = self._extract_research_id(summary)
                executed.append((topic, rid or "n/a"))
                # Persist a compact artifact entry
                self._store_artifact(topic=topic, content=summary, tags=["proactive", topic])
            except Exception as e:
                executed.append((topic, f"error: {e}"))
        payload = {"priorities": priorities, "executed": executed}
        self._recent_history.append({"at": time.time(), **payload})
        self._recent_history = self._recent_history[-50:]
        self._last_run_at = time.time()
        return payload

    # --- Internal loop ---
    async def _loop(self) -> None:
        while self._running:
            try:
                await self.run_once(max_topics=3)
            except Exception as e:
                logger.error(f"Proactive research cycle error: {e}")
            await asyncio.sleep(self.interval_seconds)

    # --- Discovery & Analysis ---
    async def _discover_trends(self) -> List[Tuple[str, float]]:
        topics: List[Tuple[str, float]] = []
        seeds = [
            "LLM routing best practices",
            "MCP protocol updates",
            "Agent collaboration frameworks",
            "Deep research automation",
            "Context management hierarchies",
        ]
        if self._fc_search is None:
            # Heuristic fallback: return seed list with mild scores
            return [(s, 0.55) for s in seeds]
        try:
            # Query a couple of seeds and pull titles/snippets as candidate topics
            for q in seeds[:3]:
                res = self._fc_search({"query": q, "limit": 5, "lang": "en", "country": "us"})
                items = (res or {}).get("data") or (res or {}).get("results") or []
                for it in items:
                    title = (it.get("title") or it.get("url") or q).strip()
                    if title:
                        topics.append((title, 0.7))
        except Exception as e:
            logger.warning(f"Firecrawl search failed; using seed trends. {e}")
            topics = [(s, 0.6) for s in seeds]
        return topics[:10]

    async def _identify_knowledge_gaps(self) -> List[Tuple[str, float]]:
        """Find topics we haven't researched recently but are relevant to our domain."""
        try:
            rows = self.storage.retrieve_memory(category="research", limit=200)
        except Exception:
            rows = []
        seen: Dict[str, float] = {}
        for r in rows:
            v = r.get("value") or ""
            try:
                obj = json.loads(v)
                q = (obj.get("query") or "").strip()
                if q:
                    seen[q.lower()] = max(seen.get(q.lower(), 0.0), r.get("timestamp", 0.0))
            except Exception:
                # fallback: naive parse
                if "query" in v:
                    seen[v.lower()] = time.time()
        # Propose candidate gaps based on curated domains
        candidates = [
            "Predictive tool orchestration",
            "Continual learning for routing",
            "Autonomous scheduling for research",
            "Knowledge graph for agent systems",
            "Semantic memory retrieval optimizations",
        ]
        gaps: List[Tuple[str, float]] = []
        now = time.time()
        for c in candidates:
            if c.lower() not in seen:
                gaps.append((c, 0.75))
            else:
                # If older than ~2 weeks, consider stale
                age = now - seen[c.lower()]
                if age > 14 * 86400:
                    gaps.append((f"Refresh: {c}", 0.65))
        return gaps[:10]

    def _prioritize_topics(self, gaps: List[Tuple[str, float]], trends: List[Tuple[str, float]], max_topics: int = 3) -> List[str]:
        # Start with manual queue
        pool: Dict[str, float] = {t.topic: t.score for t in self._queue}
        # Add gaps and trends
        for name, score in gaps + trends:
            pool[name] = max(pool.get(name, 0.0), float(score))
        # Simple ranking by score
        ranked = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)
        choices = [name for name, _ in ranked[:max_topics]]
        # Consume from queue if present
        self._queue = [t for t in self._queue if t.topic not in choices]
        return choices

    # --- Execution & Persistence ---
    def _execute_research(self, topic: str) -> str:
        # Prefer using existing deep_research handler to leverage storage and synthesis
        try:
            from server import handle_deep_research  # local import to avoid cycle at module import
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"deep_research unavailable: {e}")
        return handle_deep_research({"query": topic, "time_limit": 180, "max_depth": 4}, self.server)

    def _store_artifact(self, *, topic: str, content: str, tags: List[str]) -> None:
        # Try V2 artifacts; fallback to simple memory
        try:
            session_id = "proactive_research"
            if hasattr(self.storage, "create_artifact"):
                self.storage.create_artifact(
                    session_id=session_id,
                    artifact_type="research_summary",
                    title=f"Proactive: {topic[:60]}",
                    content=content,
                    contributors=["ProactiveResearchOrchestrator"],
                    tags=tags,
                    parent_artifact_id=None,
                )
                return
        except Exception as e:
            logger.debug(f"Artifact create failed: {e}")
        # Fallback memory entry
        try:
            key = f"proactive_{int(time.time())}"
            self.storage.store_memory(key=key, value=json.dumps({"topic": topic, "summary": content})[:4000], category="research")
        except Exception:
            pass

    @staticmethod
    def _extract_research_id(summary: str) -> Optional[str]:
        # Summary format: "Deep research {id} completed..."
        import re
        m = re.search(r"Deep research ([0-9a-f]{12}) completed", summary or "")
        return m.group(1) if m else None

