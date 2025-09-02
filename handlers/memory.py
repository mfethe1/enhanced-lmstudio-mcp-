from __future__ import annotations

import json
import time
from typing import Any, Dict


def handle_memory_store(arguments: Dict[str, Any], server) -> str:
    key = arguments.get("key", "")
    value = arguments.get("value", "")
    category = arguments.get("category", "general")

    success = server.storage.store_memory(key, value, category)

    if success:
        return f"✅ Stored memory with key '{key}' in category '{category}'"
    else:
        return f"❌ Failed to store memory with key '{key}'"


def handle_memory_retrieve(arguments: Dict[str, Any], server) -> str:
    """Enhanced retrieval: supports key/category/search_term and formats results."""
    key = arguments.get("key")
    category = arguments.get("category")
    search_term = arguments.get("search_term")

    memories = server.storage.retrieve_memory(
        key=key,
        category=category,
        search_term=search_term,
        limit=50
    )

    if not memories:
        return "No matching memories found"

    results = []
    for memory in memories:
        ts = memory.get('timestamp')
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
        results.append(f"**{memory['key']}** ({memory.get('category','')}) - {timestamp}\n{memory.get('value','')}")

    # return as single string
    return "\n\n".join(results)


def handle_memory_consolidate(arguments: Dict[str, Any], server) -> str:
    category = (arguments.get("category") or "general").strip()
    limit = int(arguments.get("limit", 50))
    rows = server.storage.retrieve_memory(category=category, limit=limit) or []
    texts = []
    for r in rows:
        v = r.get("value")
        if isinstance(v, str):
            texts.append(v)
        else:
            try:
                texts.append(json.dumps(v, ensure_ascii=False))
            except Exception:
                continue
    blob = ("\n\n".join(texts))[:3500]
    return f"Consolidated {len(texts)} notes into a {len(blob)}-char memo (preview):\n\n{blob[:600]}..."


def handle_memory_retrieve_semantic(arguments: Dict[str, Any], server) -> str:
    query = (arguments.get("query") or "").lower()
    if not query:
        raise Exception("'query' is required")
    limit = int(arguments.get("limit", 5))
    rows = server.storage.retrieve_memory(category="semantic_memory", limit=200) or []
    items = []
    import re
    tokens_q = set(re.findall(r"[a-z0-9_]+", query))
    for r in rows:
        try:
            d = json.loads(r.get("value", "{}"))
            txt = (d.get("summary") or "").lower()
            tokens_t = set(re.findall(r"[a-z0-9_]+", txt))
            score = len(tokens_q & tokens_t)
            items.append({"key": r.get("key"), "summary": d.get("summary"), "score": score})
        except Exception:
            continue
    items = sorted(items, key=lambda x: x["score"], reverse=True)[:limit]
    if not items:
        return "No semantic matches"
    return "\n\n".join(f"- {it['key']}: {it['summary']} (score={it['score']})" for it in items)

