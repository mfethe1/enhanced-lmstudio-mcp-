"""
TeamMemory: persistent cross-session memory for research projects.
Backed by server.storage when available; falls back to JSON file storage.
"""
from __future__ import annotations
import json, os, time
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    ts: float
    author: str
    kind: str
    content: Dict[str, Any]


class TeamMemory:
    def __init__(self, server, project_id: str):
        self.server = server
        self.project_id = project_id
        self.key = f"team_memory:{project_id}"
        self._fallback_path = os.path.join(os.getcwd(), f".team_memory_{project_id}.json")

    def _read_store(self) -> List[Dict[str, Any]]:
        try:
            data = self.server.storage.get(self.key)  # type: ignore[attr-defined]
            if data:
                return json.loads(data)
        except Exception:
            pass
        if os.path.exists(self._fallback_path):
            with open(self._fallback_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _write_store(self, items: List[Dict[str, Any]]):
        try:
            self.server.storage.set(self.key, json.dumps(items))  # type: ignore[attr-defined]
            return
        except Exception:
            pass
        with open(self._fallback_path, "w", encoding="utf-8") as f:
            json.dump(items, f)

    def add(self, author: str, kind: str, content: Dict[str, Any]):
        items = self._read_store()
        items.append(MemoryItem(ts=time.time(), author=author, kind=kind, content=content).__dict__)
        # keep bounded
        items = items[-500:]
        self._write_store(items)

    def query(self, kind: str | None = None) -> List[Dict[str, Any]]:
        items = self._read_store()
        if kind:
            return [i for i in items if i.get("kind") == kind]
        return items

