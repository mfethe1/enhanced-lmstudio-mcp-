from __future__ import annotations

import json
import time
from typing import Dict, Any


def _task_key(task_id: str) -> str:
    return f"task_{task_id}"


def handle_get_task_status(arguments: Dict[str, Any], server) -> str:
    """Return task status and any partial/final results.
    - Input: { task_id: str }
    - Output: JSON string with fields: id, type, status, progress, eta_seconds, data
    """
    task_id = (arguments.get("task_id") or "").strip()
    if not task_id:
        raise Exception("'task_id' is required")
    rows = server.storage.retrieve_memory(key=_task_key(task_id)) or []
    if not rows:
        return json.dumps({"id": task_id, "status": "UNKNOWN"})
    try:
        payload = rows[0].get("value")
        if isinstance(payload, str):
            return payload
        # Some storages may return already-parsed objects
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return json.dumps({"id": task_id, "status": "ERROR", "error": "Malformed task payload"})

