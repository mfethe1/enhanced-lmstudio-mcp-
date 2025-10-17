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

    This function checks multiple task tracking systems:
    1. Memory storage (for deep_research and legacy tasks)
    2. TaskManager (for agentic background tasks)
    """
    task_id = (arguments.get("task_id") or "").strip()
    if not task_id:
        raise Exception("'task_id' is required")

    # First, try memory storage (deep_research uses this)
    rows = server.storage.retrieve_memory(key=_task_key(task_id)) or []
    if rows:
        try:
            payload = rows[0].get("value")
            if isinstance(payload, str):
                return payload
            # Some storages may return already-parsed objects
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return json.dumps({"id": task_id, "status": "ERROR", "error": "Malformed task payload"})

    # Second, try TaskManager (agentic tasks use this)
    try:
        # Check if server has agentic_handlers
        if hasattr(server, 'agentic_handlers') and server.agentic_handlers:
            task = server.agentic_handlers.task_manager.get_task(task_id)
            if task:
                # Convert TaskManager task to the expected format
                return json.dumps({
                    "id": task_id,
                    "type": task.tool_name,
                    "status": task.status.value.upper(),
                    "progress": task.progress or "0%",
                    "eta_seconds": 0,  # TaskManager doesn't track ETA
                    "data": {
                        "result": task.result,
                        "error": task.error,
                        "created_at": task.created_at,
                        "started_at": task.started_at,
                        "completed_at": task.completed_at
                    }
                }, ensure_ascii=False)
            else:
                # Task not found in TaskManager, but log for debugging
                import logging
                logging.debug(f"Task {task_id} not found in TaskManager (has {len(server.agentic_handlers.task_manager.tasks)} tasks)")
    except Exception as e:
        # If TaskManager check fails, log and continue to return UNKNOWN
        import logging
        logging.error(f"TaskManager lookup failed for {task_id}: {e}", exc_info=True)

    # Not found in either system
    return json.dumps({"id": task_id, "status": "UNKNOWN", "error": f"Task '{task_id}' not found in memory storage or TaskManager"})

