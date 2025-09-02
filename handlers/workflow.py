from __future__ import annotations

import asyncio
from typing import Any, Dict

# Use workflow subsystem through server helpers
from server import (
    _get_tool_schemas,
)


def handle_smart_task(arguments: Dict[str, Any], server) -> str:
    # Delegate to server implementation to preserve complex router/refine behavior
    from server import handle_smart_task as _hs
    return _hs(arguments, server)


def handle_smart_plan_execute(arguments: Dict[str, Any], server) -> str:
    # Delegate to server for now; orchestration DAG integration pending
    from server import handle_smart_plan_execute as _sp
    return _sp(arguments, server)


def handle_workflow_create(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    wf = server._require_workflow() if hasattr(server, "_require_workflow") else None
    if wf is None:
        from server import _require_workflow as _rw
        wf = _rw()
    name = (arguments.get("name") or "").strip()
    if not name:
        raise Exception("name is required")
    description = (arguments.get("description") or "").strip()
    wid = wf.create_workflow(name=name, description=description)
    return {"workflow_id": wid}


def handle_workflow_add_node(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    from server import _require_workflow as _rw
    wf = _rw()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    node_type = (arguments.get("node_type") or "").strip()
    name = (arguments.get("name") or "").strip()
    config = arguments.get("config") or {}
    if not (workflow_id and node_type and name):
        raise Exception("workflow_id, node_type, and name are required")
    nid = wf.add_node(workflow_id, node_type, name, config)
    return {"node_id": nid}


def handle_workflow_connect_nodes(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    from server import _require_workflow as _rw
    wf = _rw()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    src = (arguments.get("source_node_id") or "").strip()
    tgt = (arguments.get("target_node_id") or "").strip()
    if not (workflow_id and src and tgt):
        raise Exception("workflow_id, source_node_id, target_node_id are required")
    ok = wf.connect_nodes(workflow_id, src, tgt)
    return {"ok": bool(ok)}


def handle_workflow_explain(arguments: Dict[str, Any], server) -> str:
    from server import _require_workflow as _rw
    wf = _rw()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    if not workflow_id:
        raise Exception("workflow_id is required")
    try:
        return asyncio.get_event_loop().run_until_complete(wf.explain_workflow(workflow_id))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(wf.explain_workflow(workflow_id))
        finally:
            loop.close()


def handle_workflow_execute(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    from server import _require_workflow as _rw
    wf = _rw()
    workflow_id = (arguments.get("workflow_id") or "").strip()
    if not workflow_id:
        raise Exception("workflow_id is required")
    inputs = arguments.get("inputs") or {}
    try:
        out = asyncio.get_event_loop().run_until_complete(wf.execute_workflow(workflow_id, inputs))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            out = loop.run_until_complete(wf.execute_workflow(workflow_id, inputs))
        finally:
            loop.close()
    return out

