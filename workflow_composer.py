from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    AGENT = "agent"
    TOOL = "tool"
    TRANSFORM = "transform"


@dataclass
class WorkflowNode:
    node_id: str
    name: str
    node_type: str
    config: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]


@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    created_at: float
    version: int


class WorkflowComposer:
    """Minimal workflow composer with storage-backed persistence.
    This is a lightweight v1 aligned with recommendation 7; extend incrementally.
    """

    def __init__(self, server, storage):
        self.server = server
        self.storage = storage
        self._cache: Dict[str, Workflow] = {}

    def _save(self, wf: Workflow) -> None:
        payload = asdict(wf)
        payload["nodes"] = [asdict(n) for n in wf.nodes]
        self.storage.store_memory(key=f"workflow_{wf.workflow_id}", value=json.dumps(payload), category="workflows")

    def _load(self, workflow_id: str) -> Optional[Workflow]:
        if workflow_id in self._cache:
            return self._cache[workflow_id]
        rows = self.storage.retrieve_memory(key=f"workflow_{workflow_id}", category="workflows")
        if not rows:
            return None
        data = json.loads(rows[0].get("value", "{}"))
        nodes = [WorkflowNode(**n) for n in data.get("nodes", [])]
        wf = Workflow(workflow_id=workflow_id, name=data.get("name",""), description=data.get("description",""), nodes=nodes, created_at=data.get("created_at", time.time()), version=int(data.get("version",1)))
        self._cache[workflow_id] = wf
        return wf

    def create_workflow(self, name: str, description: str = "") -> str:
        wid = f"wf_{uuid.uuid4().hex[:10]}"
        wf = Workflow(workflow_id=wid, name=name, description=description, nodes=[], created_at=time.time(), version=1)
        self._cache[wid] = wf
        self._save(wf)
        return wid

    def add_node(self, workflow_id: str, node_type: str, name: str, config: Dict[str, Any]) -> str:
        wf = self._load(workflow_id)
        if not wf:
            raise ValueError("workflow not found")
        nid = f"node_{uuid.uuid4().hex[:8]}"
        node = WorkflowNode(node_id=nid, name=name, node_type=node_type, config=config or {}, inputs=[], outputs=[])
        wf.nodes.append(node)
        wf.version += 1
        self._save(wf)
        return nid

    def connect_nodes(self, workflow_id: str, source_node_id: str, target_node_id: str) -> bool:
        wf = self._load(workflow_id)
        if not wf:
            raise ValueError("workflow not found")
        src = next((n for n in wf.nodes if n.node_id == source_node_id), None)
        tgt = next((n for n in wf.nodes if n.node_id == target_node_id), None)
        if not src or not tgt:
            raise ValueError("source or target not found")
        if target_node_id not in src.outputs:
            src.outputs.append(target_node_id)
        if source_node_id not in tgt.inputs:
            tgt.inputs.append(source_node_id)
        wf.version += 1
        self._save(wf)
        return True

    async def explain_workflow(self, workflow_id: str) -> str:
        wf = self._load(workflow_id)
        if not wf:
            raise ValueError("workflow not found")
        desc_lines = [f"Workflow {wf.name}: {wf.description}", f"Nodes: {len(wf.nodes)}"]
        for n in wf.nodes:
            desc_lines.append(f"- {n.name} ({n.node_type}) inputs={len(n.inputs)} outputs={len(n.outputs)} config_keys={list(n.config.keys())}")
        prompt = "Explain this workflow to a technical audience and suggest improvements.\n\n" + "\n".join(desc_lines)
        return await self.server.make_llm_request_with_retry(prompt, temperature=0.3)

    async def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal execution: returns the topology and echoes inputs
        wf = self._load(workflow_id)
        if not wf:
            raise ValueError("workflow not found")
        return {
            "workflow_id": workflow_id,
            "node_count": len(wf.nodes),
            "inputs": inputs,
            "status": "completed",
        }

