from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class ToolNode:
    name: str
    dependencies: Set[str]
    arguments: Dict[str, Any]


class DAGExecutor:
    """Execute a workflow of tools with dependency ordering and parallel levels.

    This implementation avoids external libraries to keep it portable.
    """

    def __init__(self, context_manager) -> None:
        self.context_manager = context_manager

    async def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        tools: Dict[str, Dict[str, Any]] = workflow.get("tools", {})
        graph = self._build_graph(tools)
        order = self._topo_sort(graph)
        levels = self._parallel_levels(graph, order)

        results: Dict[str, Any] = {}
        for level in levels:
            tasks = [self._execute_tool(name, tools[name], results) for name in level]
            level_results = await asyncio.gather(*tasks, return_exceptions=True)
            for tool_name, res in zip(level, level_results):
                if isinstance(res, Exception):
                    results[tool_name] = {"error": str(res)}
                else:
                    results[tool_name] = res
                    await self.context_manager.add_tool_result(tool_name, res)

        return {"workflow_id": workflow.get("id"), "results": results, "execution_order": order}

    async def _execute_tool(self, name: str, spec: Dict[str, Any], prior: Dict[str, Any]) -> Any:
        handler = spec.get("handler")
        args = dict(spec.get("arguments", {}))
        # naive context pass-through: include any direct dependencies' results
        for dep in spec.get("dependencies", []) or []:
            if dep in prior:
                args.setdefault("_inputs", {})[dep] = prior[dep]
        if asyncio.iscoroutinefunction(handler):
            return await handler(args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, handler, args)

    def _build_graph(self, tools: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = {k: set((v.get("dependencies") or [])) for k, v in tools.items()}
        return graph

    def _topo_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        # Kahn's algorithm
        incoming = {n: set(deps) for n, deps in graph.items()}
        no_incoming = [n for n, deps in incoming.items() if not deps]
        order: List[str] = []
        while no_incoming:
            n = no_incoming.pop()
            order.append(n)
            for m, deps in incoming.items():
                if n in deps:
                    deps.remove(n)
                    if not deps:
                        no_incoming.append(m)
        if any(incoming.values()):
            raise ValueError("Cycle detected in workflow graph")
        return order

    def _parallel_levels(self, graph: Dict[str, Set[str]], order: List[str]) -> List[List[str]]:
        levels: List[List[str]] = []
        processed: Set[str] = set()
        remaining = list(order)
        while remaining:
            level: List[str] = []
            for node in list(remaining):
                deps = graph.get(node, set())
                if deps.issubset(processed):
                    level.append(node)
                    remaining.remove(node)
            levels.append(level)
            processed.update(level)
        return levels

