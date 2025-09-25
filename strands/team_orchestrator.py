"""
TeamOrchestrator: assembles expert teams and coordinates plan/execute workflows
on top of the existing LM Studio MCP server and its tools.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os, json, logging

logger = logging.getLogger(__name__)

from .agent_factory import AgentFactory
from .team_memory import TeamMemory
from .workflows import WORKFLOW_TEMPLATES
from .rag import HybridRetriever
from .ingestion import KGIngestionEngine


@dataclass
class OrchestratorConfig:
    project_id: str
    max_agents: int = 6
    parallelism: int = 3
    dry_run: bool = True


class TeamOrchestrator:
    def __init__(self, server, config: OrchestratorConfig):
        self.server = server
        self.config = config
        self.factory = AgentFactory(server)
        self.memory = TeamMemory(server, project_id=config.project_id)
        # Hybrid retriever (KG + memory); safe even if external backends not configured
        self.retriever = HybridRetriever(server)
        # Ingestion engine for converting data into KG nodes/edges
        self.ingestion_engine = KGIngestionEngine(server)

    def _route(self, prompt: str, role: Optional[str], backend: Optional[str]) -> str:
        preferred = backend if backend in {"lmstudio","openai","anthropic"} else None
        # Use server's intelligent router (reuses circuit breakers and monitoring)
        try:
            return self._run_sync(self.server.route_chat(prompt, intent="planning", role=role, preferred_backend=preferred))
        except Exception:
            # Fallback path ensures robustness during development
            return self._run_sync(self.server.route_chat(prompt, intent="planning", role=role))

    def _run_sync(self, coro):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In MCP server context, we usually have a loop; create a new one
                new_loop = asyncio.new_event_loop(); asyncio.set_event_loop(new_loop)
                out = new_loop.run_until_complete(coro)
                new_loop.close()
                asyncio.set_event_loop(loop)
                return out
            return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop(); out = loop.run_until_complete(coro); loop.close(); return out

    def assemble_team(self, task_description: str) -> List[Dict[str, Any]]:
        specs = self.factory.from_description(task_description)
        specs = specs[: self.config.max_agents]
        agents = self.factory.instantiate(specs)
        # Persist the assembled team
        self.memory.add("orchestrator", "team", {"agents": agents})
        return agents

    def choose_workflow(self, task_description: str) -> Dict[str, Any]:
        text = task_description.lower()
        wf = "literature_to_experiment" if any(k in text for k in ["paper","experiment","hypothesis"]) else "software_feature"
        plan = WORKFLOW_TEMPLATES[wf]
        self.memory.add("orchestrator", "workflow", {"name": wf, "plan": plan})
        return plan

    def execute_stage(self, stage: Dict[str, Any], agents: List[Dict[str, Any]], instruction: str, context: str = "") -> Dict[str, Any]:
        owner = stage.get("owner"); tools = stage.get("tools", [])
        agent = next((a for a in agents if a.get("role") == owner), agents[0]) if agents else {"role": "research_coordinator"}
        # Combine system + task prompt
        sys_prompt = agent.get("system", "")
        # Optional hybrid retrieval augmentation with enhanced context
        ctx_aug = context
        try:
            if os.getenv("USE_HYBRID_RETRIEVAL", "1") == "1":
                retrieval_context = {
                    'project_id': self.config.project_id,
                    'agent_role': agent.get('role'),
                    'stage': stage.get('name', 'unknown')
                }
                hit = self.retriever.retrieve(instruction, top_k=5, context=retrieval_context)

                # Enhanced context augmentation with multiple sources
                ctx_parts = [context]
                if hit.get('combined'):
                    ctx_parts.append(f"[Retrieved Context ({hit.get('strategy_used', 'hybrid')})]:")
                    for i, result in enumerate(hit['combined'][:3]):  # Top 3 results
                        source = result.get('source', 'unknown')
                        content = result.get('content', str(result))[:200]  # Truncate
                        ctx_parts.append(f"  {i+1}. [{source}] {content}")

                ctx_aug = "\n".join(ctx_parts)
        except Exception as e:
            logger.warning(f"Hybrid retrieval failed: {e}")
            ctx_aug = context
        task_prompt = f"System: {sys_prompt}\nInstruction: {instruction}\nContext: {ctx_aug}"

        # If smart_plan_execute in tools, prefer it; else use smart_task for a single step
        if "smart_plan_execute" in tools:
            args = {"instruction": instruction, "context": context, "dry_run": self.config.dry_run, "max_steps": 4}
            import server as _srvmod
            result = _srvmod.handle_smart_plan_execute(args, self.server)
        elif "smart_task" in tools:
            args = {"instruction": instruction, "context": context, "dry_run": True}
            import server as _srvmod
            result = _srvmod.handle_smart_task(args, self.server)
        else:
            # Router free-form assistance
            out = self._route(task_prompt, role=agent.get("role"), backend=agent.get("preferred_backend"))
            result = {"assistant": out}

        self.memory.add(agent.get("role","agent"), "stage_result", {"stage": stage.get("name"), "result": result})

        # Ingest agent interaction into KG for future retrieval
        try:
            if os.getenv("ENABLE_KG_INGESTION", "1") == "1":
                ingestion_result = self.ingestion_engine.ingest_agent_interaction(
                    agent_role=agent.get("role", "unknown"),
                    task=instruction,
                    result=str(result)[:1000],  # Truncate for storage
                    project_id=self.config.project_id
                )
                logger.debug(f"Ingested agent interaction: {ingestion_result.nodes_created} nodes, {ingestion_result.edges_created} edges")
        except Exception as e:
            logger.warning(f"Failed to ingest agent interaction: {e}")

        return result

    def orchestrate(self, task: str, context: str = "") -> Dict[str, Any]:
        agents = self.assemble_team(task)
        workflow = self.choose_workflow(task)
        results = []
        for stage in workflow.get("stages", []):
            r = self.execute_stage(stage, agents, instruction=task, context=context)
            results.append({"stage": stage.get("name"), "owner": stage.get("owner"), "result": r})
        summary = {"task": task, "agents": agents, "results": results}
        self.memory.add("orchestrator", "summary", summary)

        # Ingest team memory into KG for enhanced retrieval
        try:
            if os.getenv("ENABLE_KG_INGESTION", "1") == "1":
                ingestion_result = self.ingestion_engine.ingest_team_memory(
                    self.memory, self.config.project_id
                )
                logger.debug(f"Ingested team memory: {ingestion_result.nodes_created} nodes, {ingestion_result.edges_created} edges")
        except Exception as e:
            logger.warning(f"Failed to ingest team memory: {e}")

        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the selection router."""
        try:
            return self.retriever.router.get_performance_summary()
        except Exception as e:
            logger.warning(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

