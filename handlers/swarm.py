"""
Swarm Pattern Implementation (Phase 2 Priority 4)

Enables dynamic agent handoffs with agent-to-agent communication protocol
for coordinated multi-agent workflows.

Author: Jarvis MCP Team
Date: 2025-01-16
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class MessageType(Enum):
    """Types of messages in swarm communication"""
    TASK = "task"
    RESULT = "result"
    HANDOFF = "handoff"
    STATUS = "status"
    QUERY = "query"
    RESPONSE = "response"


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"


class AgentSpecialization(Enum):
    """Agent specialization types"""
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    OPTIMIZER = "optimizer"
    GENERALIST = "generalist"


class HandoffStatus(Enum):
    """Handoff status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SwarmMessage:
    """Message for agent-to-agent communication"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


@dataclass
class HandoffRecord:
    """Record of a task handoff"""
    handoff_id: str
    from_agent_id: str
    to_agent_id: str
    task: Dict[str, Any]
    reason: str
    status: HandoffStatus
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def latency(self) -> float:
        """Calculate handoff latency in milliseconds"""
        if self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class AgentMetrics:
    """Metrics for an agent"""
    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    handoffs_initiated: int = 0
    handoffs_received: int = 0
    total_execution_time: float = 0.0
    avg_task_time: float = 0.0


# ============================================================================
# SwarmAgent
# ============================================================================

class SwarmAgent:
    """Individual agent in the swarm"""
    
    def __init__(
        self,
        agent_id: str,
        specialization: AgentSpecialization,
        max_tasks: int = 5,
        coordinator: Optional[SwarmCoordinator] = None
    ):
        self.agent_id = agent_id
        self.specialization = specialization
        self.max_tasks = max_tasks
        self.coordinator = coordinator
        
        self.active_tasks: List[Dict[str, Any]] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics(agent_id=agent_id)
        
        self._running = False
        self._task = None
    
    async def start(self):
        """Start agent message processing"""
        self._running = True
        self._task = asyncio.create_task(self._process_messages())
    
    async def stop(self):
        """Stop agent message processing"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def receive_message(self, message: SwarmMessage):
        """Receive a message"""
        await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process messages from queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: SwarmMessage):
        """Handle a received message"""
        try:
            if message.message_type == MessageType.TASK:
                await self._handle_task(message)
            elif message.message_type == MessageType.HANDOFF:
                await self._handle_handoff(message)
            elif message.message_type == MessageType.QUERY:
                await self._handle_query(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
        except Exception as e:
            logger.error(f"Error handling message {message.message_id}: {e}")
    
    async def _handle_task(self, message: SwarmMessage):
        """Handle a task message"""
        task = message.payload.get("task", {})
        
        # Check if overloaded
        if len(self.active_tasks) >= self.max_tasks:
            self.status = AgentStatus.OVERLOADED
            # Initiate handoff
            if self.coordinator:
                await self._initiate_handoff(task, "Agent overloaded")
            return
        
        # Execute task
        self.status = AgentStatus.BUSY
        self.active_tasks.append(task)
        
        start_time = time.time()
        try:
            result = await self._execute_task(task)
            duration = time.time() - start_time
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.total_execution_time += duration
            self.metrics.avg_task_time = (
                self.metrics.total_execution_time / self.metrics.tasks_completed
            )
            
            # Send result back
            if self.coordinator:
                result_msg = SwarmMessage(
                    message_id=f"msg-{uuid.uuid4().hex[:8]}",
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESULT,
                    payload={"result": result, "task_id": task.get("task_id")}
                )
                await self.coordinator.send_message(result_msg)
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.metrics.tasks_failed += 1
        
        finally:
            self.active_tasks.remove(task)
            self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.BUSY
    
    async def _handle_handoff(self, message: SwarmMessage):
        """Handle a handoff message"""
        task = message.payload.get("task", {})
        handoff_id = message.payload.get("handoff_id")
        
        # Accept handoff
        self.metrics.handoffs_received += 1
        
        # Execute task (same as _handle_task)
        task_msg = SwarmMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender_id="coordinator",
            receiver_id=self.agent_id,
            message_type=MessageType.TASK,
            payload={"task": task, "handoff_id": handoff_id}
        )
        await self._handle_task(task_msg)
    
    async def _handle_query(self, message: SwarmMessage):
        """Handle a query message"""
        query_type = message.payload.get("query_type")
        
        response_payload = {}
        if query_type == "capabilities":
            response_payload = {
                "specialization": self.specialization.value,
                "max_tasks": self.max_tasks,
                "active_tasks": len(self.active_tasks),
                "status": self.status.value
            }
        elif query_type == "metrics":
            response_payload = {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "avg_task_time": self.metrics.avg_task_time
            }
        
        # Send response
        if self.coordinator:
            response_msg = SwarmMessage(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                payload=response_payload
            )
            await self.coordinator.send_message(response_msg)
    
    async def _execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task (placeholder - override in subclass)"""
        # Simulate task execution
        await asyncio.sleep(0.1)
        return {"status": "success", "agent": self.agent_id, "task": task}
    
    async def _initiate_handoff(self, task: Dict[str, Any], reason: str):
        """Initiate a handoff to another agent"""
        if not self.coordinator:
            return
        
        self.metrics.handoffs_initiated += 1
        
        # Request handoff from coordinator
        handoff_msg = SwarmMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender_id=self.agent_id,
            receiver_id="coordinator",
            message_type=MessageType.HANDOFF,
            payload={"task": task, "reason": reason}
        )
        await self.coordinator.send_message(handoff_msg)


# ============================================================================
# SwarmCoordinator
# ============================================================================

class SwarmCoordinator:
    """Central coordinator for swarm management"""

    def __init__(self, swarm_id: Optional[str] = None):
        self.swarm_id = swarm_id or f"swarm-{uuid.uuid4().hex[:8]}"
        self.agents: Dict[str, SwarmAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.handoff_history: List[HandoffRecord] = []
        self.lock = asyncio.Lock()

        self._running = False
        self._task = None

    async def start(self):
        """Start coordinator"""
        self._running = True
        self._task = asyncio.create_task(self._process_messages())

        # Start all agents
        for agent in self.agents.values():
            await agent.start()

    async def stop(self):
        """Stop coordinator"""
        self._running = False

        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def register_agent(self, agent: SwarmAgent):
        """Register an agent with the swarm"""
        async with self.lock:
            self.agents[agent.agent_id] = agent
            agent.coordinator = self
            logger.info(f"Registered agent {agent.agent_id} with specialization {agent.specialization.value}")

    async def send_message(self, message: SwarmMessage):
        """Send a message to an agent"""
        receiver_id = message.receiver_id

        if receiver_id == "coordinator":
            # Message for coordinator
            await self.message_queue.put(message)
        elif receiver_id in self.agents:
            # Message for agent
            await self.agents[receiver_id].receive_message(message)
        else:
            logger.warning(f"Unknown receiver: {receiver_id}")

    async def _process_messages(self):
        """Process messages to coordinator"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing coordinator message: {e}")

    async def _handle_message(self, message: SwarmMessage):
        """Handle a message to coordinator"""
        try:
            if message.message_type == MessageType.HANDOFF:
                await self._handle_handoff_request(message)
            elif message.message_type == MessageType.RESULT:
                # Store result (could be extended)
                pass
            else:
                logger.warning(f"Coordinator received unexpected message type: {message.message_type}")
        except Exception as e:
            logger.error(f"Error handling coordinator message: {e}")

    async def _handle_handoff_request(self, message: SwarmMessage):
        """Handle a handoff request"""
        task = message.payload.get("task", {})
        reason = message.payload.get("reason", "Unknown")
        from_agent_id = message.sender_id

        # Select target agent
        target_agent = await self._select_agent_for_task(task, exclude=[from_agent_id])

        if not target_agent:
            logger.error(f"No available agent for handoff from {from_agent_id}")
            return

        # Create handoff record
        handoff_id = f"handoff-{uuid.uuid4().hex[:8]}"
        handoff_record = HandoffRecord(
            handoff_id=handoff_id,
            from_agent_id=from_agent_id,
            to_agent_id=target_agent.agent_id,
            task=task,
            reason=reason,
            status=HandoffStatus.IN_PROGRESS
        )
        self.handoff_history.append(handoff_record)

        # Send handoff message to target agent
        handoff_msg = SwarmMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender_id="coordinator",
            receiver_id=target_agent.agent_id,
            message_type=MessageType.HANDOFF,
            payload={"task": task, "handoff_id": handoff_id}
        )
        await self.send_message(handoff_msg)

        logger.info(f"Handoff {handoff_id}: {from_agent_id} -> {target_agent.agent_id}")

    async def _select_agent_for_task(
        self,
        task: Dict[str, Any],
        specialization: Optional[AgentSpecialization] = None,
        exclude: List[str] = None
    ) -> Optional[SwarmAgent]:
        """Select best agent for a task"""
        exclude = exclude or []

        # Filter available agents
        available_agents = [
            agent for agent_id, agent in self.agents.items()
            if agent_id not in exclude and agent.status != AgentStatus.OFFLINE
        ]

        if not available_agents:
            return None

        # Score agents
        best_agent = None
        best_score = -1.0

        for agent in available_agents:
            score = self._calculate_agent_score(agent, task, specialization)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def _calculate_agent_score(
        self,
        agent: SwarmAgent,
        task: Dict[str, Any],
        specialization: Optional[AgentSpecialization] = None
    ) -> float:
        """Calculate agent suitability score"""
        # Specialization match (0-1)
        if specialization:
            spec_score = 1.0 if agent.specialization == specialization else 0.5
        else:
            spec_score = 0.7  # Neutral score

        # Load score (0-1, inverse of utilization)
        load_score = 1.0 - (len(agent.active_tasks) / agent.max_tasks)

        # Availability score (0-1)
        if agent.status == AgentStatus.IDLE:
            avail_score = 1.0
        elif agent.status == AgentStatus.BUSY:
            avail_score = 0.5
        else:  # OVERLOADED
            avail_score = 0.1

        # Weighted average
        return 0.5 * spec_score + 0.3 * load_score + 0.2 * avail_score

    async def execute_task(
        self,
        task: Dict[str, Any],
        specialization: Optional[AgentSpecialization] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Execute a task using the swarm"""
        # Select agent
        agent = await self._select_agent_for_task(task, specialization)

        if not agent:
            return {"status": "error", "error": "No available agents"}

        # Send task to agent
        task_msg = SwarmMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            sender_id="coordinator",
            receiver_id=agent.agent_id,
            message_type=MessageType.TASK,
            payload={"task": task},
            timeout=timeout
        )

        await self.send_message(task_msg)

        # Wait for result (simplified - in production, use proper result tracking)
        await asyncio.sleep(0.2)  # Give agent time to process

        return {
            "status": "success",
            "agent_id": agent.agent_id,
            "task": task
        }

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        total_handoffs = len(self.handoff_history)
        successful_handoffs = sum(
            1 for h in self.handoff_history
            if h.status == HandoffStatus.COMPLETED
        )

        handoff_success_rate = (
            successful_handoffs / total_handoffs if total_handoffs > 0 else 0.0
        )

        avg_latency = 0.0
        if successful_handoffs > 0:
            latencies = [h.latency for h in self.handoff_history if h.completed_at]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "swarm_id": self.swarm_id,
            "active_agents": len(self.agents),
            "agent_status": {
                agent_id: {
                    "specialization": agent.specialization.value,
                    "status": agent.status.value,
                    "active_tasks": len(agent.active_tasks),
                    "tasks_completed": agent.metrics.tasks_completed
                }
                for agent_id, agent in self.agents.items()
            },
            "total_handoffs": total_handoffs,
            "successful_handoffs": successful_handoffs,
            "handoff_success_rate": handoff_success_rate,
            "avg_handoff_latency_ms": avg_latency
        }


# ============================================================================
# Global Swarm Registry
# ============================================================================

_swarm_registry: Dict[str, SwarmCoordinator] = {}
_registry_lock = asyncio.Lock()


async def get_or_create_swarm(swarm_id: Optional[str] = None) -> SwarmCoordinator:
    """Get existing swarm or create new one"""
    async with _registry_lock:
        if swarm_id and swarm_id in _swarm_registry:
            return _swarm_registry[swarm_id]

        # Create new swarm
        coordinator = SwarmCoordinator(swarm_id=swarm_id)
        _swarm_registry[coordinator.swarm_id] = coordinator
        return coordinator


async def get_swarm(swarm_id: str) -> Optional[SwarmCoordinator]:
    """Get existing swarm"""
    async with _registry_lock:
        return _swarm_registry.get(swarm_id)


# ============================================================================
# MCP Tool Handlers
# ============================================================================

def handle_create_swarm(arguments: Dict[str, Any], server) -> str:
    """
    Create a new swarm with specified agents

    Args:
        arguments: {
            "agents": [
                {"agent_id": "agent-1", "specialization": "planner", "max_tasks": 5},
                {"agent_id": "agent-2", "specialization": "coder", "max_tasks": 5}
            ],
            "swarm_id": "optional-swarm-id"
        }

    Returns:
        JSON with swarm_id and agent_ids
    """
    try:
        agents_config = arguments.get("agents", [])
        swarm_id = arguments.get("swarm_id")

        # Create swarm
        loop = asyncio.get_event_loop()
        coordinator = loop.run_until_complete(get_or_create_swarm(swarm_id))

        # Create and register agents
        agent_ids = []
        for agent_config in agents_config:
            agent_id = agent_config.get("agent_id", f"agent-{uuid.uuid4().hex[:8]}")
            spec_str = agent_config.get("specialization", "generalist")
            max_tasks = agent_config.get("max_tasks", 5)

            # Parse specialization
            try:
                specialization = AgentSpecialization(spec_str)
            except ValueError:
                specialization = AgentSpecialization.GENERALIST

            # Create agent
            agent = SwarmAgent(
                agent_id=agent_id,
                specialization=specialization,
                max_tasks=max_tasks
            )

            # Register agent
            loop.run_until_complete(coordinator.register_agent(agent))
            agent_ids.append(agent_id)

        # Start coordinator
        loop.run_until_complete(coordinator.start())

        return json.dumps({
            "status": "success",
            "swarm_id": coordinator.swarm_id,
            "agent_ids": agent_ids,
            "total_agents": len(agent_ids)
        }, indent=2)

    except Exception as e:
        logger.error(f"Error creating swarm: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


def handle_execute_swarm_task(arguments: Dict[str, Any], server) -> str:
    """
    Execute a task using swarm coordination

    Args:
        arguments: {
            "swarm_id": "swarm-abc123",
            "task": {"task_id": "task-1", "description": "...", ...},
            "specialization": "planner",  # optional
            "timeout": 30.0  # optional
        }

    Returns:
        JSON with task result and execution trace
    """
    try:
        swarm_id = arguments.get("swarm_id")
        task = arguments.get("task", {})
        spec_str = arguments.get("specialization")
        timeout = arguments.get("timeout", 30.0)

        if not swarm_id:
            return json.dumps({"status": "error", "error": "swarm_id required"}, indent=2)

        # Get swarm
        loop = asyncio.get_event_loop()
        coordinator = loop.run_until_complete(get_swarm(swarm_id))

        if not coordinator:
            return json.dumps({"status": "error", "error": f"Swarm {swarm_id} not found"}, indent=2)

        # Parse specialization
        specialization = None
        if spec_str:
            try:
                specialization = AgentSpecialization(spec_str)
            except ValueError:
                pass

        # Execute task
        result = loop.run_until_complete(
            coordinator.execute_task(task, specialization, timeout)
        )

        return json.dumps({
            "status": "success",
            "swarm_id": swarm_id,
            "result": result
        }, indent=2)

    except Exception as e:
        logger.error(f"Error executing swarm task: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


def handle_get_swarm_status(arguments: Dict[str, Any], server) -> str:
    """
    Get current swarm status and metrics

    Args:
        arguments: {
            "swarm_id": "swarm-abc123"
        }

    Returns:
        JSON with swarm status, agent utilization, handoff metrics
    """
    try:
        swarm_id = arguments.get("swarm_id")

        if not swarm_id:
            return json.dumps({"status": "error", "error": "swarm_id required"}, indent=2)

        # Get swarm
        loop = asyncio.get_event_loop()
        coordinator = loop.run_until_complete(get_swarm(swarm_id))

        if not coordinator:
            return json.dumps({"status": "error", "error": f"Swarm {swarm_id} not found"}, indent=2)

        # Get status
        status = coordinator.get_swarm_status()

        return json.dumps({
            "status": "success",
            **status
        }, indent=2)

    except Exception as e:
        logger.error(f"Error getting swarm status: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


def handle_visualize_swarm(arguments: Dict[str, Any], server) -> str:
    """
    Generate visualization of swarm activity

    Args:
        arguments: {
            "swarm_id": "swarm-abc123",
            "format": "json"  # json, mermaid, ascii
        }

    Returns:
        Visualization data in requested format
    """
    try:
        swarm_id = arguments.get("swarm_id")
        format_type = arguments.get("format", "json")

        if not swarm_id:
            return json.dumps({"status": "error", "error": "swarm_id required"}, indent=2)

        # Get swarm
        loop = asyncio.get_event_loop()
        coordinator = loop.run_until_complete(get_swarm(swarm_id))

        if not coordinator:
            return json.dumps({"status": "error", "error": f"Swarm {swarm_id} not found"}, indent=2)

        if format_type == "json":
            # JSON format
            status = coordinator.get_swarm_status()
            return json.dumps({
                "status": "success",
                "format": "json",
                "visualization": status
            }, indent=2)

        elif format_type == "mermaid":
            # Mermaid diagram
            lines = ["graph TD"]
            lines.append(f"    COORD[Coordinator: {swarm_id}]")

            for agent_id, agent in coordinator.agents.items():
                lines.append(f"    {agent_id}[{agent_id}<br/>{agent.specialization.value}<br/>{agent.status.value}]")
                lines.append(f"    COORD --> {agent_id}")

            # Add handoffs
            for i, handoff in enumerate(coordinator.handoff_history[-5:]):  # Last 5 handoffs
                lines.append(f"    {handoff.from_agent_id} -.handoff.-> {handoff.to_agent_id}")

            mermaid_diagram = "\n".join(lines)

            return json.dumps({
                "status": "success",
                "format": "mermaid",
                "visualization": mermaid_diagram
            }, indent=2)

        elif format_type == "ascii":
            # ASCII art
            lines = [f"Swarm: {swarm_id}"]
            lines.append("=" * 60)
            lines.append(f"Agents: {len(coordinator.agents)}")
            lines.append("")

            for agent_id, agent in coordinator.agents.items():
                lines.append(f"  [{agent.status.value.upper()}] {agent_id}")
                lines.append(f"    Specialization: {agent.specialization.value}")
                lines.append(f"    Active Tasks: {len(agent.active_tasks)}/{agent.max_tasks}")
                lines.append(f"    Completed: {agent.metrics.tasks_completed}")
                lines.append("")

            lines.append(f"Handoffs: {len(coordinator.handoff_history)}")
            for handoff in coordinator.handoff_history[-3:]:  # Last 3
                lines.append(f"  {handoff.from_agent_id} -> {handoff.to_agent_id}: {handoff.status.value}")

            ascii_viz = "\n".join(lines)

            return json.dumps({
                "status": "success",
                "format": "ascii",
                "visualization": ascii_viz
            }, indent=2)

        else:
            return json.dumps({"status": "error", "error": f"Unknown format: {format_type}"}, indent=2)

    except Exception as e:
        logger.error(f"Error visualizing swarm: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

