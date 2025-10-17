"""
Ephemeral Agent Lifecycle Management

Implements short-lived agent pattern with:
- Max concurrent agents enforcement (default 10)
- Queue system for overflow requests
- Automatic cleanup after task completion
- Integration with CrewAI-based agent system
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from crewai import Agent

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    PENDING = "pending"          # Queued, waiting for slot
    CREATING = "creating"        # Being instantiated
    ACTIVE = "active"            # Running task
    COMPLETING = "completing"    # Finishing up
    CLEANUP = "cleanup"          # Resources being released
    TERMINATED = "terminated"    # Fully cleaned up


@dataclass
class EphemeralAgent:
    """Represents a short-lived agent with lifecycle tracking"""
    agent_id: str
    role: str
    state: AgentState
    created_at: float
    task_id: Optional[str] = None
    crew_agent: Optional[Agent] = None
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    max_lifetime_seconds: int = 300  # 5 min default
    retry_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if agent has exceeded max lifetime"""
        return (time.time() - self.created_at) > self.max_lifetime_seconds
    
    def age_seconds(self) -> float:
        """Get agent age in seconds"""
        return time.time() - self.created_at


@dataclass
class AgentRequest:
    """Represents a request for an ephemeral agent"""
    request_id: str
    role: str
    task_description: str
    priority: int = 0  # Higher = more urgent
    created_at: float = field(default_factory=time.time)
    timeout_seconds: int = 60
    callback: Optional[Callable] = None
    retry_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if request has timed out"""
        return (time.time() - self.created_at) > self.timeout_seconds
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority > other.priority


class EphemeralAgentManager:
    """
    Manages lifecycle of short-lived agents with concurrency limits.
    
    Features:
    - Max concurrent agents enforcement
    - Queue system for overflow
    - Automatic cleanup
    - Performance monitoring
    """
    
    def __init__(self, max_concurrent: int = 10, max_queue_size: int = 50):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_agents: Dict[str, EphemeralAgent] = {}
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.total_created = 0
        self.total_cleaned = 0
        self.total_failed = 0
        self.creation_times: List[float] = []
        
    async def start(self):
        """Start background tasks for cleanup and queue processing"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._queue_processor_task = asyncio.create_task(self._queue_processor_loop())
        logger.info(f"EphemeralAgentManager started (max_concurrent={self.max_concurrent})")
    
    async def stop(self):
        """Stop background tasks and cleanup all agents"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all active agents
        async with self.lock:
            for agent_id in list(self.active_agents.keys()):
                await self._cleanup_agent(agent_id)
        
        logger.info("EphemeralAgentManager stopped")
    
    async def request_agent(
        self,
        role: str,
        task_description: str,
        priority: int = 0,
        timeout_seconds: int = 60,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Request an ephemeral agent. Returns request_id.
        
        If under capacity, creates immediately.
        If at capacity, queues request.
        
        Args:
            role: Agent role (e.g., "backend", "frontend")
            task_description: What the agent will do
            priority: Higher = more urgent (default 0)
            timeout_seconds: Max wait time in queue (default 60)
            callback: Optional callback when agent ready
            
        Returns:
            request_id: Unique identifier for this request
            
        Raises:
            asyncio.QueueFull: If queue is at max capacity
        """
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        request = AgentRequest(
            request_id=request_id,
            role=role,
            task_description=task_description,
            priority=priority,
            timeout_seconds=timeout_seconds,
            callback=callback
        )
        
        async with self.lock:
            # Check if we can create immediately
            if len(self.active_agents) < self.max_concurrent:
                agent_id = await self._create_agent_internal(request)
                logger.info(f"Agent {agent_id} created immediately for request {request_id}")
                return agent_id
        
        # Queue the request (use put_nowait to raise QueueFull immediately)
        try:
            self.request_queue.put_nowait(request)
            logger.info(f"Request {request_id} queued (priority={priority}, queue_size={self.request_queue.qsize()})")
            return request_id
        except asyncio.QueueFull:
            logger.error(f"Request {request_id} rejected: queue full ({self.max_queue_size})")
            raise
    
    async def _create_agent_internal(self, request: AgentRequest) -> str:
        """Internal method to create agent (assumes lock held)"""
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Create ephemeral agent record
            agent = EphemeralAgent(
                agent_id=agent_id,
                role=request.role,
                state=AgentState.CREATING,
                created_at=start_time,
                task_id=request.request_id
            )
            self.active_agents[agent_id] = agent
            
            # Create actual CrewAI agent (this is the slow part)
            # For now, we'll create a minimal agent - integration with CrewAI comes later
            crew_agent = Agent(
                role=request.role,
                goal=request.task_description,
                backstory=f"Ephemeral agent for task: {request.task_description[:100]}",
                verbose=False,
                allow_delegation=False
            )
            
            agent.crew_agent = crew_agent
            agent.state = AgentState.ACTIVE
            
            creation_time = time.time() - start_time
            self.creation_times.append(creation_time)
            self.total_created += 1
            
            logger.info(f"Agent {agent_id} created in {creation_time:.3f}s (role={request.role})")
            
            # Call callback if provided
            if request.callback:
                try:
                    await request.callback(agent_id, crew_agent)
                except Exception as e:
                    logger.error(f"Callback failed for agent {agent_id}: {e}")
            
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent for request {request.request_id}: {e}")
            self.total_failed += 1
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            raise
    
    async def release_agent(self, agent_id: str):
        """
        Release an agent, triggering cleanup.
        
        Args:
            agent_id: ID of agent to release
        """
        async with self.lock:
            await self._cleanup_agent(agent_id)
    
    async def _cleanup_agent(self, agent_id: str):
        """Internal cleanup method (assumes lock held)"""
        if agent_id not in self.active_agents:
            logger.warning(f"Agent {agent_id} not found for cleanup")
            return
        
        agent = self.active_agents[agent_id]
        agent.state = AgentState.CLEANUP
        
        try:
            # Run cleanup callbacks
            for callback in agent.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(agent)
                    else:
                        callback(agent)
                except Exception as e:
                    logger.error(f"Cleanup callback failed for agent {agent_id}: {e}")
            
            # Clear crew agent reference
            agent.crew_agent = None
            agent.state = AgentState.TERMINATED
            
            # Remove from registry
            del self.active_agents[agent_id]
            self.total_cleaned += 1
            
            logger.info(f"Agent {agent_id} cleaned up (age={agent.age_seconds():.1f}s)")
            
        except Exception as e:
            logger.error(f"Cleanup failed for agent {agent_id}: {e}")
            # Force remove even if cleanup failed
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]

    async def _cleanup_loop(self):
        """Background task to cleanup expired agents"""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                async with self.lock:
                    expired = [
                        agent_id for agent_id, agent in self.active_agents.items()
                        if agent.is_expired()
                    ]

                    for agent_id in expired:
                        logger.warning(f"Agent {agent_id} expired, forcing cleanup")
                        await self._cleanup_agent(agent_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _queue_processor_loop(self):
        """Background task to process queued requests"""
        while self._running:
            try:
                # Wait for a request (with timeout to allow checking _running)
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if request expired while in queue
                if request.is_expired():
                    logger.warning(f"Request {request.request_id} expired in queue")
                    continue

                # Wait for a slot to become available
                while self._running:
                    # Check expiration again before waiting
                    if request.is_expired():
                        logger.warning(f"Request {request.request_id} expired while waiting for slot")
                        break

                    async with self.lock:
                        if len(self.active_agents) < self.max_concurrent:
                            # Final expiration check before creating
                            if request.is_expired():
                                logger.warning(f"Request {request.request_id} expired just before creation")
                                break

                            try:
                                agent_id = await self._create_agent_internal(request)
                                logger.info(f"Agent {agent_id} created from queue for request {request.request_id}")
                                break
                            except Exception as e:
                                logger.error(f"Failed to create agent from queue: {e}")
                                # Retry logic
                                if request.retry_count < 3:
                                    request.retry_count += 1
                                    self.request_queue.put_nowait(request)
                                    logger.info(f"Request {request.request_id} requeued (retry {request.retry_count}/3)")
                                else:
                                    logger.error(f"Request {request.request_id} failed after 3 retries")
                                break

                    # Wait a bit before checking again
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor loop: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        avg_creation_time = (
            sum(self.creation_times[-100:]) / len(self.creation_times[-100:])
            if self.creation_times else 0.0
        )

        return {
            "active_agents": len(self.active_agents),
            "max_concurrent": self.max_concurrent,
            "queue_size": self.request_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "total_created": self.total_created,
            "total_cleaned": self.total_cleaned,
            "total_failed": self.total_failed,
            "avg_creation_time_ms": avg_creation_time * 1000,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "state": agent.state.value,
                    "age_seconds": agent.age_seconds(),
                    "task_id": agent.task_id
                }
                for agent in self.active_agents.values()
            ]
        }

    def get_agent(self, agent_id: str) -> Optional[EphemeralAgent]:
        """Get agent by ID"""
        return self.active_agents.get(agent_id)

    def add_cleanup_callback(self, agent_id: str, callback: Callable):
        """Add a cleanup callback to an agent"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id].cleanup_callbacks.append(callback)
        else:
            logger.warning(f"Cannot add cleanup callback: agent {agent_id} not found")


# Global singleton instance
_manager_singleton: Optional[EphemeralAgentManager] = None


def get_ephemeral_agent_manager(
    max_concurrent: int = 10,
    max_queue_size: int = 50
) -> EphemeralAgentManager:
    """Get or create the global ephemeral agent manager singleton"""
    global _manager_singleton

    if _manager_singleton is None:
        _manager_singleton = EphemeralAgentManager(
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size
        )

    return _manager_singleton

