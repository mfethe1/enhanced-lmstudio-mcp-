"""
Comprehensive tests for ephemeral agent lifecycle management.

Test Categories:
1. Agent Creation and Initialization
2. Max Concurrent Agents Enforcement
3. Queue System Behavior
4. Agent Cleanup and Resource Deallocation
5. Performance (agent creation <1s)
6. Integration with Existing Handlers
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from core.ephemeral_agents import (
    EphemeralAgentManager,
    EphemeralAgent,
    AgentRequest,
    AgentState,
    get_ephemeral_agent_manager
)


class TestAgentCreationAndInitialization:
    """Test basic agent creation and initialization"""
    
    @pytest.mark.asyncio
    async def test_create_agent_under_limit(self):
        """Test creating agent when under max_concurrent limit"""
        manager = EphemeralAgentManager(max_concurrent=10)
        await manager.start()
        
        try:
            agent_id = await manager.request_agent(
                role="backend",
                task_description="Test task",
                priority=0
            )
            
            assert agent_id is not None
            assert agent_id.startswith("agent-")
            
            # Verify agent is in active registry
            agent = manager.get_agent(agent_id)
            assert agent is not None
            assert agent.role == "backend"
            assert agent.state == AgentState.ACTIVE
            assert agent.crew_agent is not None
            
            # Verify stats
            stats = manager.get_stats()
            assert stats["active_agents"] == 1
            assert stats["total_created"] == 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_states(self):
        """Test agent transitions through lifecycle states"""
        manager = EphemeralAgentManager(max_concurrent=10)
        await manager.start()
        
        try:
            agent_id = await manager.request_agent(
                role="frontend",
                task_description="Build UI component"
            )
            
            agent = manager.get_agent(agent_id)
            assert agent.state == AgentState.ACTIVE
            
            # Release agent
            await manager.release_agent(agent_id)
            
            # Verify cleanup
            agent_after = manager.get_agent(agent_id)
            assert agent_after is None  # Should be removed from registry
            
            stats = manager.get_stats()
            assert stats["active_agents"] == 0
            assert stats["total_cleaned"] == 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_agent_expiration(self):
        """Test agent auto-cleanup after max_lifetime"""
        manager = EphemeralAgentManager(max_concurrent=10)
        await manager.start()
        
        try:
            # Create agent with very short lifetime
            agent_id = await manager.request_agent(
                role="testing",
                task_description="Short-lived task"
            )
            
            agent = manager.get_agent(agent_id)
            agent.max_lifetime_seconds = 1  # 1 second lifetime
            
            # Wait for expiration + cleanup loop
            await asyncio.sleep(12)  # Cleanup loop runs every 10s
            
            # Agent should be cleaned up
            agent_after = manager.get_agent(agent_id)
            assert agent_after is None
            
        finally:
            await manager.stop()


class TestMaxConcurrentEnforcement:
    """Test max concurrent agents enforcement"""
    
    @pytest.mark.asyncio
    async def test_enforce_max_concurrent(self):
        """Test that max_concurrent limit is enforced"""
        manager = EphemeralAgentManager(max_concurrent=3)
        await manager.start()
        
        try:
            # Create 3 agents (at limit)
            agent_ids = []
            for i in range(3):
                agent_id = await manager.request_agent(
                    role=f"agent_{i}",
                    task_description=f"Task {i}"
                )
                agent_ids.append(agent_id)
            
            stats = manager.get_stats()
            assert stats["active_agents"] == 3
            
            # 4th request should be queued
            request_id = await manager.request_agent(
                role="agent_4",
                task_description="Task 4"
            )
            
            # Should be queued, not created immediately
            assert request_id.startswith("req-")
            stats = manager.get_stats()
            assert stats["active_agents"] == 3  # Still at limit
            assert stats["queue_size"] == 1
            
            # Release one agent
            await manager.release_agent(agent_ids[0])
            
            # Wait for queue processor to create agent
            await asyncio.sleep(2)
            
            stats = manager.get_stats()
            assert stats["active_agents"] == 3  # Back to limit
            assert stats["queue_size"] == 0  # Queue processed
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_beyond_limit(self):
        """Test handling many concurrent requests beyond max_concurrent"""
        manager = EphemeralAgentManager(max_concurrent=5, max_queue_size=20)
        await manager.start()
        
        try:
            # Create 15 requests (5 immediate, 10 queued)
            request_ids = []
            for i in range(15):
                request_id = await manager.request_agent(
                    role=f"agent_{i}",
                    task_description=f"Task {i}",
                    priority=i  # Vary priority
                )
                request_ids.append(request_id)
            
            stats = manager.get_stats()
            assert stats["active_agents"] == 5  # At limit
            assert stats["queue_size"] == 10  # Rest queued
            
            # Release all agents
            for agent_id in request_ids[:5]:
                if agent_id.startswith("agent-"):
                    await manager.release_agent(agent_id)
            
            # Wait for queue to process
            await asyncio.sleep(5)
            
            stats = manager.get_stats()
            assert stats["active_agents"] == 5  # Should fill back up
            assert stats["queue_size"] <= 5  # Some processed
            
        finally:
            await manager.stop()


class TestQueueSystemBehavior:
    """Test queue system behavior"""
    
    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self):
        """Test that higher priority requests are processed first"""
        manager = EphemeralAgentManager(max_concurrent=1, max_queue_size=10)
        await manager.start()
        
        try:
            # Create one agent to fill capacity
            agent_id = await manager.request_agent(
                role="blocker",
                task_description="Blocking task",
                priority=0
            )
            
            # Queue requests with different priorities
            low_priority = await manager.request_agent(
                role="low",
                task_description="Low priority task",
                priority=1
            )
            
            high_priority = await manager.request_agent(
                role="high",
                task_description="High priority task",
                priority=10
            )
            
            medium_priority = await manager.request_agent(
                role="medium",
                task_description="Medium priority task",
                priority=5
            )
            
            stats = manager.get_stats()
            assert stats["queue_size"] == 3
            
            # Release blocker
            await manager.release_agent(agent_id)
            
            # Wait for queue processor
            await asyncio.sleep(2)
            
            # High priority should be processed first
            stats = manager.get_stats()
            assert stats["active_agents"] == 1
            
            # Check which agent was created (should be high priority)
            active_agent = list(manager.active_agents.values())[0]
            assert active_agent.role == "high"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        """Test that requests are rejected when queue is full"""
        manager = EphemeralAgentManager(max_concurrent=1, max_queue_size=2)
        await manager.start()
        
        try:
            # Fill capacity
            agent_id = await manager.request_agent(
                role="blocker",
                task_description="Blocking task"
            )
            
            # Fill queue
            req1 = await manager.request_agent(role="q1", task_description="Q1")
            req2 = await manager.request_agent(role="q2", task_description="Q2")
            
            stats = manager.get_stats()
            assert stats["queue_size"] == 2
            
            # Next request should raise QueueFull
            with pytest.raises(asyncio.QueueFull):
                await manager.request_agent(role="overflow", task_description="Overflow")
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_request_timeout_in_queue(self):
        """Test that requests timeout if waiting too long in queue"""
        manager = EphemeralAgentManager(max_concurrent=1)
        await manager.start()
        
        try:
            # Fill capacity
            agent_id = await manager.request_agent(
                role="blocker",
                task_description="Blocking task"
            )
            
            # Queue request with short timeout
            req_id = await manager.request_agent(
                role="timeout_test",
                task_description="Will timeout",
                timeout_seconds=1
            )
            
            # Wait for timeout
            await asyncio.sleep(2)
            
            # Release blocker
            await manager.release_agent(agent_id)
            
            # Wait for queue processor
            await asyncio.sleep(2)
            
            # Timed-out request should not create agent
            stats = manager.get_stats()
            assert stats["active_agents"] == 0  # No agent created
            
        finally:
            await manager.stop()


class TestAgentCleanup:
    """Test agent cleanup and resource deallocation"""
    
    @pytest.mark.asyncio
    async def test_cleanup_callbacks_executed(self):
        """Test that cleanup callbacks are executed"""
        manager = EphemeralAgentManager(max_concurrent=10)
        await manager.start()
        
        try:
            agent_id = await manager.request_agent(
                role="test",
                task_description="Test cleanup"
            )
            
            # Add cleanup callback
            callback_executed = False
            
            def cleanup_callback(agent):
                nonlocal callback_executed
                callback_executed = True
            
            manager.add_cleanup_callback(agent_id, cleanup_callback)
            
            # Release agent
            await manager.release_agent(agent_id)
            
            # Verify callback was executed
            assert callback_executed is True
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_cleanup_on_manager_stop(self):
        """Test that all agents are cleaned up when manager stops"""
        manager = EphemeralAgentManager(max_concurrent=10)
        await manager.start()
        
        # Create multiple agents
        agent_ids = []
        for i in range(5):
            agent_id = await manager.request_agent(
                role=f"agent_{i}",
                task_description=f"Task {i}"
            )
            agent_ids.append(agent_id)
        
        stats = manager.get_stats()
        assert stats["active_agents"] == 5
        
        # Stop manager
        await manager.stop()
        
        # All agents should be cleaned up
        stats = manager.get_stats()
        assert stats["active_agents"] == 0
        assert stats["total_cleaned"] == 5


class TestPerformance:
    """Test performance requirements"""
    
    @pytest.mark.asyncio
    async def test_agent_creation_time(self):
        """Test that agent creation is <1s (95th percentile)"""
        manager = EphemeralAgentManager(max_concurrent=20)
        await manager.start()
        
        try:
            creation_times = []
            
            # Create 20 agents and measure time
            for i in range(20):
                start = time.time()
                agent_id = await manager.request_agent(
                    role=f"perf_test_{i}",
                    task_description=f"Performance test {i}"
                )
                creation_time = time.time() - start
                creation_times.append(creation_time)
            
            # Calculate 95th percentile
            creation_times.sort()
            p95 = creation_times[int(len(creation_times) * 0.95)]
            
            print(f"\nAgent creation times (95th percentile): {p95:.3f}s")
            assert p95 < 1.0, f"95th percentile creation time {p95:.3f}s exceeds 1s target"
            
            # Check average too
            avg = sum(creation_times) / len(creation_times)
            print(f"Average creation time: {avg:.3f}s")
            
        finally:
            await manager.stop()


class TestSingletonPattern:
    """Test singleton manager pattern"""
    
    def test_singleton_returns_same_instance(self):
        """Test that get_ephemeral_agent_manager returns singleton"""
        manager1 = get_ephemeral_agent_manager()
        manager2 = get_ephemeral_agent_manager()
        
        assert manager1 is manager2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

