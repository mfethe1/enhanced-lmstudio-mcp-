"""
Unit tests for Swarm Pattern (Phase 2 Priority 4)

Tests SwarmAgent, SwarmCoordinator, and swarm communication protocol.

Author: Jarvis MCP Team
Date: 2025-01-16
"""

import asyncio
import pytest
import sys
import time

# Add parent directory to path
sys.path.insert(0, ".")

from handlers.swarm import (
    SwarmAgent,
    SwarmCoordinator,
    SwarmMessage,
    MessageType,
    AgentStatus,
    AgentSpecialization,
    HandoffStatus,
    HandoffRecord
)


class TestSwarmMessage:
    """Test SwarmMessage data structure"""
    
    def test_message_creation(self):
        """Test message creation"""
        msg = SwarmMessage(
            message_id="msg-1",
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.TASK,
            payload={"task": "test"}
        )
        
        assert msg.message_id == "msg-1"
        assert msg.sender_id == "agent-1"
        assert msg.receiver_id == "agent-2"
        assert msg.message_type == MessageType.TASK
        assert msg.payload == {"task": "test"}
        assert msg.timeout == 30.0
        assert msg.retry_count == 0
    
    def test_message_to_dict(self):
        """Test message serialization"""
        msg = SwarmMessage(
            message_id="msg-1",
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.TASK,
            payload={"task": "test"}
        )
        
        msg_dict = msg.to_dict()
        
        assert msg_dict["message_id"] == "msg-1"
        assert msg_dict["sender_id"] == "agent-1"
        assert msg_dict["receiver_id"] == "agent-2"
        assert msg_dict["message_type"] == "task"
        assert msg_dict["payload"] == {"task": "test"}


class TestHandoffRecord:
    """Test HandoffRecord data structure"""
    
    def test_handoff_record_creation(self):
        """Test handoff record creation"""
        record = HandoffRecord(
            handoff_id="handoff-1",
            from_agent_id="agent-1",
            to_agent_id="agent-2",
            task={"task_id": "task-1"},
            reason="Agent overloaded",
            status=HandoffStatus.PENDING
        )
        
        assert record.handoff_id == "handoff-1"
        assert record.from_agent_id == "agent-1"
        assert record.to_agent_id == "agent-2"
        assert record.status == HandoffStatus.PENDING
    
    def test_handoff_latency(self):
        """Test handoff latency calculation"""
        record = HandoffRecord(
            handoff_id="handoff-1",
            from_agent_id="agent-1",
            to_agent_id="agent-2",
            task={"task_id": "task-1"},
            reason="Test",
            status=HandoffStatus.IN_PROGRESS
        )
        
        # Initially no latency
        assert record.latency == 0.0
        
        # Complete handoff
        time.sleep(0.05)  # 50ms
        record.completed_at = time.time()
        
        # Should have latency > 0
        assert record.latency > 0


class TestSwarmAgent:
    """Test SwarmAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test agent creation"""
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER,
            max_tasks=5
        )
        
        assert agent.agent_id == "agent-1"
        assert agent.specialization == AgentSpecialization.PLANNER
        assert agent.max_tasks == 5
        assert agent.status == AgentStatus.IDLE
        assert len(agent.active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self):
        """Test agent start and stop"""
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER
        )
        
        await agent.start()
        assert agent._running is True
        
        await agent.stop()
        assert agent._running is False
    
    @pytest.mark.asyncio
    async def test_agent_receive_message(self):
        """Test agent message receiving"""
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER
        )
        
        msg = SwarmMessage(
            message_id="msg-1",
            sender_id="coordinator",
            receiver_id="agent-1",
            message_type=MessageType.TASK,
            payload={"task": {"task_id": "task-1"}}
        )
        
        await agent.receive_message(msg)
        
        # Message should be in queue
        assert agent.message_queue.qsize() == 1


class TestSwarmCoordinator:
    """Test SwarmCoordinator functionality"""
    
    @pytest.mark.asyncio
    async def test_coordinator_creation(self):
        """Test coordinator creation"""
        coordinator = SwarmCoordinator(swarm_id="swarm-1")
        
        assert coordinator.swarm_id == "swarm-1"
        assert len(coordinator.agents) == 0
        assert len(coordinator.handoff_history) == 0
    
    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test agent registration"""
        coordinator = SwarmCoordinator()
        
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER
        )
        
        await coordinator.register_agent(agent)
        
        assert "agent-1" in coordinator.agents
        assert agent.coordinator == coordinator
    
    @pytest.mark.asyncio
    async def test_agent_selection(self):
        """Test agent selection for task"""
        coordinator = SwarmCoordinator()
        
        # Register agents with different specializations
        agent1 = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER,
            max_tasks=5
        )
        agent2 = SwarmAgent(
            agent_id="agent-2",
            specialization=AgentSpecialization.CODER,
            max_tasks=5
        )
        
        await coordinator.register_agent(agent1)
        await coordinator.register_agent(agent2)
        
        # Select agent for planning task
        task = {"task_id": "task-1", "type": "planning"}
        selected = await coordinator._select_agent_for_task(
            task,
            specialization=AgentSpecialization.PLANNER
        )
        
        assert selected == agent1
    
    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """Test load balancing across agents"""
        coordinator = SwarmCoordinator()
        
        # Register two agents with same specialization
        agent1 = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.GENERALIST,
            max_tasks=5
        )
        agent2 = SwarmAgent(
            agent_id="agent-2",
            specialization=AgentSpecialization.GENERALIST,
            max_tasks=5
        )
        
        await coordinator.register_agent(agent1)
        await coordinator.register_agent(agent2)
        
        # Overload agent1
        agent1.active_tasks = [{"task": i} for i in range(4)]
        agent1.status = AgentStatus.BUSY
        
        # Select agent for task
        task = {"task_id": "task-1"}
        selected = await coordinator._select_agent_for_task(task)
        
        # Should select agent2 (less loaded)
        assert selected == agent2
    
    @pytest.mark.asyncio
    async def test_swarm_status(self):
        """Test swarm status reporting"""
        coordinator = SwarmCoordinator(swarm_id="swarm-1")
        
        # Register agents
        agent1 = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.PLANNER
        )
        agent2 = SwarmAgent(
            agent_id="agent-2",
            specialization=AgentSpecialization.CODER
        )
        
        await coordinator.register_agent(agent1)
        await coordinator.register_agent(agent2)
        
        # Get status
        status = coordinator.get_swarm_status()
        
        assert status["swarm_id"] == "swarm-1"
        assert status["active_agents"] == 2
        assert "agent-1" in status["agent_status"]
        assert "agent-2" in status["agent_status"]
        assert status["total_handoffs"] == 0


class TestSwarmIntegration:
    """Test end-to-end swarm functionality"""
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        """Test task execution through swarm"""
        coordinator = SwarmCoordinator()
        
        # Register agent
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.GENERALIST
        )
        
        await coordinator.register_agent(agent)
        await coordinator.start()
        
        # Execute task
        task = {"task_id": "task-1", "description": "Test task"}
        result = await coordinator.execute_task(task)
        
        assert result["status"] == "success"
        assert result["agent_id"] == "agent-1"
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_communication_latency(self):
        """Test communication latency is <100ms"""
        coordinator = SwarmCoordinator()
        
        agent = SwarmAgent(
            agent_id="agent-1",
            specialization=AgentSpecialization.GENERALIST
        )
        
        await coordinator.register_agent(agent)
        await coordinator.start()
        
        # Measure latency
        start_time = time.time()
        
        msg = SwarmMessage(
            message_id="msg-1",
            sender_id="coordinator",
            receiver_id="agent-1",
            message_type=MessageType.QUERY,
            payload={"query_type": "capabilities"}
        )
        
        await coordinator.send_message(msg)
        await asyncio.sleep(0.05)  # Wait for processing
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Should be < 100ms
        assert latency_ms < 100
        
        await coordinator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

