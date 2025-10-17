# Phase 2 Priority 4: Swarm Pattern - Implementation Plan

## Overview
Implement dynamic agent handoffs with agent-to-agent communication protocol for coordinated multi-agent workflows.

**Goal**: Enable agents to dynamically hand off tasks to specialized agents, communicate results, and coordinate work.

**Success Criteria**:
- Handoff reliability >95%
- Communication latency <100ms
- All tests passing (unit + integration)
- Backward compatible
- Documentation complete

---

## Phase 1: Architecture Analysis (Steps 1-5)

### Step 1: Understand Swarm Pattern Requirements
**Analysis**: A swarm pattern enables:
- Dynamic agent handoffs (agent A → agent B)
- Agent-to-agent communication (message passing)
- Task routing (select best agent for task)
- Load balancing (distribute work across agents)
- Coordination (agents work together on complex tasks)

**Key Components**:
1. **SwarmCoordinator**: Central coordinator for agent management
2. **SwarmAgent**: Individual agent with specialization
3. **SwarmMessage**: Message protocol for communication
4. **HandoffProtocol**: Rules for task handoffs
5. **TaskRouter**: Route tasks to appropriate agents

### Step 2: Review Existing Agent Infrastructure
**Existing Components** (from Priorities 1-3):
- `core/ephemeral_agents.py`: Agent lifecycle management
- `handlers/agent_teams.py`: Multi-agent collaboration
- `handlers/workflows.py`: Workflow orchestration

**Integration Points**:
- Use ephemeral agents for swarm members
- Use workflows for coordinated execution
- Extend agent_teams with swarm coordination

### Step 3: Define Swarm Architecture
**Architecture**:
```
SwarmCoordinator
├── Agent Registry (track active agents)
├── Message Queue (agent-to-agent messages)
├── Task Router (route tasks to agents)
├── Handoff Manager (manage handoffs)
└── Monitoring (track swarm activity)

SwarmAgent
├── Specialization (role/expertise)
├── Message Handler (receive messages)
├── Task Executor (execute tasks)
├── Handoff Initiator (initiate handoffs)
└── Status Reporter (report to coordinator)

SwarmMessage
├── sender_id
├── receiver_id
├── message_type (task, result, handoff, status)
├── payload
└── timestamp
```

### Step 4: Design Communication Protocol
**Message Types**:
1. **TASK**: Assign task to agent
2. **RESULT**: Return task result
3. **HANDOFF**: Hand off task to another agent
4. **STATUS**: Report agent status
5. **QUERY**: Query agent capabilities
6. **RESPONSE**: Respond to query

**Protocol**:
- Async message passing (non-blocking)
- Message queue per agent (FIFO)
- Timeout handling (default: 30s)
- Retry logic (max 3 retries)

### Step 5: Design Handoff Protocol
**Handoff Flow**:
1. Agent A determines task needs handoff
2. Agent A queries coordinator for suitable agent
3. Coordinator selects Agent B based on:
   - Specialization match
   - Current load
   - Availability
4. Agent A sends HANDOFF message to Agent B
5. Agent B acknowledges and executes task
6. Agent B sends RESULT back to Agent A or coordinator

**Handoff Criteria**:
- Task complexity exceeds agent capability
- Task requires different specialization
- Agent is overloaded (>5 active tasks)

---

## Phase 2: Core Implementation (Steps 6-15)

### Step 6: Create SwarmMessage Data Structure
**File**: `handlers/swarm.py`

**Implementation**:
```python
@dataclass
class SwarmMessage:
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType  # Enum
    payload: Dict[str, Any]
    timestamp: float
    timeout: Optional[float] = 30.0
    retry_count: int = 0
```

### Step 7: Create SwarmAgent Class
**Implementation**:
```python
class SwarmAgent:
    def __init__(self, agent_id, specialization, max_tasks=5):
        self.agent_id = agent_id
        self.specialization = specialization
        self.max_tasks = max_tasks
        self.active_tasks = []
        self.message_queue = asyncio.Queue()
        self.status = AgentStatus.IDLE
    
    async def receive_message(self, message: SwarmMessage):
        await self.message_queue.put(message)
    
    async def process_messages(self):
        while True:
            message = await self.message_queue.get()
            await self._handle_message(message)
    
    async def execute_task(self, task):
        # Execute task and return result
        pass
    
    async def initiate_handoff(self, task, target_agent_id):
        # Hand off task to another agent
        pass
```

### Step 8: Create SwarmCoordinator Class
**Implementation**:
```python
class SwarmCoordinator:
    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.message_queue = asyncio.Queue()
        self.handoff_history = []
        self.lock = asyncio.Lock()
    
    async def register_agent(self, agent: SwarmAgent):
        async with self.lock:
            self.agents[agent.agent_id] = agent
    
    async def route_task(self, task, specialization=None):
        # Select best agent for task
        best_agent = self._select_agent(task, specialization)
        return best_agent
    
    async def handle_handoff(self, from_agent, to_agent, task):
        # Manage handoff between agents
        pass
    
    def _select_agent(self, task, specialization):
        # Selection logic: specialization match + load balancing
        pass
```

### Step 9: Implement Task Router
**Implementation**:
```python
class TaskRouter:
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
    
    def select_agent(self, task, criteria):
        # Score agents based on:
        # 1. Specialization match (0-1)
        # 2. Current load (0-1, inverse)
        # 3. Availability (0-1)
        # Return agent with highest score
        pass
```

### Step 10: Implement Handoff Manager
**Implementation**:
```python
class HandoffManager:
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
        self.handoff_history = []
    
    async def initiate_handoff(self, from_agent, task, reason):
        # 1. Select target agent
        # 2. Send HANDOFF message
        # 3. Track handoff
        # 4. Return handoff_id
        pass
    
    async def complete_handoff(self, handoff_id, result):
        # Mark handoff as complete
        pass
    
    def get_handoff_stats(self):
        # Return success rate, avg latency
        pass
```

### Step 11: Implement Message Queue System
**Implementation**:
```python
class MessageQueue:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.lock = asyncio.Lock()
    
    async def send_message(self, message: SwarmMessage):
        receiver_id = message.receiver_id
        if receiver_id not in self.queues:
            async with self.lock:
                self.queues[receiver_id] = asyncio.Queue()
        await self.queues[receiver_id].put(message)
    
    async def receive_message(self, agent_id, timeout=30.0):
        if agent_id not in self.queues:
            return None
        try:
            return await asyncio.wait_for(
                self.queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
```

### Step 12: Implement Monitoring/Visualization
**Implementation**:
```python
class SwarmMonitor:
    def __init__(self, coordinator: SwarmCoordinator):
        self.coordinator = coordinator
        self.metrics = {
            "total_tasks": 0,
            "total_handoffs": 0,
            "successful_handoffs": 0,
            "failed_handoffs": 0,
            "avg_handoff_latency": 0.0,
            "agent_utilization": {}
        }
    
    def record_handoff(self, handoff_id, success, latency):
        # Update metrics
        pass
    
    def get_swarm_status(self):
        # Return current swarm state
        return {
            "active_agents": len(self.coordinator.agents),
            "total_tasks": self.metrics["total_tasks"],
            "handoff_success_rate": self._calculate_success_rate(),
            "avg_latency": self.metrics["avg_handoff_latency"]
        }
```

### Step 13: Implement Agent Specializations
**Specializations**:
- `PLANNER`: Task planning and decomposition
- `CODER`: Code generation and editing
- `REVIEWER`: Code review and testing
- `RESEARCHER`: Information gathering
- `OPTIMIZER`: Performance optimization
- `GENERALIST`: General-purpose tasks

### Step 14: Implement Load Balancing
**Algorithm**:
```python
def _calculate_agent_score(agent, task, specialization):
    # Specialization match (0-1)
    spec_score = 1.0 if agent.specialization == specialization else 0.5
    
    # Load score (0-1, inverse of utilization)
    load_score = 1.0 - (len(agent.active_tasks) / agent.max_tasks)
    
    # Availability score (0-1)
    avail_score = 1.0 if agent.status == AgentStatus.IDLE else 0.5
    
    # Weighted average
    return 0.5 * spec_score + 0.3 * load_score + 0.2 * avail_score
```

### Step 15: Implement Error Handling
**Error Cases**:
- No available agents → Queue task or reject
- Handoff timeout → Retry or fail
- Agent crash → Reassign tasks
- Message loss → Retry with exponential backoff

---

## Phase 3: MCP Tool Integration (Steps 16-20)

### Step 16: Create MCP Tool: `create_swarm`
**Purpose**: Create a new swarm with specified agents

**Parameters**:
- `agents` (list): List of agent definitions (id, specialization)
- `coordinator_config` (dict): Coordinator configuration

**Returns**: Swarm ID and agent IDs

### Step 17: Create MCP Tool: `execute_swarm_task`
**Purpose**: Execute a task using swarm coordination

**Parameters**:
- `swarm_id` (str): Swarm identifier
- `task` (dict): Task definition
- `preferred_specialization` (str): Preferred agent type
- `allow_handoffs` (bool): Allow dynamic handoffs

**Returns**: Task result and execution trace

### Step 18: Create MCP Tool: `get_swarm_status`
**Purpose**: Get current swarm status and metrics

**Parameters**:
- `swarm_id` (str): Swarm identifier

**Returns**: Swarm status, agent utilization, handoff metrics

### Step 19: Create MCP Tool: `visualize_swarm`
**Purpose**: Generate visualization of swarm activity

**Parameters**:
- `swarm_id` (str): Swarm identifier
- `format` (str): Output format (json, mermaid, ascii)

**Returns**: Visualization data

### Step 20: Register MCP Tools in server.py
**Registration**:
```python
server.registry.register("create_swarm", swarm.handle_create_swarm, needs_server=True)
server.registry.register("execute_swarm_task", swarm.handle_execute_swarm_task, needs_server=True)
server.registry.register("get_swarm_status", swarm.handle_get_swarm_status, needs_server=True)
server.registry.register("visualize_swarm", swarm.handle_visualize_swarm, needs_server=True)
```

---

## Phase 4: Testing (Steps 21-25)

### Step 21: Create Unit Tests for SwarmMessage
**Tests**:
- Message creation and serialization
- Message validation
- Timeout handling

### Step 22: Create Unit Tests for SwarmAgent
**Tests**:
- Agent creation and initialization
- Message receiving and processing
- Task execution
- Handoff initiation
- Status reporting

### Step 23: Create Unit Tests for SwarmCoordinator
**Tests**:
- Agent registration
- Task routing
- Handoff management
- Load balancing
- Error handling

### Step 24: Create Integration Tests
**Tests**:
- End-to-end swarm execution
- Multi-agent handoff scenarios
- Communication latency measurement
- Handoff reliability measurement
- Integration with ephemeral agents

### Step 25: Create Performance Tests
**Metrics**:
- Handoff success rate (target: >95%)
- Communication latency (target: <100ms)
- Agent utilization
- Task throughput

---

## Phase 5: Documentation and Completion (Steps 26-30)

### Step 26: Update README.md
**Sections**:
- Swarm Pattern overview
- Usage examples
- Agent specializations
- Configuration options
- Performance metrics

### Step 27: Create Completion Summary
**File**: `PHASE_2_PRIORITY_4_COMPLETE.md`

**Contents**:
- Implementation summary
- Test results
- Performance metrics
- Usage examples
- Next steps

### Step 28: Validate Success Criteria
**Checklist**:
- ✅ All swarm components implemented
- ✅ Handoff reliability >95%
- ✅ Communication latency <100ms
- ✅ All tests passing
- ✅ Backward compatible
- ✅ Documentation complete

### Step 29: Commit Changes
**Commit Message**:
```
feat: Phase 2 Priority 4 - Swarm Pattern

Implemented dynamic agent handoffs with communication protocol:
- handlers/swarm.py: SwarmCoordinator, SwarmAgent, HandoffManager
- 4 new MCP tools: create_swarm, execute_swarm_task, get_swarm_status, visualize_swarm
- Tests: X/X passing
- Performance: >95% handoff success, <100ms latency

Status: PRODUCTION READY
```

### Step 30: Prepare for Phase 3
**Next Steps**:
- Review all Phase 2 priorities (1-4)
- Validate all tests passing
- Confirm production readiness
- Begin Phase 3 planning

---

## Implementation Order

1. **Core Data Structures** (Steps 6-7): SwarmMessage, SwarmAgent
2. **Coordination Layer** (Steps 8-12): SwarmCoordinator, TaskRouter, HandoffManager, MessageQueue, SwarmMonitor
3. **Specializations** (Step 13): Define agent types
4. **Load Balancing** (Step 14): Agent selection algorithm
5. **Error Handling** (Step 15): Robust error recovery
6. **MCP Tools** (Steps 16-20): User-facing tools
7. **Testing** (Steps 21-25): Comprehensive test suite
8. **Documentation** (Steps 26-30): Complete docs and commit

---

**Estimated Time**: 8-10 hours
**Complexity**: High (multi-agent coordination, async messaging)
**Risk**: Medium (complex state management, race conditions)

