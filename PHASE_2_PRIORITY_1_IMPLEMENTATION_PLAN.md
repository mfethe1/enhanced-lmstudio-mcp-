# Phase 2 Priority 1: Short-Lived Agent Pattern - Implementation Plan

## 🎯 Objective
Implement ephemeral agent lifecycle management with max 10 concurrent agents, queue system for overflow, and integration with existing CrewAI-based agent_teams.py.

## 📋 Detailed Reasoning Steps (30+ steps)

### 1. Architecture Analysis
**Step 1**: Review existing agent creation in `handlers/agent_teams.py`
- Current: Agents created via CrewAI `Agent()` constructor
- Current: No lifecycle tracking or cleanup
- Current: No concurrency limits

**Step 2**: Identify integration points
- `server.py`: Main server class, registry system
- `handlers/agent_teams.py`: Agent creation handlers
- `cognitive_architecture/agent_spawner.py`: Existing agent pool (can leverage patterns)
- `strands/agent_factory.py`: Agent specification system

**Step 3**: Define agent lifecycle states
```python
class AgentState(Enum):
    PENDING = "pending"      # Queued, waiting for slot
    CREATING = "creating"    # Being instantiated
    ACTIVE = "active"        # Running task
    COMPLETING = "completing" # Finishing up
    CLEANUP = "cleanup"      # Resources being released
    TERMINATED = "terminated" # Fully cleaned up
```

### 2. Data Structures Design

**Step 4**: Agent Registry Structure
```python
@dataclass
class EphemeralAgent:
    agent_id: str
    role: str
    state: AgentState
    created_at: float
    task_id: Optional[str]
    crew_agent: Any  # CrewAI Agent instance
    cleanup_callbacks: List[Callable]
    max_lifetime_seconds: int = 300  # 5 min default
```

**Step 5**: Queue Structure
```python
@dataclass
class AgentRequest:
    request_id: str
    role: str
    task_description: str
    priority: int = 0
    created_at: float
    timeout_seconds: int = 60
    callback: Optional[Callable] = None
```

**Step 6**: Manager Class Design
```python
class EphemeralAgentManager:
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_agents: Dict[str, EphemeralAgent] = {}
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
```

### 3. Core Functionality

**Step 7**: Agent Creation Logic
- Check if under max_concurrent limit
- If yes: create immediately
- If no: add to queue
- Return agent_id or queue position

**Step 8**: Agent Cleanup Logic
- Call cleanup callbacks (close connections, release memory)
- Remove from active_agents registry
- Update state to TERMINATED
- Process next queued request if any

**Step 9**: Queue Processing
- Background task monitors queue
- When slot available, dequeue highest priority request
- Create agent and execute task
- Auto-cleanup on completion

### 4. Integration with Existing Code

**Step 10**: Modify `handle_agent_team_plan_and_code`
- Replace direct `Agent()` creation with `manager.request_agent()`
- Add cleanup callback after task completion
- Handle queue timeout gracefully

**Step 11**: Modify `handle_agent_collaborate`
- Use ephemeral agents for each role
- Ensure cleanup after each round
- Track agent lifecycle across rounds

**Step 12**: Add to server.py registry
- Register `ephemeral_agent_manager` as singleton
- Expose via MCP tools: `request_ephemeral_agent`, `release_agent`, `get_agent_status`

### 5. Error Handling

**Step 13**: Creation Failures
- If agent creation fails, mark as TERMINATED
- Requeue request with incremented retry count
- Max 3 retries before failing permanently

**Step 14**: Execution Failures
- Catch exceptions during task execution
- Ensure cleanup still happens (try/finally)
- Log error details for debugging

**Step 15**: Cleanup Failures
- If cleanup fails, log warning but continue
- Force-remove from registry after timeout
- Prevent memory leaks from stuck agents

**Step 16**: Queue Overflow
- Set max queue size (e.g., 50 requests)
- Reject new requests if queue full
- Return error with retry-after suggestion

### 6. Performance Optimization

**Step 17**: Fast Agent Creation (<1s target)
- Pre-warm LLM connections if possible
- Use lightweight agent initialization
- Lazy-load heavy resources

**Step 18**: Efficient Cleanup
- Async cleanup to not block new requests
- Batch cleanup operations
- Use weak references where appropriate

**Step 19**: Queue Processing Efficiency
- Process queue in background asyncio task
- Use priority queue for urgent requests
- Batch process multiple requests if possible

### 7. Concurrency Safety

**Step 20**: Thread-Safe Operations
- Use asyncio.Lock for registry modifications
- Atomic operations for state transitions
- No race conditions in queue processing

**Step 21**: Prevent Deadlocks
- Timeout on lock acquisition
- No nested locks
- Clear lock ordering if multiple locks needed

### 8. Monitoring and Observability

**Step 22**: Metrics Collection
- Track: active_agent_count, queue_length, avg_creation_time, cleanup_failures
- Expose via `get_ephemeral_agent_stats()` tool

**Step 23**: Logging
- Log agent lifecycle events (created, active, terminated)
- Log queue operations (enqueued, dequeued, timeout)
- Use structured logging for easy parsing

### 9. Testing Strategy

**Step 24**: Unit Tests
- Test agent creation under limit
- Test queue behavior when at capacity
- Test cleanup logic
- Test error handling paths

**Step 25**: Integration Tests
- Test with real CrewAI agents
- Test concurrent requests (spawn 15, verify only 10 active)
- Test queue timeout behavior
- Test cleanup after task completion

**Step 26**: Performance Tests
- Measure agent creation time (target <1s)
- Measure cleanup time
- Test under load (100 requests)

### 10. Edge Cases

**Step 27**: Agent Timeout
- If agent runs longer than max_lifetime, force cleanup
- Notify caller of timeout
- Prevent zombie agents

**Step 28**: Server Restart
- Agents are ephemeral, so OK to lose on restart
- Queue is in-memory, so also lost (acceptable)
- Document this behavior

**Step 29**: Rapid Request Bursts
- Queue handles bursts gracefully
- Prioritize by request priority
- Shed load if queue full

**Step 30**: Resource Exhaustion
- Monitor memory usage
- If approaching limit, reject new requests
- Force cleanup of oldest agents if needed

### 11. Backward Compatibility

**Step 31**: Existing Code Compatibility
- Old handlers still work (create agents directly)
- New handlers use ephemeral manager
- Gradual migration path

**Step 32**: Configuration
- Environment variable: `MAX_EPHEMERAL_AGENTS` (default 10)
- Environment variable: `EPHEMERAL_AGENT_MAX_QUEUE` (default 50)
- Environment variable: `EPHEMERAL_AGENT_LIFETIME` (default 300s)

---

## 📁 Files to Create/Modify

### New Files
1. `core/ephemeral_agents.py` - Main implementation
2. `tests/test_ephemeral_agents.py` - Comprehensive tests

### Modified Files
1. `handlers/agent_teams.py` - Integrate ephemeral manager
2. `server.py` - Register manager and tools
3. `README.md` - Document new functionality

---

## 🎯 Success Criteria

- ✅ Max 10 concurrent agents enforced
- ✅ Queue handles overflow gracefully
- ✅ Agent creation <1s (95th percentile)
- ✅ Cleanup happens reliably
- ✅ All tests passing
- ✅ Backward compatible with existing code
- ✅ Documentation complete

---

## 🚀 Implementation Order

1. Create `core/ephemeral_agents.py` with basic structure
2. Implement agent lifecycle management
3. Implement queue system
4. Add cleanup logic
5. Integrate with `handlers/agent_teams.py`
6. Add MCP tools to `server.py`
7. Write comprehensive tests
8. Performance optimization
9. Documentation

---

**Status**: Ready to implement  
**Estimated Time**: 4-6 hours  
**Dependencies**: None (standalone module)

