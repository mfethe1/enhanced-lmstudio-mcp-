# Phase 2 Priority 2: File-Level Locking - Implementation Plan

## 🎯 Objective
Implement file-level locking to prevent concurrent modifications by multiple agents, with timeout mechanisms, conflict detection, and integration with plan execution.

## 📋 Detailed Reasoning Steps (30+ steps)

### 1. Architecture Analysis

**Step 1**: Review existing file modification patterns
- Current: No locking mechanism exists
- Current: Multiple agents can modify same file simultaneously
- Current: Plan execution doesn't track file access
- Risk: Race conditions and data corruption

**Step 2**: Identify integration points
- `handlers/plan_generator.py`: Plan execution with file lists
- `handlers/agent_teams.py`: Agent file modifications
- `core/ephemeral_agents.py`: Agent lifecycle (just implemented)
- `server.py`: File operations and tool handlers

**Step 3**: Define lock granularity
- **File-level**: Lock individual files (chosen approach)
- Not directory-level (too coarse)
- Not line-level (too fine-grained)
- Rationale: Balance between safety and concurrency

### 2. Data Structures Design

**Step 4**: Lock State Representation
```python
class LockState(Enum):
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    PENDING = "pending"      # Waiting for lock
    EXPIRED = "expired"      # Timeout occurred
    FORCE_RELEASED = "force_released"  # Manually released
```

**Step 5**: File Lock Metadata
```python
@dataclass
class FileLock:
    file_path: str
    state: LockState
    owner_id: str  # Agent ID or task ID
    acquired_at: float
    expires_at: float
    timeout_seconds: int = 60
    lock_id: str  # Unique lock identifier
    retry_count: int = 0
```

**Step 6**: Lock Registry Structure
```python
class FileLockManager:
    def __init__(self, default_timeout: int = 60):
        self.locks: Dict[str, FileLock] = {}  # file_path -> FileLock
        self.pending_requests: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.lock = asyncio.Lock()  # Protect registry modifications
        self._cleanup_task: Optional[asyncio.Task] = None
```

### 3. Core Functionality

**Step 7**: Lock Acquisition Logic
- Check if file already locked
- If unlocked: acquire immediately
- If locked: add to pending queue
- Wait for lock with timeout
- Return lock_id on success

**Step 8**: Lock Release Logic
- Verify owner matches
- Update state to UNLOCKED
- Remove from registry
- Process next pending request if any
- Call cleanup callbacks

**Step 9**: Timeout Handling
- Background task checks for expired locks every 5s
- If lock expired: force release
- Notify owner of timeout
- Log timeout event

**Step 10**: Conflict Detection
- Detect when multiple agents request same file
- Track pending requests per file
- Provide conflict resolution strategies:
  - Priority-based (higher priority wins)
  - FIFO (first come, first served)
  - Fail-fast (reject if locked)

### 4. Deadlock Prevention

**Step 11**: Deadlock Detection Strategy
- **Approach 1**: Lock ordering (alphabetical file paths)
- **Approach 2**: Timeout-based (force release after timeout)
- **Approach 3**: Dependency graph analysis (detect cycles)
- **Chosen**: Combination of timeout + lock ordering

**Step 12**: Lock Ordering Implementation
- Sort file paths alphabetically before locking
- Always acquire locks in same order
- Prevents circular wait condition
- Example: Lock [a.py, b.py, c.py] always in that order

**Step 13**: Timeout as Deadlock Breaker
- Default 60s timeout
- If timeout occurs, assume deadlock
- Force release and retry
- Max 3 retries before failing

### 5. Integration with Plan Execution

**Step 14**: Auto-Lock During Task Execution
- Modify `handle_agent_team_plan_and_code` to lock files
- Extract file list from task specification
- Acquire locks before execution
- Release locks after completion (try/finally)

**Step 15**: Integration with TaskPlan
- Add `files_to_lock` field to AtomicTask
- Automatically extract from `files_affected`
- Lock all files before task starts
- Release all files when task completes

**Step 16**: Integration with Ephemeral Agents
- When agent created, lock its assigned files
- When agent released, unlock its files
- Add lock cleanup to agent cleanup callbacks

### 6. Error Handling

**Step 17**: Lock Acquisition Failures
- Timeout: Return error with retry suggestion
- Already locked: Return conflict info (owner, expires_at)
- Invalid file path: Return validation error
- Max retries exceeded: Return permanent failure

**Step 18**: Lock Release Failures
- Owner mismatch: Log warning, allow force release
- Lock not found: Log warning, return success (idempotent)
- Cleanup callback failure: Log error, continue release

**Step 19**: Force Release Mechanism
- Admin tool: `force_release_lock(file_path, reason)`
- Requires justification
- Logs force release event
- Notifies original owner

### 7. Performance Optimization

**Step 20**: Fast Lock Acquisition (<100ms target)
- Use asyncio.Lock for thread-safe operations
- Minimize lock hold time
- Use weak references where appropriate
- Cache lock lookups

**Step 21**: Efficient Cleanup
- Background task runs every 5s (not every 1s)
- Batch process expired locks
- Use indexed data structures (dict, not list)

**Step 22**: Scalability Considerations
- Support 100+ concurrent locks
- Efficient pending queue processing
- Memory-efficient lock metadata

### 8. Monitoring and Observability

**Step 23**: Metrics Collection
- Track: active_locks, pending_requests, timeouts, force_releases
- Track: avg_lock_acquisition_time, max_lock_hold_time
- Expose via `get_file_lock_stats()` tool

**Step 24**: Logging
- Log lock acquisition (file, owner, timeout)
- Log lock release (file, owner, duration)
- Log conflicts (file, requesters)
- Log timeouts and force releases

### 9. Testing Strategy

**Step 25**: Unit Tests
- Test lock acquisition (unlocked file)
- Test lock acquisition (locked file, timeout)
- Test lock release
- Test timeout handling
- Test force release
- Test lock ordering

**Step 26**: Concurrent Access Tests
- Spawn 10 agents trying to lock same file
- Verify only 1 succeeds
- Verify others wait or timeout
- Verify all eventually get lock

**Step 27**: Deadlock Prevention Tests
- Create scenario with circular dependencies
- Verify lock ordering prevents deadlock
- Verify timeout breaks deadlock
- Verify max retries prevents infinite loops

**Step 28**: Integration Tests
- Test with real plan execution
- Test with ephemeral agents
- Test auto-lock during task execution
- Test cleanup on agent termination

### 10. Edge Cases

**Step 29**: Rapid Lock/Unlock Cycles
- Agent locks, unlocks, locks again quickly
- Verify no race conditions
- Verify lock state consistency

**Step 30**: Server Restart
- Locks are in-memory, so lost on restart
- Document this behavior
- Consider persistence for future enhancement

**Step 31**: File Deletion While Locked
- Detect file deletion
- Auto-release lock
- Notify owner

**Step 32**: Lock Inheritance
- If agent spawns sub-agent, inherit locks?
- Decision: No inheritance (explicit locking only)
- Rationale: Clearer ownership model

---

## 📁 Files to Create/Modify

### New Files
1. `core/file_locking.py` - Main implementation
2. `tests/test_file_locking.py` - Comprehensive tests

### Modified Files
1. `handlers/agent_teams.py` - Integrate auto-locking
2. `handlers/plan_generator.py` - Add file locking to plan execution
3. `server.py` - Register lock management tools
4. `README.md` - Document new functionality

---

## 🎯 Success Criteria

- ✅ Lock acquisition <100ms (95th percentile)
- ✅ Prevents concurrent modifications (100% reliable)
- ✅ Timeout handling works correctly
- ✅ Deadlock prevention effective
- ✅ All tests passing
- ✅ Backward compatible
- ✅ Documentation complete

---

## 🚀 Implementation Order

1. Create `core/file_locking.py` with basic structure
2. Implement lock acquisition and release
3. Implement timeout handling
4. Implement deadlock prevention (lock ordering)
5. Add monitoring and stats
6. Integrate with plan execution
7. Integrate with ephemeral agents
8. Write comprehensive tests
9. Performance optimization
10. Documentation

---

**Status**: Ready to implement  
**Estimated Time**: 4-6 hours  
**Dependencies**: Phase 2 Priority 1 (Ephemeral Agents) - COMPLETE

