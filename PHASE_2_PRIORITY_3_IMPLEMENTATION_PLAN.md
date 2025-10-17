# Phase 2 Priority 3: Composable Workflows - Implementation Plan

## 🎯 Objective
Implement workflow patterns (Parallel, Sequential, Evaluator-Optimizer) for orchestrating multiple tasks with proper concurrency, error handling, and result aggregation.

---

## 📋 30-Step Reasoning Plan

### Phase 1: Architecture Analysis (Steps 1-5)

**Step 1: Review Existing Systems**
- Ephemeral agents (Phase 2 Priority 1): Short-lived agents with lifecycle management
- File locking (Phase 2 Priority 2): Prevent concurrent file modifications
- Task specification (Phase 1): Structured task definitions with validation
- Integration points: How workflows will use these systems

**Step 2: Define Workflow Abstractions**
- Base `Workflow` class with common functionality
- Workflow states: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- Result aggregation strategies: ALL, FIRST, BEST, CUSTOM
- Error handling strategies: FAIL_FAST, CONTINUE, RETRY

**Step 3: Identify Common Patterns**
- Task execution: All workflows execute tasks
- Result collection: All workflows aggregate results
- Error handling: All workflows handle failures
- State management: All workflows track progress
- Timeout handling: All workflows respect time limits

**Step 4: Design Data Structures**
- `WorkflowTask`: Task definition with dependencies
- `WorkflowResult`: Result with status, output, error, duration
- `WorkflowState`: Current state with progress tracking
- `WorkflowConfig`: Configuration with timeout, error strategy, etc.

**Step 5: Plan Integration Points**
- Ephemeral agents: Use for task execution (create agent per task)
- File locking: Auto-lock files before task execution
- Task specification: Validate tasks before execution
- CrewAI: Optional integration for complex tasks

---

### Phase 2: Parallel Workflow Design (Steps 6-10)

**Step 6: Parallel Execution Strategy**
- Use `asyncio.gather()` for concurrent execution
- Limit concurrency with `asyncio.Semaphore` (default: 5 concurrent tasks)
- Return results in original task order (not completion order)
- Handle partial failures (some tasks succeed, some fail)

**Step 7: Result Aggregation**
- ALL: Wait for all tasks, return all results (fail if any fails)
- FIRST: Return first successful result, cancel others
- BEST: Wait for all, return best result (custom scoring function)
- CUSTOM: User-provided aggregation function

**Step 8: Error Handling**
- FAIL_FAST: Cancel all tasks on first failure
- CONTINUE: Continue execution, collect all results (success + failures)
- RETRY: Retry failed tasks up to N times (default: 3)

**Step 9: Timeout Handling**
- Global timeout: Cancel all tasks after timeout
- Per-task timeout: Cancel individual tasks after timeout
- Graceful cancellation: Allow cleanup before termination

**Step 10: Performance Optimization**
- Measure speedup: Compare parallel vs sequential execution
- Target: >1.5x speedup for 3+ tasks
- Monitor: Task start/end times, queue wait times
- Optimize: Adjust concurrency limit based on system resources

---

### Phase 3: Sequential Workflow Design (Steps 11-15)

**Step 11: Sequential Execution Strategy**
- Execute tasks in order using `for` loop with `await`
- Pass state between tasks (output of task N → input of task N+1)
- Support conditional execution (skip task if condition not met)
- Support early termination (stop if condition met)

**Step 12: State Passing**
- Accumulator pattern: Collect outputs in dict
- Pipeline pattern: Output of task N → input of task N+1
- Context pattern: Shared context accessible to all tasks
- Immutable state: Prevent accidental mutations

**Step 13: Dependency Management**
- Explicit dependencies: Task N depends on tasks [A, B, C]
- Implicit dependencies: Sequential order implies dependency
- Validation: Ensure dependencies are satisfied before execution
- Cycle detection: Prevent circular dependencies

**Step 14: Conditional Execution**
- Condition function: `(state) -> bool`
- Skip task if condition returns False
- Log skipped tasks for debugging
- Continue to next task

**Step 15: Early Termination**
- Termination condition: `(state) -> bool`
- Stop workflow if condition returns True
- Return partial results
- Mark workflow as COMPLETED (not FAILED)

---

### Phase 4: Evaluator-Optimizer Workflow Design (Steps 16-20)

**Step 16: Evaluation-Optimization Loop**
- Evaluator: Assess current solution quality (score 0.0-1.0)
- Optimizer: Improve solution based on evaluation feedback
- Convergence: Stop when score meets threshold or max iterations reached
- History: Track all iterations for analysis

**Step 17: Convergence Criteria**
- Score threshold: Stop when score >= threshold (e.g., 0.95)
- Max iterations: Stop after N iterations (default: 10)
- No improvement: Stop if score doesn't improve for K iterations (default: 3)
- Time limit: Stop after timeout

**Step 18: Feedback Loop**
- Evaluator output: Score + feedback (what to improve)
- Optimizer input: Current solution + feedback
- Optimizer output: Improved solution
- Iteration: Repeat until convergence

**Step 19: History Tracking**
- Store all iterations: solution, score, feedback, duration
- Analyze trends: Is score improving? Converging? Oscillating?
- Visualization: Plot score over iterations
- Debugging: Inspect specific iterations

**Step 20: Optimization Strategies**
- Gradient-based: Use feedback to guide improvements
- Random search: Try random variations
- Genetic algorithm: Evolve solutions over iterations
- Custom: User-provided optimization function

---

### Phase 5: Implementation (Steps 21-25)

**Step 21: Create Base Workflow Class**
- File: `handlers/workflows.py`
- Base class with common functionality
- Abstract methods: `execute()`, `validate()`
- Shared methods: `_handle_error()`, `_aggregate_results()`, `_check_timeout()`

**Step 22: Implement ParallelWorkflow**
- Inherit from base Workflow class
- Implement `execute()` with `asyncio.gather()`
- Add concurrency limiting with `asyncio.Semaphore`
- Add result aggregation strategies

**Step 23: Implement SequentialWorkflow**
- Inherit from base Workflow class
- Implement `execute()` with sequential loop
- Add state passing between tasks
- Add conditional execution and early termination

**Step 24: Implement EvaluatorOptimizerWorkflow**
- Inherit from base Workflow class
- Implement `execute()` with evaluation-optimization loop
- Add convergence criteria checking
- Add history tracking

**Step 25: Add MCP Tool Handlers**
- File: `handlers/agent_teams.py`
- `handle_execute_parallel_workflow()`
- `handle_execute_sequential_workflow()`
- `handle_execute_evaluator_optimizer_workflow()`
- Register in `server.py`

---

### Phase 6: Testing (Steps 26-28)

**Step 26: Unit Tests**
- File: `tests/test_workflows.py`
- Test each workflow pattern independently
- Test error handling strategies
- Test timeout behavior
- Test result aggregation
- Target: 15+ tests, 100% passing

**Step 27: Integration Tests**
- File: `test_workflows_integration.py`
- Test workflows with real tasks
- Test integration with ephemeral agents
- Test integration with file locking
- Test MCP tool handlers
- Target: 5+ tests, 100% passing

**Step 28: Performance Tests**
- Measure parallel workflow speedup
- Target: >1.5x speedup for 3+ tasks
- Measure overhead: workflow setup time
- Measure reliability: error handling effectiveness

---

### Phase 7: Documentation and Completion (Steps 29-30)

**Step 29: Documentation**
- Update `README.md` with workflow patterns
- Add usage examples for each workflow
- Document configuration options
- Add performance metrics

**Step 30: Completion Summary**
- Create `PHASE_2_PRIORITY_3_COMPLETE.md`
- Create `WHERE_TO_FIND_RESULTS_PHASE2_PRIORITY3.md`
- Update `ENHANCEMENT_SUMMARY.md`
- Commit all changes to version control

---

## 🎯 Success Criteria

- ✅ All workflow patterns implemented and tested
- ✅ Performance: Parallel workflow shows >1.5x speedup for 3+ tasks
- ✅ Error handling: 100% reliable (failed tasks don't crash workflow)
- ✅ All tests passing (unit + integration)
- ✅ Backward compatible
- ✅ Documentation complete with usage examples

---

## 📊 Estimated Timeline

- Architecture & Design: 1 hour
- Implementation: 3-4 hours
- Testing: 2 hours
- Documentation: 1 hour
- **Total**: 6-8 hours

---

## 🔧 Technical Decisions

### Concurrency Model
- Use `asyncio` for all async operations
- Limit concurrency with `asyncio.Semaphore` (default: 5)
- Use `asyncio.gather()` for parallel execution
- Use `asyncio.wait_for()` for timeout handling

### Error Handling
- Capture all exceptions in `WorkflowResult`
- Provide multiple error strategies (FAIL_FAST, CONTINUE, RETRY)
- Log all errors for debugging
- Return partial results when possible

### State Management
- Immutable state: Use frozen dataclasses
- State passing: Explicit input/output
- Context: Shared dict for cross-task communication
- History: Store all iterations for analysis

### Integration
- Ephemeral agents: Create agent per task (auto-cleanup)
- File locking: Auto-lock files before task execution
- Task specification: Validate tasks before execution
- CrewAI: Optional integration for complex tasks

---

**Implementation Start**: 2025-01-16  
**Target Completion**: 2025-01-16  
**Status**: READY TO IMPLEMENT

