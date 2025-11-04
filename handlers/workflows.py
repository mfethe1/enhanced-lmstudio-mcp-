"""
Composable Workflow Patterns for Jarvis MCP

Provides three workflow patterns:
1. ParallelWorkflow: Execute tasks concurrently with result aggregation
2. SequentialWorkflow: Execute tasks in order with state passing
3. EvaluatorOptimizerWorkflow: Iterative evaluation and optimization loop

Author: Jarvis MCP Team
Date: 2025-01-16
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


# ============================================================================
# Enums and Data Structures
# ============================================================================

class WorkflowState(Enum):
    """Workflow execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorStrategy(Enum):
    """Error handling strategies"""
    FAIL_FAST = "fail_fast"  # Cancel all tasks on first failure
    CONTINUE = "continue"    # Continue execution, collect all results
    RETRY = "retry"          # Retry failed tasks up to N times


class AggregationStrategy(Enum):
    """Result aggregation strategies"""
    ALL = "all"          # Wait for all tasks, return all results
    FIRST = "first"      # Return first successful result, cancel others
    BEST = "best"        # Wait for all, return best result
    CUSTOM = "custom"    # User-provided aggregation function


@dataclass
class WorkflowTask:
    """Task definition for workflow execution"""
    task_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    timeout: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict], bool]] = None
    files_to_lock: List[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result of a workflow task execution"""
    task_id: str
    status: str  # "success", "failed", "cancelled", "timeout"
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    timeout: Optional[int] = None
    error_strategy: ErrorStrategy = ErrorStrategy.CONTINUE
    aggregation_strategy: AggregationStrategy = AggregationStrategy.ALL
    max_retries: int = 3
    max_concurrency: int = 5
    aggregation_function: Optional[Callable] = None


# ============================================================================
# Base Workflow Class
# ============================================================================

class Workflow(ABC):
    """Base class for all workflow patterns"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        self.state = WorkflowState.PENDING
        self.results: List[WorkflowResult] = []
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def execute(self, tasks: List[WorkflowTask]) -> List[WorkflowResult]:
        """Execute workflow tasks. Must be implemented by subclasses."""
        pass
    
    def validate(self, tasks: List[WorkflowTask]) -> bool:
        """Validate tasks before execution"""
        if not tasks:
            raise ValueError("No tasks provided")
        
        # Check for duplicate task IDs
        task_ids = [t.task_id for t in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs found")
        
        # Check dependencies exist
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(f"Task {task.task_id} depends on non-existent task {dep_id}")
        
        return True
    
    async def _execute_task(self, task: WorkflowTask, context: Dict[str, Any]) -> WorkflowResult:
        """Execute a single task with error handling and timeout"""
        result = WorkflowResult(
            task_id=task.task_id,
            status="pending",
            started_at=time.time()
        )
        
        try:
            # Check condition if provided
            if task.condition and not task.condition(context):
                result.status = "skipped"
                result.completed_at = time.time()
                result.duration = result.completed_at - result.started_at
                return result
            
            # Acquire file locks if needed
            lock_ids = []
            if task.files_to_lock:
                from core.file_locking import get_file_lock_manager
                manager = get_file_lock_manager()
                await manager.start()
                
                for file_path in sorted(task.files_to_lock):  # Alphabetical for deadlock prevention
                    lock_id = await manager.acquire_lock(
                        file_path=file_path,
                        owner_id=self.workflow_id,
                        timeout_seconds=task.timeout or 60
                    )
                    lock_ids.append((file_path, lock_id))
            
            try:
                # Execute task with timeout
                if task.timeout:
                    output = await asyncio.wait_for(
                        task.function(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    output = await task.function(*task.args, **task.kwargs)
                
                result.status = "success"
                result.output = output
                
            finally:
                # Release file locks
                if lock_ids:
                    from core.file_locking import get_file_lock_manager
                    manager = get_file_lock_manager()
                    for file_path, lock_id in lock_ids:
                        await manager.release_lock(
                            file_path=file_path,
                            owner_id=self.workflow_id
                        )
        
        except asyncio.TimeoutError:
            result.status = "timeout"
            result.error = f"Task timed out after {task.timeout}s"
        
        except asyncio.CancelledError:
            result.status = "cancelled"
            result.error = "Task was cancelled"
            raise  # Re-raise to propagate cancellation
        
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
        
        result.completed_at = time.time()
        result.duration = result.completed_at - result.started_at
        
        return result
    
    async def _handle_error(self, result: WorkflowResult, task: WorkflowTask) -> Optional[WorkflowResult]:
        """Handle task failure based on error strategy"""
        if self.config.error_strategy == ErrorStrategy.RETRY and result.retry_count < self.config.max_retries:
            # Retry the task
            result.retry_count += 1
            return await self._execute_task(task, {})

        # FAIL_FAST and CONTINUE: just return the failed result
        # The caller will decide whether to stop execution
        return result
    
    def _aggregate_results(self, results: List[WorkflowResult]) -> List[WorkflowResult]:
        """Aggregate results based on aggregation strategy"""
        if self.config.aggregation_strategy == AggregationStrategy.ALL:
            return results
        
        elif self.config.aggregation_strategy == AggregationStrategy.FIRST:
            # Return first successful result
            for result in results:
                if result.status == "success":
                    return [result]
            return results  # No successful results
        
        elif self.config.aggregation_strategy == AggregationStrategy.BEST:
            # Return best result (highest output value)
            successful = [r for r in results if r.status == "success"]
            if not successful:
                return results
            best = max(successful, key=lambda r: r.output if isinstance(r.output, (int, float)) else 0)
            return [best]
        
        elif self.config.aggregation_strategy == AggregationStrategy.CUSTOM:
            if self.config.aggregation_function:
                return self.config.aggregation_function(results)
            return results
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        duration = 0.0
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
        
        successful = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        cancelled = sum(1 for r in self.results if r.status == "cancelled")
        timeout = sum(1 for r in self.results if r.status == "timeout")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "total_tasks": len(self.results),
            "successful": successful,
            "failed": failed,
            "cancelled": cancelled,
            "timeout": timeout,
            "skipped": skipped,
            "duration": duration,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


# ============================================================================
# Parallel Workflow
# ============================================================================

class ParallelWorkflow(Workflow):
    """Execute tasks concurrently with result aggregation"""

    async def execute(self, tasks: List[WorkflowTask]) -> List[WorkflowResult]:
        """Execute tasks in parallel with concurrency limiting"""
        self.validate(tasks)

        async with self._lock:
            self.state = WorkflowState.RUNNING
            self.started_at = time.time()

        try:
            # Create semaphore for concurrency limiting
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            # For FIRST strategy, we need to cancel other tasks after first success
            if self.config.aggregation_strategy == AggregationStrategy.FIRST:
                first_success = asyncio.Event()
                results = []

                async def execute_with_first_strategy(task: WorkflowTask) -> WorkflowResult:
                    async with semaphore:
                        # Check if we already have a success
                        if first_success.is_set():
                            return WorkflowResult(
                                task_id=task.task_id,
                                status="cancelled",
                                error="Cancelled after first success",
                                started_at=time.time(),
                                completed_at=time.time()
                            )

                        result = await self._execute_task(task, {})

                        # If this is the first success, signal others to stop
                        if result.status == "success" and not first_success.is_set():
                            first_success.set()

                        return result

                # Execute all tasks
                if self.config.timeout:
                    results = await asyncio.wait_for(
                        asyncio.gather(*[execute_with_first_strategy(task) for task in tasks], return_exceptions=False),
                        timeout=self.config.timeout
                    )
                else:
                    results = await asyncio.gather(*[execute_with_first_strategy(task) for task in tasks], return_exceptions=False)

                # Return only first successful result
                for result in results:
                    if result.status == "success":
                        self.results = [result]
                        break
                else:
                    self.results = results  # No successful results

            else:
                # Standard parallel execution
                async def execute_with_semaphore(task: WorkflowTask) -> WorkflowResult:
                    async with semaphore:
                        result = await self._execute_task(task, {})

                        # Handle errors based on strategy
                        if result.status == "failed":
                            retry_result = await self._handle_error(result, task)
                            if retry_result:
                                result = retry_result

                        return result

                # Execute all tasks in parallel
                if self.config.timeout:
                    results = await asyncio.wait_for(
                        asyncio.gather(*[execute_with_semaphore(task) for task in tasks], return_exceptions=False),
                        timeout=self.config.timeout
                    )
                else:
                    results = await asyncio.gather(*[execute_with_semaphore(task) for task in tasks], return_exceptions=False)

                # Aggregate results
                self.results = self._aggregate_results(results)

            # Check if any task failed
            if any(r.status == "failed" for r in self.results):
                self.state = WorkflowState.FAILED
            else:
                self.state = WorkflowState.COMPLETED

        except asyncio.TimeoutError:
            self.state = WorkflowState.FAILED
            raise

        except asyncio.CancelledError:
            self.state = WorkflowState.CANCELLED
            raise

        except Exception as e:
            self.state = WorkflowState.FAILED
            raise

        finally:
            self.completed_at = time.time()

        return self.results


# ============================================================================
# Sequential Workflow
# ============================================================================

class SequentialWorkflow(Workflow):
    """Execute tasks in order with state passing"""

    async def execute(self, tasks: List[WorkflowTask]) -> List[WorkflowResult]:
        """Execute tasks sequentially with state passing"""
        self.validate(tasks)

        async with self._lock:
            self.state = WorkflowState.RUNNING
            self.started_at = time.time()

        try:
            context = {}  # Shared context for state passing
            results = []

            for task in tasks:
                # Check dependencies
                for dep_id in task.dependencies:
                    dep_result = next((r for r in results if r.task_id == dep_id), None)
                    if not dep_result or dep_result.status != "success":
                        # Dependency failed, skip this task
                        result = WorkflowResult(
                            task_id=task.task_id,
                            status="skipped",
                            error=f"Dependency {dep_id} not satisfied",
                            started_at=time.time(),
                            completed_at=time.time()
                        )
                        results.append(result)
                        continue

                # Execute task
                result = await self._execute_task(task, context)

                # Handle errors based on strategy
                if result.status == "failed":
                    retry_result = await self._handle_error(result, task)
                    if retry_result:
                        result = retry_result

                results.append(result)

                # Update context with task output
                if result.status == "success":
                    context[task.task_id] = result.output

                # Check for early termination
                if result.status == "failed" and self.config.error_strategy == ErrorStrategy.FAIL_FAST:
                    break

            # Aggregate results
            self.results = self._aggregate_results(results)

            # Check if any task failed
            if any(r.status == "failed" for r in self.results):
                self.state = WorkflowState.FAILED
            else:
                self.state = WorkflowState.COMPLETED

        except asyncio.CancelledError:
            self.state = WorkflowState.CANCELLED
            raise

        except Exception as e:
            self.state = WorkflowState.FAILED
            raise

        finally:
            self.completed_at = time.time()

        return self.results


# ============================================================================
# Evaluator-Optimizer Workflow
# ============================================================================

@dataclass
class EvaluationIteration:
    """Single iteration of evaluation-optimization loop"""
    iteration: int
    solution: Any
    score: float
    feedback: str
    duration: float
    timestamp: float


class EvaluatorOptimizerWorkflow(Workflow):
    """Iterative evaluation and optimization loop"""

    def __init__(
        self,
        evaluator: Callable[[Any], tuple[float, str]],
        optimizer: Callable[[Any, str], Any],
        initial_solution: Any,
        score_threshold: float = 0.95,
        max_iterations: int = 10,
        no_improvement_limit: int = 3,
        config: Optional[WorkflowConfig] = None
    ):
        super().__init__(config)
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.initial_solution = initial_solution
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations
        self.no_improvement_limit = no_improvement_limit
        self.history: List[EvaluationIteration] = []

    async def execute(self, tasks: List[WorkflowTask] = None) -> List[WorkflowResult]:
        """Execute evaluation-optimization loop until convergence"""
        async with self._lock:
            self.state = WorkflowState.RUNNING
            self.started_at = time.time()

        try:
            solution = self.initial_solution
            best_score = 0.0
            no_improvement_count = 0

            for iteration in range(self.max_iterations):
                iteration_start = time.time()

                # Evaluate current solution
                if asyncio.iscoroutinefunction(self.evaluator):
                    score, feedback = await self.evaluator(solution)
                else:
                    score, feedback = self.evaluator(solution)

                # Record iteration
                self.history.append(EvaluationIteration(
                    iteration=iteration,
                    solution=solution,
                    score=score,
                    feedback=feedback,
                    duration=time.time() - iteration_start,
                    timestamp=time.time()
                ))

                # Check convergence
                if score >= self.score_threshold:
                    # Converged!
                    self.state = WorkflowState.COMPLETED
                    break

                # Check for improvement
                if score <= best_score:
                    no_improvement_count += 1
                    if no_improvement_count >= self.no_improvement_limit:
                        # No improvement for too long, stop
                        self.state = WorkflowState.COMPLETED
                        break
                else:
                    best_score = score
                    no_improvement_count = 0

                # Check if we've reached max iterations (before optimizing)
                if iteration >= self.max_iterations - 1:
                    # Reached max iterations
                    self.state = WorkflowState.COMPLETED
                    break

                # Optimize solution
                if asyncio.iscoroutinefunction(self.optimizer):
                    solution = await self.optimizer(solution, feedback)
                else:
                    solution = self.optimizer(solution, feedback)

            # Create final result
            final_iteration = self.history[-1] if self.history else None
            result = WorkflowResult(
                task_id="evaluator_optimizer",
                status="success" if final_iteration and final_iteration.score >= self.score_threshold else "completed",
                output={
                    "solution": solution,
                    "score": final_iteration.score if final_iteration else 0.0,
                    "iterations": len(self.history),
                    "converged": final_iteration.score >= self.score_threshold if final_iteration else False
                },
                started_at=self.started_at,
                completed_at=time.time(),
                duration=time.time() - self.started_at
            )

            self.results = [result]

            if self.state != WorkflowState.COMPLETED:
                self.state = WorkflowState.COMPLETED

        except asyncio.CancelledError:
            self.state = WorkflowState.CANCELLED
            raise

        except Exception as e:
            self.state = WorkflowState.FAILED
            result = WorkflowResult(
                task_id="evaluator_optimizer",
                status="failed",
                error=str(e),
                started_at=self.started_at,
                completed_at=time.time(),
                duration=time.time() - self.started_at
            )
            self.results = [result]
            raise

        finally:
            self.completed_at = time.time()

        return self.results

    def get_history(self) -> List[Dict[str, Any]]:
        """Get iteration history"""
        return [
            {
                "iteration": h.iteration,
                "score": h.score,
                "feedback": h.feedback,
                "duration": h.duration,
                "timestamp": h.timestamp
            }
            for h in self.history
        ]

