"""
Agentic Task Management System for Jarvis MCP Server

This module provides background task execution, persistent state management,
and progress tracking to transform the MCP server into a true agentic platform.

Key Features:
- Background task execution with unique job IDs
- Persistent task state storage (JSON-based)
- Progress tracking and activity logging
- Task lifecycle management (pending, running, completed, failed, cancelled)
- User check-in capabilities for autonomous workflows
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import threading
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskActivity:
    """Single activity log entry"""
    timestamp: str
    message: str
    level: str = "info"  # info, warning, error
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    """Background task representation"""
    task_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: str = "0%"
    result: Optional[Any] = None
    error: Optional[str] = None
    activity_log: List[TaskActivity] = None
    
    def __post_init__(self):
        if self.activity_log is None:
            self.activity_log = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['activity_log'] = [activity.to_dict() for activity in self.activity_log]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create Task from dictionary"""
        # Convert status back to enum
        data['status'] = TaskStatus(data['status'])
        
        # Convert activity log back to objects
        activity_data = data.get('activity_log', [])
        data['activity_log'] = [TaskActivity(**activity) for activity in activity_data]
        
        return cls(**data)
    
    def add_activity(self, message: str, level: str = "info"):
        """Add activity log entry"""
        activity = TaskActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=message,
            level=level
        )
        self.activity_log.append(activity)
        logger.info(f"Task {self.task_id}: {message}")


class TaskManager:
    """Manages background task execution and persistence"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize TaskManager with persistent storage"""
        self.storage_path = storage_path or self._get_default_storage_path()
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # Use re-entrant lock to avoid deadlocks when _save_tasks is called within other locked sections
        self._lock = threading.RLock()
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing tasks
        self._load_tasks()
        
        logger.info(f"TaskManager initialized with storage: {self.storage_path}")
    
    def _get_default_storage_path(self) -> str:
        """Get platform-specific default storage path"""
        if os.name == 'nt':  # Windows
            base_dir = os.path.expandvars(r'%APPDATA%\jarvis-mcp')
        elif os.uname().sysname == 'Darwin':  # macOS
            base_dir = os.path.expanduser('~/Library/Application Support/jarvis-mcp')
        else:  # Linux and others
            base_dir = os.path.expanduser('~/.local/share/jarvis-mcp')
        
        return os.path.join(base_dir, 'tasks.json')
    
    def _load_tasks(self):
        """Load tasks from persistent storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for task_id, task_data in data.get('tasks', {}).items():
                    try:
                        task = Task.from_dict(task_data)
                        self.tasks[task_id] = task
                        
                        # Reset running tasks to pending on startup
                        if task.status == TaskStatus.RUNNING:
                            task.status = TaskStatus.PENDING
                            task.add_activity("Task reset to pending after server restart", "warning")
                            
                    except Exception as e:
                        logger.error(f"Failed to load task {task_id}: {e}")
                        
                logger.info(f"Loaded {len(self.tasks)} tasks from storage")
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
    
    def _save_tasks(self):
        """Save tasks to persistent storage (thread-safe with retries for Windows)."""
        data = {
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        attempts = 0
        last_err = None
        while attempts < 5:
            attempts += 1
            try:
                with self._lock:
                    # Use a unique temp file per attempt to avoid AV/lock issues
                    temp_path = self.storage_path + f'.tmp.{os.getpid()}.{threading.get_ident()}.{attempts}'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    # Atomic replace
                    os.replace(temp_path, self.storage_path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.05 * attempts)  # backoff
            except Exception as e:
                last_err = e
                break
        if last_err:
            logger.error(f"Failed to save tasks after {attempts} attempts: {last_err}")
    
    def create_task(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create a new background task"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            tool_name=tool_name,
            arguments=arguments,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        task.add_activity(f"Task created for tool '{tool_name}'")
        
        with self._lock:
            self.tasks[task_id] = task
            self._save_tasks()
        
        logger.info(f"Created task {task_id} for tool {tool_name}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status_filter: Optional[TaskStatus] = None, limit: int = 50) -> List[Task]:
        """List tasks with optional status filtering"""
        tasks = list(self.tasks.values())
        
        if status_filter:
            tasks = [task for task in tasks if task.status == status_filter]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    async def execute_task_async(self, task_id: str, handler_func: Callable, server_instance: Any):
        """Execute task asynchronously in background"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            task.add_activity("Task execution started")
            self._save_tasks()
            
            # Execute the handler function
            if asyncio.iscoroutinefunction(handler_func):
                result = await handler_func(task.arguments, server_instance)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler_func, task.arguments, server_instance)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.progress = "100%"
            task.result = result
            task.add_activity("Task completed successfully")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.error = str(e)
            task.add_activity(f"Task failed: {str(e)}", "error")
            logger.error(f"Task {task_id} failed: {e}")
            
        finally:
            # Clean up running task reference
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self._save_tasks()
    
    def start_task(self, task_id: str, handler_func: Callable, server_instance: Any) -> bool:
        """Start task execution in background"""
        if task_id in self.running_tasks:
            logger.warning(f"Task {task_id} is already running")
            return False

        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False

        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not in pending status")
            return False

        # Handle async task creation with proper event loop management
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule the task
                async_task = asyncio.create_task(
                    self.execute_task_async(task_id, handler_func, server_instance)
                )
                self.running_tasks[task_id] = async_task
            else:
                # If no loop is running, start task in thread
                self._start_task_in_thread(task_id, handler_func, server_instance)
        except RuntimeError:
            # No event loop exists, start task in thread
            self._start_task_in_thread(task_id, handler_func, server_instance)

        logger.info(f"Started background execution for task {task_id}")
        return True

    def _start_task_in_thread(self, task_id: str, handler_func: Callable, server_instance: Any):
        """Start task execution in a separate thread with its own event loop"""
        def run_task():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Check if handler is async or sync
                if asyncio.iscoroutinefunction(handler_func):
                    # Run the async task
                    loop.run_until_complete(
                        self.execute_task_async(task_id, handler_func, server_instance)
                    )
                else:
                    # Run sync handler in executor
                    loop.run_until_complete(
                        self._execute_sync_task_async(task_id, handler_func, server_instance)
                    )
            except Exception as e:
                logger.error(f"Thread task execution failed for {task_id}: {e}")
                # Update task status to failed
                task = self.get_task(task_id)
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    task.add_activity(f"Task failed: {str(e)}", "error")
                    self._save_tasks()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass  # Ignore close errors

        # Start thread
        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

        # Store thread reference instead of async task
        self.running_tasks[task_id] = thread

    async def _execute_sync_task_async(self, task_id: str, handler_func: Callable, server_instance: Any):
        """Execute a synchronous handler function in an async context"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            task.add_activity("Task execution started", "info")
            self._save_tasks()

            # Execute the sync handler directly in this thread's context
            # This allows the handler to create/own its own event loop if needed
            result = handler_func(task.arguments, server_instance)

            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = "100%"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.add_activity("Task completed successfully", "info")
            self._save_tasks()

            logger.info(f"Task {task_id}: Task completed successfully")

        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.add_activity(f"Task failed: {str(e)}", "error")
            self._save_tasks()

            logger.error(f"Task {task_id} failed: {e}")
        finally:
            # Clean up running task reference
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            running_item = self.running_tasks[task_id]

            # Handle both async tasks and threads
            if hasattr(running_item, 'cancel'):
                # It's an async task
                running_item.cancel()
            elif hasattr(running_item, 'is_alive'):
                # It's a thread - we can't directly cancel threads in Python
                # but we can mark the task as cancelled
                logger.warning(f"Cannot directly cancel thread for task {task_id}, marking as cancelled")

            del self.running_tasks[task_id]

            task = self.get_task(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.add_activity("Task cancelled by user", "warning")
                self._save_tasks()

            logger.info(f"Cancelled task {task_id}")
            return True

        return False
    
    def get_task_status_summary(self) -> Dict[str, Any]:
        """Get summary of all task statuses"""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([t for t in self.tasks.values() if t.status == status])
        
        return {
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'status_breakdown': status_counts,
            'storage_path': self.storage_path
        }
