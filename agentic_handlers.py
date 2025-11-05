"""
Agentic Tool Handlers for Jarvis MCP Server

This module provides the new agentic tool handlers that enable background task execution
and autonomous agent workflows. Users can start tasks and check back later for results.

Key Features:
- Background task execution for long-running operations
- Task status monitoring and progress tracking
- Autonomous agent workflows with persistent state
- User check-in capabilities for task management
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from task_manager import TaskManager, TaskStatus

logger = logging.getLogger(__name__)


class AgenticToolHandlers:
    """Collection of agentic tool handlers for background task execution"""
    
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        
        # Map of tool names to their original handler functions
        self.tool_handlers = {}
        
    def register_tool_handler(self, tool_name: str, handler_func):
        """Register a tool handler for background execution"""
        self.tool_handlers[tool_name] = handler_func
        logger.info(f"Registered agentic handler for tool: {tool_name}")
    
    def handle_start_agentic_task(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """Start any tool as a background agentic task
        
        Arguments:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments to pass to the tool
            description: Optional description of what the task will do
        
        Returns:
            task_id: Unique identifier for tracking the task
            status: Initial task status
            message: Human-readable status message
        """
        tool_name = arguments.get("tool_name", "").strip()
        tool_arguments = arguments.get("tool_arguments", {})
        description = arguments.get("description", "").strip()
        
        if not tool_name:
            return {
                "error": "tool_name is required",
                "status": "failed"
            }
        
        # Check if tool handler is registered
        if tool_name not in self.tool_handlers:
            return {
                "error": f"Tool '{tool_name}' is not available for agentic execution",
                "status": "failed",
                "available_tools": list(self.tool_handlers.keys())
            }
        
        try:
            # Create the background task
            task_id = self.task_manager.create_task(tool_name, tool_arguments)
            
            # Start background execution
            handler_func = self.tool_handlers[tool_name]
            success = self.task_manager.start_task(task_id, handler_func, server)
            
            if not success:
                return {
                    "error": "Failed to start background task execution",
                    "task_id": task_id,
                    "status": "failed"
                }
            
            return {
                "task_id": task_id,
                "tool_name": tool_name,
                "status": "started",
                "message": f"Background task started for '{tool_name}'. Use get_task_status to monitor progress.",
                "description": description or f"Executing {tool_name} in background",
                "check_status_with": f"get_task_status(task_id='{task_id}')"
            }
            
        except Exception as e:
            logger.error(f"Failed to start agentic task for {tool_name}: {e}")
            return {
                "error": f"Failed to start agentic task: {str(e)}",
                "status": "failed"
            }
    
    def handle_get_task_status(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """Get the current status and progress of a background task
        
        Arguments:
            task_id: Unique task identifier
            include_log: Whether to include activity log (default: False)
        
        Returns:
            Task status, progress, and optionally activity log
        """
        task_id = arguments.get("task_id", "").strip()
        include_log = arguments.get("include_log", False)
        
        if not task_id:
            return {
                "error": "task_id is required",
                "status": "failed"
            }
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return {
                "error": f"Task '{task_id}' not found",
                "status": "not_found"
            }
        
        response = {
            "task_id": task.task_id,
            "tool_name": task.tool_name,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at
        }
        
        # Include result if completed
        if task.status == TaskStatus.COMPLETED and task.result is not None:
            response["result"] = task.result
            response["message"] = "Task completed successfully. Result available."
        
        # Include error if failed
        if task.status == TaskStatus.FAILED and task.error:
            response["error"] = task.error
            response["message"] = f"Task failed: {task.error}"
        
        # Include activity log if requested
        if include_log:
            response["activity_log"] = [activity.to_dict() for activity in task.activity_log]
        
        # Add helpful messages based on status
        if task.status == TaskStatus.PENDING:
            response["message"] = "Task is queued and waiting to start."
        elif task.status == TaskStatus.RUNNING:
            response["message"] = f"Task is currently running. Progress: {task.progress}"
        elif task.status == TaskStatus.CANCELLED:
            response["message"] = "Task was cancelled."
        
        return response
    
    def handle_list_all_tasks(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """List all tasks with optional filtering
        
        Arguments:
            status_filter: Optional status to filter by (pending, running, completed, failed, cancelled)
            limit: Maximum number of tasks to return (default: 50)
            include_results: Whether to include task results (default: False)
        
        Returns:
            List of tasks with their status and metadata
        """
        status_filter_str = arguments.get("status_filter", "").strip().lower()
        limit = int(arguments.get("limit", 50))
        include_results = arguments.get("include_results", False)
        
        # Convert status filter string to enum
        status_filter = None
        if status_filter_str:
            try:
                status_filter = TaskStatus(status_filter_str)
            except ValueError:
                return {
                    "error": f"Invalid status filter: {status_filter_str}",
                    "valid_statuses": [status.value for status in TaskStatus]
                }
        
        tasks = self.task_manager.list_tasks(status_filter, limit)
        
        task_list = []
        for task in tasks:
            task_info = {
                "task_id": task.task_id,
                "tool_name": task.tool_name,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            }
            
            # Include results if requested and available
            if include_results:
                if task.status == TaskStatus.COMPLETED and task.result is not None:
                    task_info["result"] = task.result
                if task.status == TaskStatus.FAILED and task.error:
                    task_info["error"] = task.error
            
            task_list.append(task_info)
        
        # Get summary statistics
        summary = self.task_manager.get_task_status_summary()
        
        return {
            "tasks": task_list,
            "total_found": len(task_list),
            "summary": summary,
            "filter_applied": status_filter.value if status_filter else "none"
        }
    
    def handle_get_task_results(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """Get the complete results of a completed task
        
        Arguments:
            task_id: Unique task identifier
        
        Returns:
            Complete task results and metadata
        """
        task_id = arguments.get("task_id", "").strip()
        
        if not task_id:
            return {
                "error": "task_id is required",
                "status": "failed"
            }
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return {
                "error": f"Task '{task_id}' not found",
                "status": "not_found"
            }
        
        if task.status != TaskStatus.COMPLETED:
            return {
                "error": f"Task is not completed. Current status: {task.status.value}",
                "status": task.status.value,
                "message": "Task must be completed to retrieve results"
            }
        
        return {
            "task_id": task.task_id,
            "tool_name": task.tool_name,
            "status": task.status.value,
            "result": task.result,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "execution_time": self._calculate_execution_time(task),
            "activity_log": [activity.to_dict() for activity in task.activity_log]
        }
    
    def handle_cancel_task(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """Cancel a running background task
        
        Arguments:
            task_id: Unique task identifier
        
        Returns:
            Cancellation status and message
        """
        task_id = arguments.get("task_id", "").strip()
        
        if not task_id:
            return {
                "error": "task_id is required",
                "status": "failed"
            }
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return {
                "error": f"Task '{task_id}' not found",
                "status": "not_found"
            }
        
        if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            return {
                "error": f"Cannot cancel task with status: {task.status.value}",
                "status": task.status.value,
                "message": "Only pending or running tasks can be cancelled"
            }
        
        success = self.task_manager.cancel_task(task_id)
        
        if success:
            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": "Task cancelled successfully"
            }
        else:
            return {
                "error": "Failed to cancel task",
                "task_id": task_id,
                "status": "failed"
            }
    
    def handle_get_task_activity_log(self, arguments: Dict[str, Any], server) -> Dict[str, Any]:
        """Get detailed activity log for a task
        
        Arguments:
            task_id: Unique task identifier
            limit: Maximum number of log entries to return (default: 100)
        
        Returns:
            Detailed activity log with timestamps and messages
        """
        task_id = arguments.get("task_id", "").strip()
        limit = int(arguments.get("limit", 100))
        
        if not task_id:
            return {
                "error": "task_id is required",
                "status": "failed"
            }
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return {
                "error": f"Task '{task_id}' not found",
                "status": "not_found"
            }
        
        # Get recent activity log entries
        activity_log = task.activity_log[-limit:] if limit > 0 else task.activity_log
        
        return {
            "task_id": task.task_id,
            "tool_name": task.tool_name,
            "status": task.status.value,
            "activity_log": [activity.to_dict() for activity in activity_log],
            "total_entries": len(task.activity_log),
            "showing_entries": len(activity_log)
        }
    
    def _calculate_execution_time(self, task) -> Optional[str]:
        """Calculate task execution time"""
        if not task.started_at or not task.completed_at:
            return None
        
        try:
            from datetime import datetime
            start = datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(task.completed_at.replace('Z', '+00:00'))
            duration = end - start
            
            total_seconds = int(duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
                
        except Exception:
            return None
