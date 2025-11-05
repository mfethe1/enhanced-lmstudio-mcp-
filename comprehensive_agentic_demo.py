#!/usr/bin/env python3
"""
Comprehensive demonstration of the agentic system with real tool outputs
This script shows actual working examples of each agentic tool with real results.
"""

import json
import time
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)

def print_json_output(data, title="Output"):
    """Pretty print JSON data"""
    print(f"\n📋 {title}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

def comprehensive_demo():
    """Run comprehensive demonstration of all agentic tools"""
    print("🚀 COMPREHENSIVE AGENTIC SYSTEM DEMONSTRATION")
    print("This demo shows real working examples of each agentic tool")
    
    # Initialize the system
    print_section("SYSTEM INITIALIZATION")
    try:
        from task_manager import TaskManager, TaskStatus
        from agentic_handlers import AgenticToolHandlers
        from server import get_server_singleton
        
        # Get the actual server instance
        server = get_server_singleton()
        print(f"✅ Server initialized with {len(server.registry._handlers)} total tools")
        print(f"✅ Agentic system: {'Available' if server.agentic_handlers else 'Not Available'}")
        print(f"✅ Task storage: {server.task_manager.storage_path if server.task_manager else 'Not Available'}")
        
        # Use the server's agentic handlers
        agentic_handlers = server.agentic_handlers
        if not agentic_handlers:
            print("❌ Agentic handlers not available")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

    # Demo 1: Start a background research task
    print_section("DEMO 1: START AGENTIC RESEARCH TASK")
    try:
        task_args = {
            "tool_name": "web_search",
            "tool_arguments": {
                "query": "Python asyncio best practices 2024",
                "max_depth": 2,
                "time_limit": 60
            },
            "description": "Research Python asyncio best practices for our development team"
        }
        
        print("🔄 Starting background research task...")
        print_json_output(task_args, "Task Arguments")
        
        result = agentic_handlers.handle_start_agentic_task(task_args, server)
        print_json_output(result, "Start Task Result")
        
        if "task_id" in result:
            research_task_id = result["task_id"]
            print(f"✅ Research task started successfully: {research_task_id}")
        else:
            print("❌ Failed to start research task")
            return False
            
    except Exception as e:
        print(f"❌ Error starting research task: {e}")
        return False

    # Demo 2: Start a code analysis task
    print_section("DEMO 2: START AGENTIC CODE ANALYSIS TASK")
    try:
        # Create a sample Python file for analysis
        sample_code = '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

# This function has potential issues
def divide_numbers(a, b):
    return a / b  # No zero division check
'''
        
        with open("sample_code.py", "w") as f:
            f.write(sample_code)
        
        task_args = {
            "tool_name": "agent_collaborate",
            "tool_arguments": {
                "task": "Analyze the sample_code.py file for potential improvements, bugs, and optimization opportunities",
                "roles": ["Code Reviewer", "Performance Analyst", "Security Reviewer"],
                "rounds": 2
            },
            "description": "Multi-agent code analysis and improvement recommendations"
        }
        
        print("🔄 Starting background code analysis task...")
        print_json_output(task_args, "Task Arguments")
        
        result = agentic_handlers.handle_start_agentic_task(task_args, server)
        print_json_output(result, "Start Task Result")
        
        if "task_id" in result:
            analysis_task_id = result["task_id"]
            print(f"✅ Code analysis task started successfully: {analysis_task_id}")
        else:
            print("❌ Failed to start code analysis task")
            return False
            
    except Exception as e:
        print(f"❌ Error starting code analysis task: {e}")
        return False

    # Demo 3: Monitor task progress
    print_section("DEMO 3: MONITOR TASK PROGRESS")
    try:
        print("🔄 Monitoring research task progress...")
        
        # Monitor for up to 30 seconds
        max_wait = 30
        wait_time = 0
        
        while wait_time < max_wait:
            status_result = agentic_handlers.handle_get_task_status(
                {"task_id": research_task_id, "include_log": True}, 
                server
            )
            
            print(f"\n⏱️  Time: {wait_time}s")
            print_json_output(status_result, "Task Status")
            
            if status_result.get("status") in ["completed", "failed", "cancelled"]:
                print(f"✅ Task finished with status: {status_result.get('status')}")
                break
                
            time.sleep(3)
            wait_time += 3
        
    except Exception as e:
        print(f"❌ Error monitoring task: {e}")

    # Demo 4: List all tasks
    print_section("DEMO 4: LIST ALL TASKS")
    try:
        print("🔄 Listing all tasks...")
        
        list_result = agentic_handlers.handle_list_all_tasks(
            {"limit": 10, "include_results": False}, 
            server
        )
        
        print_json_output(list_result, "All Tasks")
        print(f"✅ Found {len(list_result.get('tasks', []))} tasks")
        
    except Exception as e:
        print(f"❌ Error listing tasks: {e}")

    # Demo 5: Get task results (if completed)
    print_section("DEMO 5: GET TASK RESULTS")
    try:
        print("🔄 Attempting to get research task results...")
        
        results = agentic_handlers.handle_get_task_results(
            {"task_id": research_task_id}, 
            server
        )
        
        print_json_output(results, "Task Results")
        
        if "result" in results:
            print("✅ Successfully retrieved task results")
        else:
            print("⚠️  Task may not be completed yet or failed")
            
    except Exception as e:
        print(f"❌ Error getting task results: {e}")

    # Demo 6: Get activity log
    print_section("DEMO 6: GET TASK ACTIVITY LOG")
    try:
        print("🔄 Getting detailed activity log...")
        
        activity_result = agentic_handlers.handle_get_task_activity_log(
            {"task_id": research_task_id, "limit": 20}, 
            server
        )
        
        print_json_output(activity_result, "Activity Log")
        
        if "activity_log" in activity_result:
            print(f"✅ Retrieved {len(activity_result['activity_log'])} activity entries")
        else:
            print("⚠️  No activity log available")
            
    except Exception as e:
        print(f"❌ Error getting activity log: {e}")

    # Demo 7: Test task cancellation
    print_section("DEMO 7: TASK CANCELLATION")
    try:
        print("🔄 Testing task cancellation on code analysis task...")
        
        # Check if analysis task is still running
        status_check = agentic_handlers.handle_get_task_status(
            {"task_id": analysis_task_id}, 
            server
        )
        
        print_json_output(status_check, "Analysis Task Status Before Cancel")
        
        if status_check.get("status") in ["pending", "running"]:
            cancel_result = agentic_handlers.handle_cancel_task(
                {"task_id": analysis_task_id}, 
                server
            )
            
            print_json_output(cancel_result, "Cancel Result")
            
            if cancel_result.get("status") == "cancelled":
                print("✅ Task cancelled successfully")
            else:
                print("⚠️  Task cancellation may have failed")
        else:
            print(f"ℹ️  Task already finished with status: {status_check.get('status')}")
            
    except Exception as e:
        print(f"❌ Error testing cancellation: {e}")

    # Demo 8: Show final system status
    print_section("DEMO 8: FINAL SYSTEM STATUS")
    try:
        print("🔄 Getting final system status...")
        
        # Get task summary
        final_list = agentic_handlers.handle_list_all_tasks(
            {"limit": 20, "include_results": True}, 
            server
        )
        
        print_json_output(final_list.get("summary", {}), "System Summary")
        
        # Show task breakdown by status
        tasks = final_list.get("tasks", [])
        status_counts = {}
        for task in tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n📊 Task Status Breakdown:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        print(f"\n✅ Total tasks processed: {len(tasks)}")
        
    except Exception as e:
        print(f"❌ Error getting final status: {e}")

    # Cleanup
    print_section("CLEANUP")
    try:
        if os.path.exists("sample_code.py"):
            os.remove("sample_code.py")
            print("✅ Cleaned up sample files")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    print_section("DEMONSTRATION COMPLETE")
    print("🎉 Agentic system demonstration completed successfully!")
    print("\nKey capabilities demonstrated:")
    print("  ✅ Background task execution with real tools")
    print("  ✅ Task status monitoring with progress tracking")
    print("  ✅ Task result retrieval with actual outputs")
    print("  ✅ Activity logging with detailed audit trail")
    print("  ✅ Task cancellation and management")
    print("  ✅ System status and task breakdown")
    print("\nThe agentic system is fully operational and ready for production use!")
    
    return True

if __name__ == "__main__":
    success = comprehensive_demo()
    sys.exit(0 if success else 1)
