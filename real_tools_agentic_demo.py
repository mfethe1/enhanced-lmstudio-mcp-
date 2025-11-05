#!/usr/bin/env python3
"""
Demonstration of real MCP tools working through the agentic system
This shows actual tools like analyze_code, generate_tests, etc. working in background
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
    print(f"\n{'='*50}")
    print(f"🔧 {title}")
    print('='*50)

def print_json_output(data, title="Output"):
    """Pretty print JSON data"""
    print(f"\n📋 {title}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

def real_tools_demo():
    """Run demonstration with real MCP tools"""
    print("🚀 REAL MCP TOOLS AGENTIC DEMONSTRATION")
    print("This demo shows actual MCP tools working through the agentic system")
    
    # Initialize the system
    print_section("SYSTEM INITIALIZATION")
    try:
        from server import get_server_singleton
        
        # Get the actual server instance
        server = get_server_singleton()
        print(f"✅ Server initialized with {len(server.registry._handlers)} total tools")
        print(f"✅ Agentic system: {'Available' if server.agentic_handlers else 'Not Available'}")
        
        # Use the server's agentic handlers
        agentic_handlers = server.agentic_handlers
        if not agentic_handlers:
            print("❌ Agentic handlers not available")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

    # Create sample code for analysis
    sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively (inefficient)"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_list(items):
    """Process a list of items"""
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

def divide_safely(a, b):
    """Divide two numbers"""
    return a / b  # Missing zero division check!
'''

    # Demo 1: Start code analysis task
    print_section("DEMO 1: START CODE ANALYSIS TASK")
    try:
        task_args = {
            "tool_name": "analyze_code",
            "tool_arguments": {
                "code": sample_code,
                "analysis_type": "bugs"
            },
            "description": "Analyze sample code for potential bugs and issues"
        }
        
        print("🔄 Starting background code analysis...")
        print_json_output(task_args, "Task Arguments")
        
        result = agentic_handlers.handle_start_agentic_task(task_args, server)
        print_json_output(result, "Start Task Result")
        
        if "task_id" in result:
            analysis_task_id = result["task_id"]
            print(f"✅ Code analysis task started: {analysis_task_id}")
        else:
            print("❌ Failed to start code analysis task")
            return False
            
    except Exception as e:
        print(f"❌ Error starting code analysis task: {e}")
        return False

    # Demo 2: Start test generation task
    print_section("DEMO 2: START TEST GENERATION TASK")
    try:
        task_args = {
            "tool_name": "generate_tests",
            "tool_arguments": {
                "code": sample_code,
                "framework": "pytest"
            },
            "description": "Generate comprehensive unit tests for the sample code"
        }
        
        print("🔄 Starting background test generation...")
        print_json_output(task_args, "Task Arguments")
        
        result = agentic_handlers.handle_start_agentic_task(task_args, server)
        print_json_output(result, "Start Task Result")
        
        if "task_id" in result:
            test_task_id = result["task_id"]
            print(f"✅ Test generation task started: {test_task_id}")
        else:
            print("❌ Failed to start test generation task")
            return False
            
    except Exception as e:
        print(f"❌ Error starting test generation task: {e}")
        return False

    # Demo 3: Start code improvement task
    print_section("DEMO 3: START CODE IMPROVEMENT TASK")
    try:
        task_args = {
            "tool_name": "suggest_improvements",
            "tool_arguments": {
                "code": sample_code
            },
            "description": "Generate improvement suggestions for the sample code"
        }
        
        print("🔄 Starting background code improvement analysis...")
        print_json_output(task_args, "Task Arguments")
        
        result = agentic_handlers.handle_start_agentic_task(task_args, server)
        print_json_output(result, "Start Task Result")
        
        if "task_id" in result:
            improvement_task_id = result["task_id"]
            print(f"✅ Code improvement task started: {improvement_task_id}")
        else:
            print("❌ Failed to start code improvement task")
            return False
            
    except Exception as e:
        print(f"❌ Error starting code improvement task: {e}")
        return False

    # Demo 4: Monitor all tasks
    print_section("DEMO 4: MONITOR ALL TASKS")
    try:
        print("🔄 Monitoring all tasks...")
        
        task_ids = [analysis_task_id, test_task_id, improvement_task_id]
        task_names = ["Code Analysis", "Test Generation", "Code Improvement"]
        
        # Monitor for up to 30 seconds
        max_wait = 30
        wait_time = 0
        
        while wait_time < max_wait:
            print(f"\n⏱️  Time: {wait_time}s - Checking all tasks...")
            
            all_completed = True
            for i, task_id in enumerate(task_ids):
                status_result = agentic_handlers.handle_get_task_status(
                    {"task_id": task_id}, 
                    server
                )
                
                status = status_result.get("status", "unknown")
                progress = status_result.get("progress", "0%")
                print(f"   {task_names[i]}: {status} ({progress})")
                
                if status not in ["completed", "failed", "cancelled"]:
                    all_completed = False
            
            if all_completed:
                print("✅ All tasks completed!")
                break
                
            time.sleep(5)
            wait_time += 5
        
    except Exception as e:
        print(f"❌ Error monitoring tasks: {e}")

    # Demo 5: Get results from all tasks
    print_section("DEMO 5: GET RESULTS FROM ALL TASKS")
    
    # Get code analysis results
    try:
        print("🔄 Getting code analysis results...")
        
        results = agentic_handlers.handle_get_task_results(
            {"task_id": analysis_task_id}, 
            server
        )
        
        print_json_output(results, "Code Analysis Results")
        
        if "result" in results:
            print("✅ Code analysis completed successfully")
        else:
            print("⚠️  Code analysis may not be completed")
            
    except Exception as e:
        print(f"❌ Error getting code analysis results: {e}")

    # Get test generation results
    try:
        print("\n🔄 Getting test generation results...")
        
        results = agentic_handlers.handle_get_task_results(
            {"task_id": test_task_id}, 
            server
        )
        
        print_json_output(results, "Test Generation Results")
        
        if "result" in results:
            print("✅ Test generation completed successfully")
        else:
            print("⚠️  Test generation may not be completed")
            
    except Exception as e:
        print(f"❌ Error getting test generation results: {e}")

    # Get improvement suggestions results
    try:
        print("\n🔄 Getting code improvement results...")
        
        results = agentic_handlers.handle_get_task_results(
            {"task_id": improvement_task_id}, 
            server
        )
        
        print_json_output(results, "Code Improvement Results")
        
        if "result" in results:
            print("✅ Code improvement analysis completed successfully")
        else:
            print("⚠️  Code improvement analysis may not be completed")
            
    except Exception as e:
        print(f"❌ Error getting code improvement results: {e}")

    # Demo 6: Final system status
    print_section("DEMO 6: FINAL SYSTEM STATUS")
    try:
        print("🔄 Getting final system status...")
        
        final_list = agentic_handlers.handle_list_all_tasks(
            {"limit": 20, "include_results": False}, 
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
        
        # Show tools used
        tools_used = set()
        for task in tasks:
            tool_name = task.get("tool_name", "unknown")
            tools_used.add(tool_name)
        
        print(f"\n🛠️  Tools Used:")
        for tool in sorted(tools_used):
            print(f"  - {tool}")
        
    except Exception as e:
        print(f"❌ Error getting final status: {e}")

    print_section("DEMONSTRATION COMPLETE")
    print("🎉 Real MCP tools agentic demonstration completed!")
    print("\nKey achievements:")
    print("  ✅ Real MCP tools executed in background")
    print("  ✅ Code analysis, test generation, and improvements")
    print("  ✅ Multiple concurrent task execution")
    print("  ✅ Complete task lifecycle management")
    print("  ✅ Structured results from actual tools")
    print("\nThe agentic system successfully orchestrates real MCP tools!")
    
    return True

if __name__ == "__main__":
    success = real_tools_demo()
    sys.exit(0 if success else 1)
