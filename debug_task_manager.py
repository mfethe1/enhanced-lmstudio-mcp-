"""
Debug Task Manager

Simple script to check if tasks are being stored in the TaskManager.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import get_server_singleton

def main():
    print("Initializing server...")
    server = get_server_singleton()
    
    print(f"Server has agentic_handlers: {hasattr(server, 'agentic_handlers')}")
    print(f"Agentic handlers value: {server.agentic_handlers}")
    
    if server.agentic_handlers:
        print(f"Task manager: {server.agentic_handlers.task_manager}")
        print(f"Task manager tasks: {server.agentic_handlers.task_manager.tasks}")
        print(f"Number of tasks: {len(server.agentic_handlers.task_manager.tasks)}")
        
        # Create a test task
        print("\nCreating test task...")
        task_id = server.agentic_handlers.task_manager.create_task("test_tool", {"arg": "value"})
        print(f"Created task: {task_id}")
        
        # Try to retrieve it
        print("\nRetrieving task...")
        task = server.agentic_handlers.task_manager.get_task(task_id)
        print(f"Retrieved task: {task}")
        
        if task:
            print(f"Task ID: {task.task_id}")
            print(f"Tool name: {task.tool_name}")
            print(f"Status: {task.status}")
            print(f"Status value: {task.status.value}")
            print(f"Status upper: {task.status.value.upper()}")
        
        # Now test via handle_get_task_status
        print("\nTesting via handle_get_task_status...")
        from handlers import tasks
        
        result_str = tasks.handle_get_task_status({"task_id": task_id}, server)
        print(f"Result: {result_str}")
        
        result = json.loads(result_str)
        print(f"Parsed result: {json.dumps(result, indent=2)}")
        
    else:
        print("Agentic handlers not available")

if __name__ == "__main__":
    main()

