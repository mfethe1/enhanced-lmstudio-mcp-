#!/usr/bin/env python3
"""
Test script to verify agentic tools are properly integrated into the server
"""

def test_server_integration():
    print("🔧 Testing Server Integration")
    print("=" * 40)
    
    try:
        from server import get_all_tools
        tools = get_all_tools()
        
        # Find agentic tools
        agentic_tools = [t for t in tools['tools'] if 'agentic' in t['name'] or 'task' in t['name']]
        
        print(f"Found {len(agentic_tools)} agentic tools:")
        for tool in agentic_tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Check if all expected tools are present
        expected_tools = [
            'start_agentic_task',
            'get_task_status', 
            'list_all_tasks',
            'get_task_results',
            'cancel_task',
            'get_task_activity_log'
        ]
        
        found_tools = [t['name'] for t in agentic_tools]
        missing_tools = [t for t in expected_tools if t not in found_tools]
        
        if missing_tools:
            print(f"\n❌ Missing tools: {missing_tools}")
            return False
        else:
            print(f"\n✅ All {len(expected_tools)} agentic tools are properly integrated!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing server integration: {e}")
        return False

if __name__ == "__main__":
    success = test_server_integration()
    exit(0 if success else 1)
