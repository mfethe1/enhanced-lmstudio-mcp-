#!/usr/bin/env python3
"""
Test script for Enhanced LM Studio MCP Server

This script demonstrates the key capabilities of the enhanced MCP server
and can be used for validation and testing.
"""

import json
import subprocess
import sys
import time

def test_mcp_tools():
    """Test the enhanced MCP server tools"""
    
    # Test data
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# This will cause performance issues for large n
result = fibonacci(35)
print(f"Fibonacci result: {result}")
"""
    
    test_cases = [
        # Test 1: Sequential Thinking
        {
            "name": "Sequential Thinking Test",
            "tool": "sequential_thinking",
            "arguments": {
                "thought": "I need to analyze this fibonacci implementation for performance issues",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True
            }
        },
        
        # Test 2: Code Analysis
        {
            "name": "Code Analysis Test", 
            "tool": "analyze_code",
            "arguments": {
                "code": test_code,
                "analysis_type": "optimization"
            }
        },
        
        # Test 3: Code Execution
        {
            "name": "Code Execution Test",
            "tool": "execute_code", 
            "arguments": {
                "code": "print('Hello from MCP!')\nfor i in range(3):\n    print(f'Count: {i}')",
                "language": "python",
                "timeout": 10
            }
        },
        
        # Test 4: Memory Storage
        {
            "name": "Memory Storage Test",
            "tool": "store_memory",
            "arguments": {
                "key": "fibonacci_optimization",
                "value": "Use memoization or dynamic programming to optimize recursive fibonacci",
                "category": "optimization"
            }
        },
        
        # Test 5: Memory Retrieval
        {
            "name": "Memory Retrieval Test", 
            "tool": "retrieve_memory",
            "arguments": {
                "category": "optimization"
            }
        },
        
        # Test 6: File Operations
        {
            "name": "File Operations Test",
            "tool": "list_directory",
            "arguments": {
                "directory_path": ".",
                "include_hidden": False
            }
        },
        
        # Test 7: Debug Analysis
        {
            "name": "Debug Analysis Test",
            "tool": "debug_analyze",
            "arguments": {
                "code": "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
                "error_message": "ZeroDivisionError: division by zero",
                "context": "User input validation missing"
            }
        }
    ]
    
    print("🚀 Enhanced LM Studio MCP Server Test Suite")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        # Create MCP message
        message = {
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {
                "name": test_case["tool"],
                "arguments": test_case["arguments"]
            }
        }
        
        print(f"Tool: {test_case['tool']}")
        print(f"Arguments: {json.dumps(test_case['arguments'], indent=2)}")
        print("\n✅ Test case prepared successfully")
        
        # Note: In a real test, you would send this to the MCP server
        # via stdin/stdout communication. This is just a demonstration
        # of the test structure.
        
    print(f"\n🎉 All {len(test_cases)} test cases prepared!")
    print("\nTo run actual tests:")
    print("1. Start LM Studio with DeepSeek R1 model")
    print("2. Run: python server.py")
    print("3. Send JSON messages via stdin to test functionality")

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "lm_studio": {
            "url": "http://localhost:1234",
            "model": "deepseek-r1-distill-qwen-7b",
            "timeout": 120
        },
        "server": {
            "max_memory_entries": 1000,
            "code_execution_timeout": 30,
            "temp_file_cleanup": True
        },
        "debugging": {
            "enable_trace": True,
            "log_level": "INFO",
            "store_debug_sessions": True
        },
        "safety": {
            "sandbox_execution": True,
            "max_file_size": "10MB",
            "allowed_extensions": [".py", ".js", ".txt", ".md"]
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("📝 Created config.json with default settings")

def performance_benchmark():
    """Simple performance benchmark for key operations"""
    print("\n⚡ Performance Benchmark")
    print("-" * 25)
    
    # Simulate tool response times
    tools_benchmark = {
        "sequential_thinking": "~50ms",
        "code_analysis": "~2-5s (depends on LLM)",
        "code_execution": "~100ms-30s (depends on code)",
        "file_operations": "~10-50ms",
        "memory_operations": "~5-10ms",
        "debug_analysis": "~1-3s (depends on LLM)"
    }
    
    for tool, time_estimate in tools_benchmark.items():
        print(f"• {tool:<20}: {time_estimate}")
    
    print("\n💡 Performance Tips:")
    print("- Use appropriate timeouts for code execution")
    print("- Consider LLM response times for analysis tools")
    print("- Memory operations are fastest for quick lookups")
    print("- File operations scale with file size")

if __name__ == "__main__":
    print("🧪 Enhanced MCP Server Test & Configuration Utility")
    print("=" * 55)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            test_mcp_tools()
        elif command == "config":
            create_sample_config()
        elif command == "benchmark":
            performance_benchmark()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, config, benchmark")
    else:
        print("Available commands:")
        print("  python test_mcp_server.py test       - Run test suite")
        print("  python test_mcp_server.py config     - Create config file")
        print("  python test_mcp_server.py benchmark  - Show performance info")
        print("\nOr run without arguments to see this help.") 