# Enhanced LM Studio MCP Server v2.0

A significantly enhanced Model Context Protocol (MCP) server that provides advanced tools for coding agents to solve complex problems iteratively. This server connects to LM Studio and provides a comprehensive toolkit for code analysis, debugging, execution, and iterative problem-solving.

## 🚀 Key Features

### Sequential Thinking & Problem Solving
- **Dynamic reasoning workflows** with adaptive thought processes
- **Reflective thinking** that can revise and branch previous thoughts
- **Context-aware analysis** that builds understanding over time
- **Progress tracking** through complex problem-solving sessions

### Advanced Code Analysis
- **Multi-dimensional code analysis** (bugs, optimization, explanation, refactoring)
- **Detailed code explanations** with step-by-step breakdowns
- **Intelligent improvement suggestions** based on best practices
- **Automated test generation** for various testing frameworks

### Code Execution & Testing
- **Safe code execution** in isolated environments
- **Multi-language support** (Python, JavaScript, Bash)
- **Test framework integration** with detailed result reporting
- **Timeout protection** and error handling

### File System Operations
- **Smart file reading/writing** with line-specific operations
- **Directory exploration** and file discovery
- **Pattern-based file searching** with regex support
- **Project structure analysis**

### Memory & Context Management
- **Persistent memory storage** for learning and context retention
- **Categorized information** retrieval
- **Search-based memory access** for relevant information lookup
- **Session continuity** across interactions

### Enhanced Debugging Tools
- **Comprehensive debug analysis** with error context
- **Execution tracing** for step-by-step debugging
- **Error pattern recognition** and solution suggestions
- **Memory-based error learning**

## 📋 Available Tools

### 1. Sequential Thinking Tools
- `sequential_thinking` - Dynamic problem-solving through structured thoughts

### 2. Code Analysis Tools
- `analyze_code` - Multi-type code analysis (bugs/optimization/explanation/refactor)
- `explain_code` - Detailed code explanation and documentation
- `suggest_improvements` - Intelligent code improvement recommendations
- `generate_tests` - Automated test case generation

### 3. Execution & Testing Tools
- `execute_code` - Safe code execution in isolated environments
- `run_tests` - Test framework integration and execution

### 4. File System Tools
- `read_file_content` - Smart file reading with line range support
- `write_file_content` - Safe file writing with mode options
- `list_directory` - Directory exploration and file discovery
- `search_files` - Pattern-based file searching with regex

### 5. Memory Management Tools
- `store_memory` - Persistent information storage with categorization
- `retrieve_memory` - Intelligent memory retrieval and searching

### 6. Debugging Tools
- `debug_analyze` - Comprehensive debugging analysis with context
- `trace_execution` - Step-by-step execution tracing

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- LM Studio running locally (default: http://localhost:1234)
- DeepSeek R1 or compatible model loaded in LM Studio

### Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables (optional):**
   ```bash
   export LM_STUDIO_URL="http://localhost:1234"
   export MODEL_NAME="deepseek-r1-distill-qwen-7b"
   ```

3. **Start the MCP server:**
   ```bash
   python server.py
   ```

### Configuration Options
The server can be configured through environment variables:

- `LM_STUDIO_URL` - LM Studio API endpoint (default: http://localhost:1234)
- `MODEL_NAME` - Model name to use (default: deepseek-r1-distill-qwen-7b)


### MCP Manager (Augment) setup
- Command: `python`
- Args: `server.py`
- Working directory: `<project root>`
- Do not auto-start server at OS boot. Let the MCP manager spawn it so stdin/stdout pipes are wired correctly.

### Health check
- `health_check` supports a lightweight readiness probe. You can optionally pass `{ "probe_lm": true }` in arguments to ping the LM backend quickly.

### LLM tool flags
- `compact: true` to request shorter bullet lists and fewer tests
- `code_only: true` (generate_tests) to return only test code (we post-process to extract fenced code when present)

## 💡 Usage Examples

### Sequential Problem Solving
```json
{
  "method": "tools/call",
  "params": {
    "name": "sequential_thinking",
    "arguments": {
      "thought": "First, I need to understand the core problem...",
      "thought_number": 1,
      "total_thoughts": 5,
      "next_thought_needed": true
    }
  }
}
```

### Code Analysis & Debugging
```json
{
  "method": "tools/call",
  "params": {
    "name": "debug_analyze",
    "arguments": {
      "code": "def problematic_function():\n    return x / 0",
      "error_message": "ZeroDivisionError: division by zero",
      "context": "Function called during user input validation"
    }
  }
}
```

### Safe Code Execution
```json
{
  "method": "tools/call",
  "params": {
    "name": "execute_code",
    "arguments": {
      "code": "print('Hello, World!')\nfor i in range(3):\n    print(f'Count: {i}')",
      "language": "python",
      "timeout": 30
    }
  }
}
```

### Memory Management
```json
{
  "method": "tools/call",
  "params": {
    "name": "store_memory",
    "arguments": {
      "key": "authentication_pattern",
      "value": "Always use JWT tokens with 1-hour expiration",
      "category": "security"
    }
  }
}
```

## 🔧 Advanced Features

### Iterative Problem Solving Workflow
1. **Analysis Phase**: Use `sequential_thinking` to break down complex problems
2. **Research Phase**: Use `search_files` and `read_file_content` to gather context
3. **Development Phase**: Use `execute_code` and `generate_tests` for implementation
4. **Debugging Phase**: Use `debug_analyze` and `trace_execution` for troubleshooting
5. **Learning Phase**: Use `store_memory` to capture insights for future use

### Memory-Driven Learning
The server maintains persistent memory across sessions:
- **Error Patterns**: Automatically stores debugging insights
- **Solution Patterns**: Remembers successful problem-solving approaches
- **Code Patterns**: Captures effective code structures and practices
- **Context Information**: Maintains project-specific knowledge

### Safety Features
- **Sandboxed Execution**: Code runs in temporary, isolated environments
- **Timeout Protection**: Prevents infinite loops and long-running processes
- **Error Handling**: Comprehensive error catching and reporting
- **Resource Limits**: Built-in protection against resource exhaustion

## 🎯 Integration with Coding Agents

This enhanced MCP server is specifically designed to work with advanced coding agents that need:

### Iterative Problem Solving
- **Step-by-step reasoning** through complex technical challenges
- **Adaptive thinking** that can revise approaches based on new information
- **Context accumulation** across multiple interaction sessions

### Advanced Code Understanding
- **Deep code analysis** beyond basic syntax checking
- **Pattern recognition** for common coding problems and solutions
- **Multi-dimensional improvement** suggestions

### Practical Development Tools
- **Real execution** capabilities for testing and validation
- **File system integration** for project-wide operations
- **Memory persistence** for learning and improvement over time

## 🔍 Troubleshooting

### Common Issues

**Connection Errors:**
- Ensure LM Studio is running on the configured port (default: 1234)
- Check that the model is loaded in LM Studio
- Verify network connectivity to LM Studio

**Execution Errors:**
- Ensure Python interpreter is available for code execution
- Check file permissions for temporary file creation
- Verify timeout settings for long-running operations

**Memory Issues:**
- Memory storage is in-process and will reset on server restart
- For persistent memory, consider implementing database storage
- Monitor memory usage with large datasets

### Performance Optimization
- Use appropriate timeout values for your use case
- Consider model-specific optimizations in LM Studio
- Monitor server resource usage during heavy operations

## 📈 Version History
## ⚖️ Safety Defaults (P0)

- ALLOWED_BASE_DIR: All filesystem tools are restricted to this base directory (default: current working directory). Set via environment variable.
- EXECUTION_ENABLED: Code and test execution tools are disabled by default. Enable by setting EXECUTION_ENABLED=true.
- Input validation: Tools now return JSON-RPC -32602 for invalid parameters (e.g., paths outside base dir).
- File search ignores heavy directories by default (e.g., .git, node_modules, venv, dist, build) and caps results.

### New/Updated Environment Variables
- ALLOWED_BASE_DIR: Absolute path to restrict file operations
- EXECUTION_ENABLED: true|false to gate code/test execution
- PERFORMANCE_THRESHOLD: seconds for performance alerts (default 0.2)



### v2.0.0 (Current)
- Complete rewrite with 16 advanced tools
- Sequential thinking and reasoning capabilities
- Memory management and context persistence
- Enhanced debugging and execution tools
- Comprehensive file system operations
- Safety features and error handling

### v1.0.0 (Previous)
- Basic code analysis tools (4 tools)
- Simple LM Studio integration
- Limited functionality

## 🤝 Contributing

This project is designed to be extended and improved. Key areas for contribution:
- Additional language support for code execution
- Enhanced memory persistence (database integration)
- More sophisticated debugging tools
- Integration with additional development tools
- Performance optimizations

## 📄 License

MIT License - see package.json for details.

---

**Enhanced LM Studio MCP Server v2.0** - Empowering coding agents with advanced iterative problem-solving capabilities.