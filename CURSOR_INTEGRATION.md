# Enhanced MCP Server v2.1 - Cursor Integration Guide

## 🚀 Quick Start

### 1. Start LM Studio
- Launch LM Studio
- Load the DeepSeek R1 model: `deepseek/deepseek-r1-0528-qwen3-8b`
- Enable API Server (Settings > Developer > Enable Local Server)
- Verify it's running on http://localhost:1234

### 2. Start Enhanced MCP Server
```bash
cd /path/to/lmstudio-mcp
python server.py
```

### 3. Enhanced Features Available

#### 🧠 Persistent Memory
- `store_memory` - Store information that persists across sessions
- `retrieve_memory` - Search and retrieve stored memories
- All memories are saved to SQLite database

#### 📊 Performance Monitoring
- `get_performance_stats` - View tool performance metrics
- `get_error_patterns` - Analyze error trends
- Automatic performance tracking for all tools

#### 🔍 Sequential Thinking
- Enhanced with persistent storage
- Thoughts are saved and can be retrieved later
- Supports branching and revision of thoughts

#### 🛠 Enhanced Tools
- All original tools now have performance monitoring
- Error tracking and pattern analysis
- Improved memory management

## 📈 New Capabilities

### Performance Monitoring
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_performance_stats",
    "arguments": {"hours": 24}
  }
}
```

### Error Analysis
```json
{
  "method": "tools/call", 
  "params": {
    "name": "get_error_patterns",
    "arguments": {"hours": 24}
  }
}
```

### Enhanced Memory
```json
{
  "method": "tools/call",
  "params": {
    "name": "retrieve_memory",
    "arguments": {
      "category": "debugging",
      "search_term": "error"
    }
  }
}
```

## 🔧 Configuration

### Environment Variables
- `LM_STUDIO_URL` - LM Studio API endpoint (default: http://localhost:1234)
- `MODEL_NAME` - Model to use (default: deepseek/deepseek-r1-0528-qwen3-8b)
- `PERFORMANCE_THRESHOLD` - Performance alert threshold in seconds (default: 0.2)

### Database Location
- SQLite database: `mcp_data.db` (created automatically)
- Contains: memories, thinking sessions, performance metrics, error logs

## 💡 Tips for Cursor Integration

1. **Use Sequential Thinking** for complex problems
2. **Store important insights** in memory for later retrieval  
3. **Monitor performance** to identify slow operations
4. **Check error patterns** to improve code quality
5. **Leverage persistent context** across sessions

## 🚨 Troubleshooting

### Common Issues
1. **Database locked** - Close other instances of the server
2. **LM Studio connection failed** - Check if API server is enabled
3. **Performance alerts** - Check system resources
4. **Memory full** - Use cleanup tools or increase limits

### Performance Optimization
- Monitor tool execution times with `get_performance_stats`
- Set appropriate `PERFORMANCE_THRESHOLD`
- Use database cleanup for old data
- Monitor error patterns to prevent issues

## 📊 Monitoring Dashboard

The enhanced server provides real-time monitoring:
- Tool execution times and success rates
- Error frequency and patterns
- Memory usage and storage statistics
- Thinking session analytics

## 🎯 Best Practices

1. **Regular Memory Management**: Store key insights and solutions
2. **Performance Monitoring**: Check stats regularly for optimization
3. **Error Prevention**: Use error patterns to improve code
4. **Context Persistence**: Leverage long-term memory for complex projects
5. **Sequential Thinking**: Use for iterative problem-solving

---

**Enhanced MCP Server v2.1** - Your persistent, intelligent coding companion! 🤖✨
