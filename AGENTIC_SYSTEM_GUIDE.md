# Agentic Task Management System for Jarvis MCP

## 🎯 Overview

The Jarvis MCP server has been enhanced with a comprehensive **Agentic Task Management System** that transforms it from a simple tool executor into a true autonomous agent platform. Users can now start complex tasks and check back later to see what the agent accomplished autonomously.

## 🌟 Key Features

### ✅ **Background Task Execution**
- Start any tool as a background task with unique job IDs
- Tasks run autonomously without blocking the user
- Support for long-running operations (research, code generation, analysis)

### ✅ **Persistent State Management**
- Tasks persist across server restarts
- JSON-based storage with platform-specific paths
- Complete task history and metadata tracking

### ✅ **Progress Tracking & Monitoring**
- Real-time status updates (pending, running, completed, failed, cancelled)
- Progress indicators and activity logging
- Detailed execution timelines

### ✅ **User Check-in Capabilities**
- Check task status anytime with `get_task_status`
- List all tasks with filtering options
- Retrieve complete results when ready

### ✅ **Activity Logging**
- Comprehensive audit trail for each task
- Timestamped progress updates
- Error tracking and debugging information

## 🛠️ Available Tools

### Core Agentic Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `start_agentic_task` | Start any tool as background task | Launch autonomous research, coding, analysis |
| `get_task_status` | Check progress of running task | Monitor agent progress |
| `list_all_tasks` | Show all tasks with filtering | View agent work history |
| `get_task_results` | Retrieve completed task output | Get final results |
| `cancel_task` | Stop running task | Cancel long-running operations |
| `get_task_activity_log` | View detailed progress history | Debug and audit agent work |

### Tools Available for Agentic Execution

The following existing tools have been enhanced for background execution:

- **`deep_research`** - Multi-round research with Firecrawl and CrewAI
- **`agent_team_plan_and_code`** - Multi-agent code planning and generation
- **`agent_team_review_and_test`** - Code review and testing workflows
- **`agent_team_refactor`** - Code refactoring analysis
- **`web_search`** - Web research and information gathering
- **`agent_collaborate`** - Multi-agent collaboration sessions

## 🚀 Usage Examples

### Starting a Background Research Task

```json
{
  "tool": "start_agentic_task",
  "arguments": {
    "tool_name": "deep_research",
    "tool_arguments": {
      "query": "Latest developments in autonomous AI agents 2024",
      "rounds": 3,
      "max_depth": 4,
      "time_limit": 300
    },
    "description": "Research autonomous AI agent developments for strategic planning"
  }
}
```

**Response:**
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "tool_name": "deep_research",
  "status": "started",
  "message": "Background task started for 'deep_research'. Use get_task_status to monitor progress.",
  "check_status_with": "get_task_status(task_id='a1b2c3d4-e5f6-7890-abcd-ef1234567890')"
}
```

### Checking Task Progress

```json
{
  "tool": "get_task_status",
  "arguments": {
    "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "include_log": true
  }
}
```

**Response:**
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "tool_name": "deep_research",
  "status": "running",
  "progress": "60%",
  "created_at": "2025-01-01T10:00:00Z",
  "started_at": "2025-01-01T10:00:05Z",
  "message": "Task is currently running. Progress: 60%",
  "activity_log": [
    {
      "timestamp": "2025-01-01T10:00:05Z",
      "message": "Task execution started",
      "level": "info"
    },
    {
      "timestamp": "2025-01-01T10:02:30Z",
      "message": "Completed round 1 of 3 - found 15 sources",
      "level": "info"
    },
    {
      "timestamp": "2025-01-01T10:05:45Z",
      "message": "Processing round 2 - analyzing key trends",
      "level": "info"
    }
  ]
}
```

### Retrieving Completed Results

```json
{
  "tool": "get_task_results",
  "arguments": {
    "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

**Response:**
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "tool_name": "deep_research",
  "status": "completed",
  "result": {
    "research_id": "research_a1b2c3d4",
    "summary": "Comprehensive analysis of autonomous AI agent developments...",
    "key_findings": [...],
    "sources": [...],
    "recommendations": [...]
  },
  "execution_time": "8m 45s",
  "created_at": "2025-01-01T10:00:00Z",
  "completed_at": "2025-01-01T10:08:45Z"
}
```

### Listing All Tasks

```json
{
  "tool": "list_all_tasks",
  "arguments": {
    "status_filter": "completed",
    "limit": 10,
    "include_results": false
  }
}
```

## 📁 Storage and Persistence

### Storage Location

Tasks are stored in platform-specific locations:

- **Linux**: `~/.local/share/jarvis-mcp/tasks.json`
- **macOS**: `~/Library/Application Support/jarvis-mcp/tasks.json`
- **Windows**: `%APPDATA%\jarvis-mcp\tasks.json`

### Custom Storage Path

Set the `AGENTIC_STORAGE_PATH` environment variable to use a custom location:

```bash
export AGENTIC_STORAGE_PATH="/path/to/custom/tasks.json"
```

### Task Data Structure

```json
{
  "tasks": {
    "task-uuid-123": {
      "task_id": "task-uuid-123",
      "tool_name": "deep_research",
      "arguments": {...},
      "status": "completed",
      "progress": "100%",
      "created_at": "2025-01-01T10:00:00Z",
      "started_at": "2025-01-01T10:00:05Z",
      "completed_at": "2025-01-01T10:08:45Z",
      "result": {...},
      "error": null,
      "activity_log": [...]
    }
  },
  "last_updated": "2025-01-01T10:08:45Z"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTIC_STORAGE_PATH` | Custom task storage location | Platform-specific |
| `AGENTIC_MAX_CONCURRENT_TASKS` | Max concurrent background tasks | 10 |
| `AGENTIC_TASK_TIMEOUT` | Default task timeout (seconds) | 3600 |

### Server Integration

The agentic system is automatically initialized when the Jarvis MCP server starts:

```python
# In server.py
if AGENTIC_AVAILABLE:
    self.task_manager = TaskManager(storage_path)
    self.agentic_handlers = AgenticToolHandlers(self.task_manager)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_agentic_system.py
```

This validates:
- ✅ Task creation and background execution
- ✅ Status monitoring and progress tracking
- ✅ Results retrieval and activity logging
- ✅ Task persistence across restarts
- ✅ Error handling and recovery

## 🎯 Use Cases

### 1. **Autonomous Research**
Start comprehensive research tasks that run for hours, gathering and analyzing information from multiple sources.

### 2. **Code Generation Projects**
Launch multi-agent coding sessions that plan, implement, review, and test code changes autonomously.

### 3. **Long-running Analysis**
Execute complex data analysis, performance testing, or system diagnostics in the background.

### 4. **Scheduled Workflows**
Set up recurring tasks for monitoring, reporting, or maintenance activities.

### 5. **Multi-step Automation**
Chain together multiple tools and processes for complex automation workflows.

## 🔄 Task Lifecycle

```
PENDING → RUNNING → COMPLETED
    ↓         ↓         ↑
CANCELLED ← FAILED ←──┘
```

1. **PENDING**: Task created, waiting to start
2. **RUNNING**: Task executing in background
3. **COMPLETED**: Task finished successfully
4. **FAILED**: Task encountered an error
5. **CANCELLED**: Task stopped by user

## 🛡️ Error Handling

- **Automatic retry** for transient failures
- **Graceful degradation** when dependencies unavailable
- **Comprehensive logging** for debugging
- **Task recovery** after server restarts
- **Resource cleanup** for cancelled tasks

## 🚀 Getting Started

1. **Start the Jarvis MCP server** (agentic system loads automatically)
2. **Launch a background task** using `start_agentic_task`
3. **Monitor progress** with `get_task_status`
4. **Retrieve results** when complete using `get_task_results`
5. **Review activity** with `get_task_activity_log`

The agentic system transforms Jarvis MCP into a true autonomous agent platform where users can delegate complex tasks and check back later to see what the agent accomplished!
