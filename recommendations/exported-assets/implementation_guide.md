# Implementation Guide: Enhanced Multi-Agent MCP Server

## Overview

This guide shows how to integrate the enhanced multi-agent architecture with your existing MCP server for optimal collaboration with Augment Code.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                Enhanced MCP Server                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Task Analysis  │  │    Complexity Router       │   │
│  │  & Routing      │  │  ┌─────┬────────┬────────┐ │   │
│  └─────────────────┘  │  │Local│ OpenAI │Anthropic│ │   │
│           │            │  └─────┴────────┴────────┘ │   │
│           ▼            └─────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Collaborative Agent Teams                 ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │Researcher│ │ Planner  │ │  Coder   │  ...      ││
│  │  └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────┘│
│           │                                             │
│           ▼                                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │        Enhanced Storage & Context Management        ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   ││
│  │  │   Context   │ │Collaboration│ │  Lineage    │   ││
│  │  │  Envelopes  │ │  Artifacts  │ │  Tracking   │   ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘   ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Step 1: Update Your Main MCP Server

Replace your existing server initialization with the enhanced version:

```python
# enhanced_mcp_server.py
import asyncio
import json
import sys
import os
import logging
from pathlib import Path

from enhanced_mcp_storage import EnhancedMCPStorage
from enhanced_agent_teams import EnhancedAgentTeamManager
from enhanced_mcp_tools import get_enhanced_mcp_tools, EnhancedMCPToolHandlers

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_mcp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMCPServer:
    def __init__(self):
        # Initialize enhanced storage
        self.storage = EnhancedMCPStorage("enhanced_mcp.db")
        
        # Initialize tool handlers
        self.tool_handlers = EnhancedMCPToolHandlers(self.storage)
        
        # Initialize agent team manager
        self.team_manager = EnhancedAgentTeamManager(self.storage)
        
        # Performance monitoring
        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.5"))
        
        logger.info("Enhanced MCP Server initialized")

    def get_tools(self):
        """Return enhanced tool definitions"""
        return get_enhanced_mcp_tools()

    async def handle_tool_call(self, message):
        """Enhanced tool call handler with proper routing"""
        try:
            params = message.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Route to enhanced handlers
            result = await self._route_tool_call(tool_name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2)}
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Tool call error: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Tool execution error: {str(e)}"
                }
            }

    async def _route_tool_call(self, tool_name: str, arguments: dict):
        """Route tool calls to appropriate handlers"""
        handler_map = {
            # Enhanced collaborative tools
            "collaborative_agent_task": self.tool_handlers.handle_collaborative_agent_task,
            "collaborative_research": self._handle_collaborative_research,
            
            # Context management
            "create_context_envelope": self.tool_handlers.handle_create_context_envelope,
            "get_context_chain": self.tool_handlers.handle_get_context_chain,
            
            # Analytics and monitoring
            "get_agent_analytics": self.tool_handlers.handle_get_agent_analytics,
            "check_backend_health": self.tool_handlers.handle_check_backend_health,
            
            # Maintenance
            "cleanup_expired_contexts": self.tool_handlers.handle_cleanup_expired_contexts,
        }
        
        handler = handler_map.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return await handler(arguments)

    def handle_message(self, message):
        """Handle incoming MCP messages"""
        method = message.get("method")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "enhanced-collaborative-mcp-server",
                        "version": "3.0.0"
                    }
                }
            }
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": self.get_tools()
            }
        elif method == "tools/call":
            return asyncio.create_task(self.handle_tool_call(message))
        else:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {}
            }

def main():
    """Main entry point for enhanced MCP server"""
    try:
        server = EnhancedMCPServer()
        logger.info("Enhanced Collaborative MCP Server v3.0 starting...")
        
        # MCP protocol communication via stdin/stdout
        for line in sys.stdin:
            try:
                message = json.loads(line.strip())
                response = server.handle_message(message)
                
                if asyncio.iscoroutine(response):
                    # Handle async responses
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(response)
                    loop.close()
                
                if response:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Step 2: Environment Configuration

Create an `.env` file with the required configuration:

```bash
# .env file for Enhanced MCP Server

# LLM Backend Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LM_STUDIO_URL=http://localhost:1234
MODEL_NAME=openai/gpt-oss-20b

# Backend Routing Override (optional)
# Format: role=backend,role=backend or all=backend
# AGENT_BACKEND_OVERRIDE=security=anthropic,domain_expert=anthropic

# Storage Configuration
STORAGE_BACKEND=sqlite  # or postgres
# POSTGRES_DSN=postgresql://user:password@localhost/enhanced_mcp

# Performance Settings
PERFORMANCE_THRESHOLD=0.5  # seconds
EXECUTION_ENABLED=true

# File System Security
ALLOWED_BASE_DIR=/your/project/root

# Augment Code Integration
MCP_ADMIN_TOKEN=your_secure_token
MCP_STORAGE_PATH=/data/mcp-storage

# Research Tools
FIRECRAWL_API_KEY=your_firecrawl_key
```

## Step 3: Augment Code Integration

Configure your `augment.config.json` for optimal MCP integration:

```json
{
  "mcp": {
    "servers": {
      "enhanced-collaborative": {
        "command": "python",
        "args": ["enhanced_mcp_server.py"],
        "env": {
          "PYTHONPATH": ".",
          "LOG_LEVEL": "INFO"
        },
        "capabilities": {
          "context_persistence": true,
          "collaboration": true,
          "complexity_routing": true
        }
      }
    }
  },
  "context": {
    "lineage": {
      "enabled": true,
      "track_changes": true,
      "mcp_integration": true
    },
    "envelope": {
      "max_size_mb": 10,
      "ttl_hours": 24,
      "auto_cleanup": true
    }
  },
  "agents": {
    "collaboration": {
      "max_concurrent_sessions": 5,
      "default_discussion_rounds": 2,
      "research_timeout_minutes": 10
    },
    "routing": {
      "complexity_analysis": true,
      "backend_failover": true,
      "performance_monitoring": true
    }
  }
}
```

## Step 4: Usage Examples

### Basic Collaborative Task

```python
# Example: Using the enhanced collaborative agent system
import json
import subprocess

def call_mcp_tool(tool_name, arguments):
    """Helper function to call MCP tools"""
    message = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    # This would normally be handled by Augment Code
    # Here's a simulation for testing
    result = subprocess.run(
        ["python", "enhanced_mcp_server.py"],
        input=json.dumps(message),
        capture_output=True,
        text=True
    )
    
    return json.loads(result.stdout)

# Example 1: Collaborative code analysis and improvement
result = call_mcp_tool("collaborative_agent_task", {
    "task": "Analyze the security vulnerabilities in the authentication module and propose fixes",
    "target_files": ["auth.py", "models.py"],
    "research_required": True,
    "discussion_rounds": 3,
    "complexity_override": "expert"
})

print("Collaboration Result:", json.dumps(result, indent=2))

# Example 2: Context-aware research
research_result = call_mcp_tool("collaborative_research", {
    "research_topic": "Best practices for secure password storage in Python web applications",
    "research_domains": ["security", "cryptography", "web_development"],
    "depth_level": "deep",
    "synthesis_required": True
})

print("Research Result:", json.dumps(research_result, indent=2))
```

### Advanced Context Management

```python
# Example: Working with context envelopes for persistent state

# Start a new collaborative session
session_result = call_mcp_tool("start_collaboration_session", {
    "session_type": "architecture",
    "participants": ["architect", "security_expert", "performance_expert"],
    "initial_context": {
        "project": "web_application_refactor",
        "goals": ["improve_scalability", "enhance_security", "reduce_complexity"]
    }
})

session_id = session_result["session_id"]

# Create context envelope for current analysis
context_result = call_mcp_tool("create_context_envelope", {
    "session_id": session_id,
    "context_type": "analysis",
    "content": {
        "current_architecture": "monolithic_django_app",
        "pain_points": ["slow_response_times", "difficult_deployments", "security_concerns"],
        "requirements": ["sub_second_response", "zero_downtime_deploys", "security_compliance"]
    },
    "metadata": {
        "analysis_date": "2025-01-01",
        "team_size": 5
    },
    "ttl_seconds": 7200  # 2 hours
})

# Execute collaborative task with context
task_result = call_mcp_tool("collaborative_agent_task", {
    "task": "Design microservices architecture for the Django monolith",
    "constraints": "Must maintain data consistency and support gradual migration",
    "research_required": True,
    "discussion_rounds": 2
})

# Get the complete context chain
context_chain = call_mcp_tool("get_context_chain", {
    "session_id": session_id,
    "include_expired": False
})

print("Complete Context Chain:", json.dumps(context_chain, indent=2))
```

## Step 5: Performance Monitoring and Analytics

```python
# Monitor agent performance and collaboration effectiveness

# Check backend health
health_check = call_mcp_tool("check_backend_health", {
    "include_latency": True,
    "test_simple_request": True
})

print("Backend Health:", json.dumps(health_check, indent=2))

# Get detailed analytics for a session
analytics = call_mcp_tool("get_agent_analytics", {
    "session_id": session_id,
    "include_interactions": True,
    "time_range_hours": 24
})

print("Session Analytics:", json.dumps(analytics, indent=2))

# Analyze task complexity before execution
complexity_analysis = call_mcp_tool("analyze_task_complexity", {
    "task_description": "Implement OAuth2 authentication with JWT tokens and refresh token rotation",
    "context": {
        "existing_auth": "django_sessions",
        "user_base": "10000_active_users"
    },
    "suggest_team": True
})

print("Complexity Analysis:", json.dumps(complexity_analysis, indent=2))
```

## Step 6: Best Practices

### 1. Context Management
- Use context envelopes for maintaining state across agent interactions
- Set appropriate TTL values based on task duration
- Clean up expired contexts regularly

### 2. Backend Selection
- Use local models for simple tasks (formatting, basic queries)
- Use OpenAI for moderate complexity (code explanation, documentation)
- Use Anthropic for complex reasoning and domain expertise

### 3. Collaboration Patterns
- Start with research phase for complex tasks
- Use 2-3 discussion rounds for most collaborative tasks
- Include domain experts for specialized areas

### 4. Performance Optimization
- Monitor backend latency and adjust routing
- Use async execution for parallel research
- Implement circuit breakers for failing backends

### 5. Security
- Validate all file paths through `_safe_path`
- Use environment variables for API keys
- Implement proper error handling and logging

## Step 7: Troubleshooting

### Common Issues:

1. **Backend Connectivity Issues**
   - Check API keys and network connectivity
   - Verify local model server is running
   - Review backend health check results

2. **Context Envelope Errors**
   - Ensure proper JSON serialization of content
   - Check TTL settings and cleanup policies
   - Verify session IDs are valid

3. **Collaboration Timeouts**
   - Adjust timeout settings in configuration
   - Monitor backend response times
   - Consider reducing discussion rounds for simple tasks

4. **Performance Issues**
   - Enable performance monitoring
   - Review backend selection logic
   - Check database performance and indexing

## Conclusion

This enhanced architecture provides:
- **Intelligent routing** based on task complexity
- **Persistent context management** compatible with Augment Code
- **Collaborative workflows** with research and discussion phases
- **Comprehensive monitoring** and analytics
- **Scalable backend selection** across local, OpenAI, and Anthropic models

The system automatically adapts to task complexity, routes agents to appropriate backends, and maintains context for seamless collaboration with Augment Code's MCP integration.