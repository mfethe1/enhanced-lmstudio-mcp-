# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Build and Run
```powershell
# Start the MCP server
python server.py

# Quick test of server functionality
python quick_test_server.py

# Run specific validation tools
python validate_critical_tools.py
python test_mcp_protocol_compliance.py
```

### Testing
```powershell
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_mcp_basic.py

# Run with verbose output
python -m pytest -v

# Run a single test function
python -m pytest tests/test_mcp_basic.py::test_function_name

# Run tests matching a pattern
python -m pytest -k "test_agent"

# Run tests with coverage (if coverage installed)
python -m pytest --cov=.

# Run integration tests
python test_reliability_improvements.py
python test_mcp_server.py
```

### Linting and Code Quality
```powershell
# Format code with black
python -m black .

# Run linting with flake8
python -m flake8 server.py

# Check all Python files
python -m flake8 .

# Type checking (if mypy installed)
python -m mypy --ignore-missing-imports .
```

### Docker Operations
```powershell
# Build Docker image
docker build -t enhanced-lmstudio-mcp .

# Run with Docker Compose
docker-compose up --build

# Run container directly
docker run -e REMOTE_ENABLED=true -e MCP_REMOTE_TOKEN=changeme -p 8787:8787 enhanced-lmstudio-mcp
```

### Environment Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy and edit)
cp .env.example .secrets/.env.local

# Load environment from PowerShell
.\scripts\set_env_from_file.ps1
```

## High-Level Architecture

### Core System Design
The Enhanced LM Studio MCP Server is a Model Context Protocol implementation that bridges coding agents with LM Studio and other LLM providers. It follows a modular, event-driven architecture with several key subsystems:

#### 1. MCP Protocol Layer (`server.py`)
- **Singleton Server Pattern**: Uses `get_server_singleton()` to ensure single instance with proper lifecycle management
- **Tool Discovery**: Dynamically registers 63+ tools through handler modules
- **Request Routing**: Implements JSON-RPC 2.0 protocol with full MCP compliance
- **Async Execution**: Centralized async loop via `core/executor.py` (AsyncExecutor) eliminates event loop conflicts

#### 2. Multi-Agent Orchestration System

##### Advanced Strands Architecture (`strands/`)
- **TeamOrchestrator**: Coordinates multiple specialized agents for complex tasks
- **Dynamic Team Assembly**: Automatically creates expert teams based on task requirements
- **Hybrid Retrieval System**: Combines vector search, knowledge graphs, and memory
- **Business Frameworks**: Built-in McKinsey/BCG/Goldman Sachs analytical frameworks
- **Query Routing**: AI-powered classification routes queries to optimal strategies

##### CrewAI Integration (`agents/`)
- **CodingCrewSystem**: Multi-stage pipeline with specialized agents
- **Agent Pool**: Reuses agent instances to avoid per-request overhead
- **Pipeline Stages**: requirements_analyst → architect → coder → test_engineer → performance_tuner → security_auditor
- **Configuration**: Agent specs in `config/agents.yaml`, tasks in `config/tasks.yaml`

#### 3. Agentic Task Management (`agentic_handlers.py`, `task_manager.py`)
- **Background Execution**: Tasks run autonomously with job IDs
- **Persistent State**: Tasks survive server restarts with SQLite/PostgreSQL storage
- **Activity Logging**: Complete audit trail of agent actions
- **Status Monitoring**: Real-time progress tracking and result retrieval

#### 4. Tool Handlers (`handlers/`)
Modular tool implementations organized by functionality:
- **code_tools.py**: Code analysis, execution, debugging
- **file_tools.py**: File system operations with safety boundaries
- **memory_tools.py**: Persistent context storage and retrieval
- **agent_teams.py**: CrewAI team coordination tools
- **thinking_tools.py**: Sequential reasoning capabilities
- **coding_agent_tools.py**: Advanced development workflows

#### 5. Backend Adapters
- **LM Studio**: Primary local model interface with fallback chains
- **OpenAI/Anthropic**: Cloud provider integration with rate limiting
- **Amazon Bedrock**: Enterprise Claude access via AWS
- **Circuit Breakers**: Fault tolerance for all external services

#### 6. Production Hardening (`production_config.py`)
- **Environment-Based Config**: Production settings via environment variables
- **Model Monitoring**: Real-time tracking of LM Studio availability
- **Performance Alerts**: Threshold-based alerting system
- **Singleton Enforcement**: Prevents multiple orchestrator instances
- **Graceful Shutdown**: Proper cleanup of background threads

### Key Architectural Patterns

#### Adaptive Routing (`core/adaptive_router.py`)
- Learns from performance metrics to optimize tool and backend selection
- Tracks P50/P99 latency and success rates
- Biases future routing decisions based on past performance
- Falls back to heuristics when ML routing unavailable

#### Storage Abstraction (`storage.py`, `storage_postgres.py`)
- Pluggable backend supporting SQLite and PostgreSQL
- Migration support for schema evolution
- Thread-safe operations with connection pooling
- Automatic cleanup of stale data

#### Remote Server Support (`remote_server.py`)
- HTTP/WebSocket JSON-RPC endpoints
- Bearer token authentication
- Rate limiting and burst protection
- Health check endpoints for monitoring

### Critical Environment Variables

```bash
# LM Studio Configuration
LM_STUDIO_URL=http://localhost:1234
LMSTUDIO_MODEL=deepseek-r1-distill-qwen-7b

# Feature Toggles
EXPOSE_PUBLIC_ONLY=1          # Production: show only essential tools
PROACTIVE_RESEARCH_ENABLED=0  # Disable background research
CREWAI_ENABLED=true           # Enable multi-agent teams
ADAPTIVE_ROUTER_ENABLED=true  # Use ML-powered routing

# Performance Tuning
HTTP_CONNECT_TIMEOUT=2
HTTP_READ_TIMEOUT_SIMPLE=8
HTTP_READ_TIMEOUT_COMPLEX=45
ASYNC_EXECUTOR_TIMEOUT=60
ROUTER_BG_TIMEOUT_SEC=240

# Production Settings
LOG_LEVEL=WARNING
CIRCUIT_BREAKER_ENABLED=1
MODEL_MONITOR_INTERVAL=300
PERFORMANCE_THRESHOLD=0.2

# Storage Configuration
STORAGE_BACKEND=sqlite        # or postgres
POSTGRES_DSN=postgresql://user:pass@host:5432/db

# Remote Access (optional)
REMOTE_ENABLED=true
REMOTE_BIND=0.0.0.0:8787
MCP_REMOTE_TOKEN=changeme
```

### Tool Categories and Usage

The server provides 63+ tools across these categories:

1. **Agentic Management**: Background task orchestration
2. **Sequential Thinking**: Structured reasoning chains
3. **Code Analysis**: Multi-dimensional code inspection
4. **Execution & Testing**: Safe code/test runners
5. **File Operations**: Controlled filesystem access
6. **Memory Management**: Persistent context storage
7. **Debugging Tools**: Advanced error analysis
8. **Team Coordination**: Multi-agent pipelines
9. **Research Tools**: Deep web research capabilities
10. **Business Analysis**: Strategic frameworks

Tools are dynamically discovered and registered at startup, with production mode (`EXPOSE_PUBLIC_ONLY=1`) filtering to essential tools only.

## Development Workflow Notes

### Testing Strategy
- Unit tests in `tests/` with pytest fixtures
- Integration tests validate MCP protocol compliance
- Reliability tests (`test_reliability_improvements.py`) verify production hardening
- Use `test_mcp_protocol_compliance.py` to validate protocol implementation

### Adding New Tools
1. Create handler module in `handlers/`
2. Implement tool functions with proper error handling
3. Register in appropriate handler's tool list
4. Add tests in `tests/`
5. Update tool count in README if needed

### Debugging Tips
- Set `LOG_LEVEL=DEBUG` for verbose output
- Use `debug_lmstudio_requests.py` for LM Studio connectivity issues
- Check `audit_results.json` for tool execution history
- Monitor `enhanced_mcp_storage.db` for persistent state

### Performance Optimization
- Enable circuit breakers for external services
- Use adaptive router for intelligent backend selection
- Configure appropriate timeouts per operation type
- Monitor with metrics endpoints when enabled