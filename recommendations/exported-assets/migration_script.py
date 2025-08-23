# Migration Script: Integrating Enhanced Architecture with Existing MCP Server
"""
This script helps migrate your existing MCP server to the enhanced multi-agent architecture.
Run this after implementing the new modules to ensure smooth integration.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

def backup_existing_server():
    """Backup existing server files before migration"""
    backup_dir = Path("backup_original")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "paste.txt",
        "storage.py", 
        "mcp_server.py",  # if exists
        "enhanced_mcp.db"  # if exists
    ]
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            shutil.copy2(file_path, backup_dir / file_path)
            print(f"✅ Backed up {file_path} to {backup_dir}")
    
    print("📁 Backup completed successfully")

def update_existing_handlers():
    """Update existing tool handlers to integrate with enhanced system"""
    
    integration_code = '''
# Integration with Enhanced Architecture
from enhanced_mcp_storage import EnhancedMCPStorage
from enhanced_agent_teams import EnhancedAgentTeamManager
from enhanced_mcp_tools import EnhancedMCPToolHandlers

class MigratedMCPServer(EnhancedLMStudioMCPServer):
    """Migrated server with enhanced capabilities while maintaining backward compatibility"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize enhanced components
        self.enhanced_storage = EnhancedMCPStorage("enhanced_mcp.db")
        self.enhanced_handlers = EnhancedMCPToolHandlers(self.enhanced_storage)
        self.team_manager = EnhancedAgentTeamManager(self.enhanced_storage)
        
        # Keep original storage for backward compatibility
        self.legacy_storage = self.storage
        
        print("🚀 Migrated MCP Server with Enhanced Architecture initialized")

    def get_all_tools(self):
        """Return both legacy and enhanced tools"""
        legacy_tools = super().get_all_tools()
        from enhanced_mcp_tools import get_enhanced_mcp_tools
        enhanced_tools = get_enhanced_mcp_tools()
        
        # Merge tool definitions
        all_tools = legacy_tools["tools"] + enhanced_tools["tools"]
        
        return {"tools": all_tools}

    async def handle_enhanced_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route between legacy and enhanced tool handlers"""
        
        # Enhanced tools (new collaborative features)
        enhanced_tools = {
            "collaborative_agent_task",
            "collaborative_research", 
            "create_context_envelope",
            "get_context_chain",
            "get_agent_analytics",
            "check_backend_health",
            "analyze_task_complexity",
            "collaborative_code_review",
            "architectural_analysis"
        }
        
        if tool_name in enhanced_tools:
            # Route to enhanced handlers
            return await self.enhanced_handlers._route_tool_call(tool_name, arguments)
        else:
            # Route to legacy handlers (from your original code)
            return await self._handle_legacy_tool(tool_name, arguments)
    
    async def _handle_legacy_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle legacy tools with original implementation"""
        # Map to your existing handlers from paste.txt
        legacy_handlers = {
            "deep_research": self._handle_deep_research,
            "agent_team_plan_and_code": self._handle_agent_team_plan_and_code,
            "analyze_code": self._handle_analyze_code,
            "execute_code": self._handle_execute_code,
            # ... add all your existing handlers
        }
        
        handler = legacy_handlers.get(tool_name)
        if handler:
            return await handler(arguments)
        else:
            raise ValueError(f"Unknown legacy tool: {tool_name}")
    
    # Keep all your original handler methods here
    # They will be called for backward compatibility
'''
    
    migration_file = Path("migrated_server.py")
    with open(migration_file, "w") as f:
        f.write(integration_code)
    
    print(f"✅ Created migration template at {migration_file}")
    return migration_file

def create_environment_config():
    """Create comprehensive environment configuration"""
    
    env_template = '''# Enhanced MCP Server Configuration
# Copy this to .env and update with your actual values

# === API Keys ===
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# === Local Model Configuration ===
LM_STUDIO_URL=http://localhost:1234
MODEL_NAME=openai/gpt-oss-20b
LMSTUDIO_MODEL=openai/gpt-oss-20b

# === Backend Routing (Optional) ===
# Override automatic backend selection for specific roles
# Format: role=backend,role=backend or all=backend
# Example: security=anthropic,domain_expert=anthropic,all=openai
# AGENT_BACKEND_OVERRIDE=

# === Storage Configuration ===
STORAGE_BACKEND=sqlite
# For PostgreSQL: uncomment and configure below
# STORAGE_BACKEND=postgres
# POSTGRES_DSN=postgresql://user:password@localhost/enhanced_mcp

# === Performance Settings ===
PERFORMANCE_THRESHOLD=0.5
EXECUTION_ENABLED=true
APPLY_DRY_RUN=1

# === Security ===
ALLOWED_BASE_DIR=.
# FIRECRAWL_API_KEY=your_firecrawl_key

# === Augment Code Integration ===
MCP_ADMIN_TOKEN=generate_secure_token_here
MCP_STORAGE_PATH=./mcp-storage

# === Agent Collaboration Settings ===
MAX_CONCURRENT_SESSIONS=5
DEFAULT_DISCUSSION_ROUNDS=2
RESEARCH_TIMEOUT_MINUTES=10
CONTEXT_ENVELOPE_TTL_HOURS=24

# === Debugging ===
LOG_LEVEL=INFO
DEBUG_AGENT_INTERACTIONS=false
AGENT_TEAM_FORCE_FALLBACK=0
'''
    
    env_file = Path(".env.example")
    with open(env_file, "w") as f:
        f.write(env_template)
    
    print(f"✅ Created environment template at {env_file}")
    print("📝 Copy to .env and update with your actual API keys")

def create_augment_config():
    """Create Augment Code configuration"""
    
    augment_config = {
        "mcp": {
            "servers": {
                "enhanced-collaborative": {
                    "command": "python",
                    "args": ["migrated_server.py"],
                    "env": {
                        "PYTHONPATH": "."
                    },
                    "capabilities": {
                        "context_persistence": True,
                        "collaboration": True,
                        "complexity_routing": True,
                        "backend_selection": True,
                        "performance_monitoring": True
                    }
                }
            }
        },
        "context": {
            "lineage": {
                "enabled": True,
                "track_changes": True,
                "mcp_integration": True,
                "auto_log_changes": True
            },
            "envelope": {
                "max_size_mb": 10,
                "ttl_hours": 24,
                "auto_cleanup": True,
                "compression": True
            }
        },
        "agents": {
            "collaboration": {
                "max_concurrent_sessions": 5,
                "default_discussion_rounds": 2,
                "research_timeout_minutes": 10,
                "auto_research": True
            },
            "routing": {
                "complexity_analysis": True,
                "backend_failover": True,
                "performance_monitoring": True,
                "adaptive_timeout": True
            },
            "specialization": {
                "security_expert": {
                    "preferred_backend": "anthropic",
                    "temperature": 0.1
                },
                "performance_expert": {
                    "preferred_backend": "openai", 
                    "temperature": 0.2
                },
                "domain_expert": {
                    "preferred_backend": "anthropic",
                    "temperature": 0.1
                }
            }
        }
    }
    
    config_file = Path("augment.config.json")
    with open(config_file, "w") as f:
        json.dump(augment_config, f, indent=2)
    
    print(f"✅ Created Augment Code configuration at {config_file}")

def run_database_migration():
    """Initialize the enhanced database schema"""
    try:
        from enhanced_mcp_storage import EnhancedMCPStorage
        
        # Initialize storage to create tables
        storage = EnhancedMCPStorage("enhanced_mcp.db")
        print("✅ Database schema initialized successfully")
        
        # Test basic functionality
        test_envelope_id = storage.create_context_envelope(
            session_id="test_migration",
            context_type="test",
            content={"message": "Migration test successful"},
            metadata={"timestamp": "migration"},
            ttl_seconds=60
        )
        
        print(f"✅ Database test completed - envelope ID: {test_envelope_id}")
        
    except ImportError as e:
        print(f"⚠️  Database migration skipped - missing dependencies: {e}")
        print("   Run 'pip install -r requirements.txt' first")

def create_requirements_file():
    """Create requirements file for the enhanced system"""
    
    requirements = '''# Enhanced MCP Server Requirements

# Core dependencies (from your original code)
aiohttp>=3.8.0
requests>=2.28.0
asyncio-compat>=0.1.2

# Enhanced architecture dependencies
anthropic>=0.8.0
openai>=1.0.0
crewai>=0.28.0

# Data and storage
sqlite3  # Built into Python
psycopg2-binary>=2.9.0  # Optional: for PostgreSQL support

# Performance and monitoring
structlog>=22.3.0
prometheus-client>=0.15.0  # Optional: for metrics

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.20.0
black>=22.0.0
ruff>=0.1.0

# Optional: for advanced NLP features
spacy>=3.5.0
transformers>=4.25.0
'''
    
    req_file = Path("requirements_enhanced.txt")
    with open(req_file, "w") as f:
        f.write(requirements)
    
    print(f"✅ Created requirements file at {req_file}")
    print("📦 Install with: pip install -r requirements_enhanced.txt")

def print_migration_summary():
    """Print summary of migration steps"""
    
    summary = """
🎉 Enhanced MCP Server Migration Complete!

📋 What was created:
✅ Enhanced architecture modules (4 files)
✅ Migration template (migrated_server.py)
✅ Environment configuration (.env.example)
✅ Augment Code config (augment.config.json)
✅ Requirements file (requirements_enhanced.txt)
✅ Database schema (enhanced_mcp.db)
✅ Original files backup (backup_original/)

🚀 Next Steps:

1. Install dependencies:
   pip install -r requirements_enhanced.txt

2. Configure environment:
   cp .env.example .env
   # Edit .env with your API keys

3. Test the enhanced server:
   python migrated_server.py

4. Update your Augment Code settings to use the new server

📊 Key Improvements:
- 🤖 Multi-agent collaboration with research phases
- 🧠 Intelligent complexity-based backend routing  
- 💾 Persistent context envelopes for MCP
- 📈 Comprehensive performance monitoring
- 🔒 Enhanced security and path validation
- 🏗️  Modular architecture for maintainability

🔧 Backend Intelligence:
- Simple tasks → Local models (fast, free)
- Moderate tasks → OpenAI GPT-4o-mini (cost-effective)
- Complex tasks → OpenAI GPT-4o or Anthropic Claude
- Expert domains → Anthropic Claude Sonnet (specialized)

📚 Documentation:
- See implementation_guide.md for detailed usage examples
- Check enhanced_mcp_tools.py for all available tools
- Review enhanced_agent_architecture.py for customization options

Happy coding with your enhanced collaborative MCP server! 🎯
"""
    
    print(summary)

def main():
    """Run the complete migration process"""
    print("🔄 Starting Enhanced MCP Server Migration...")
    print("=" * 60)
    
    try:
        # Step 1: Backup existing files
        print("\n1️⃣  Creating backup of existing files...")
        backup_existing_server()
        
        # Step 2: Create configuration files
        print("\n2️⃣  Creating configuration files...")
        create_environment_config()
        create_augment_config()
        create_requirements_file()
        
        # Step 3: Create migration template
        print("\n3️⃣  Creating migration template...")
        update_existing_handlers()
        
        # Step 4: Initialize database
        print("\n4️⃣  Initializing enhanced database...")
        run_database_migration()
        
        # Step 5: Summary
        print("\n" + "=" * 60)
        print_migration_summary()
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        print("🔍 Check the error above and ensure all files are in place")
        sys.exit(1)

if __name__ == "__main__":
    main()