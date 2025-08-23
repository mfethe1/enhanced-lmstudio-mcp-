# Enhanced MCP Tools for Augment Code Integration
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from enhanced_mcp_storage import EnhancedMCPStorage
from enhanced_agent_teams import EnhancedAgentTeamManager

def get_enhanced_mcp_tools() -> Dict[str, Any]:
    """Return enhanced MCP tool definitions optimized for Augment Code"""
    return {
        "tools": [
            # Enhanced Collaborative Agent Tools
            {
                "name": "collaborative_agent_task",
                "description": "Execute complex tasks using collaborative multi-agent teams with complexity-based routing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task description - will be analyzed for complexity and routed to appropriate agents"
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files to analyze for context (optional)"
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Task constraints and requirements (optional)"
                        },
                        "research_required": {
                            "type": "boolean",
                            "description": "Whether agents should conduct research before collaborating",
                            "default": True
                        },
                        "discussion_rounds": {
                            "type": "integer",
                            "description": "Number of collaborative discussion rounds",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5
                        },
                        "complexity_override": {
                            "type": "string",
                            "enum": ["simple", "moderate", "complex", "expert"],
                            "description": "Override automatic complexity analysis (optional)"
                        },
                        "preferred_backends": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["local", "openai", "anthropic"]},
                            "description": "Preferred LLM backends in priority order"
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply code changes to files",
                            "default": False
                        }
                    },
                    "required": ["task"]
                }
            },
            
            # Research and Analysis Tools
            {
                "name": "collaborative_research",
                "description": "Conduct collaborative research using multiple specialized research agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "research_topic": {
                            "type": "string",
                            "description": "Main research topic or question"
                        },
                        "research_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific domains to focus on (e.g., 'security', 'performance', 'architecture')"
                        },
                        "depth_level": {
                            "type": "string",
                            "enum": ["shallow", "moderate", "deep"],
                            "description": "Research depth level",
                            "default": "moderate"
                        },
                        "max_sources": {
                            "type": "integer",
                            "description": "Maximum number of sources per research query",
                            "default": 10,
                            "minimum": 5,
                            "maximum": 50
                        },
                        "synthesis_required": {
                            "type": "boolean",
                            "description": "Whether to synthesize findings across researchers",
                            "default": True
                        }
                    },
                    "required": ["research_topic"]
                }
            },
            
            # Context Management Tools
            {
                "name": "create_context_envelope",
                "description": "Create persistent context envelope for MCP integration",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier for context persistence"
                        },
                        "context_type": {
                            "type": "string",
                            "enum": ["task", "research", "collaboration", "file", "discussion"],
                            "description": "Type of context being stored"
                        },
                        "content": {
                            "type": "object",
                            "description": "Context content as JSON object"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata for the context"
                        },
                        "parent_envelope_id": {
                            "type": "string",
                            "description": "ID of parent context envelope (for chaining)"
                        },
                        "ttl_seconds": {
                            "type": "integer",
                            "description": "Time-to-live in seconds (optional)"
                        }
                    },
                    "required": ["session_id", "context_type", "content"]
                }
            },
            
            {
                "name": "get_context_chain",
                "description": "Retrieve context chain for session (MCP envelope pattern)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier"
                        },
                        "context_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by context types (optional)"
                        },
                        "include_expired": {
                            "type": "boolean",
                            "description": "Include expired contexts",
                            "default": False
                        }
                    },
                    "required": ["session_id"]
                }
            },
            
            # Agent Performance and Analytics
            {
                "name": "get_agent_analytics",
                "description": "Get comprehensive analytics for agent performance and collaboration",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Specific session to analyze (optional)"
                        },
                        "agent_roles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific agent roles"
                        },
                        "time_range_hours": {
                            "type": "integer",
                            "description": "Time range in hours for analysis",
                            "default": 24,
                            "minimum": 1,
                            "maximum": 168
                        },
                        "include_interactions": {
                            "type": "boolean",
                            "description": "Include agent-to-agent interaction details",
                            "default": True
                        }
                    }
                }
            },
            
            # Backend Health and Routing
            {
                "name": "check_backend_health",
                "description": "Check health and availability of all LLM backends",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_latency": {
                            "type": "boolean",
                            "description": "Include latency measurements",
                            "default": True
                        },
                        "test_simple_request": {
                            "type": "boolean",
                            "description": "Test with simple request to measure response time",
                            "default": False
                        }
                    }
                }
            },
            
            {
                "name": "analyze_task_complexity",
                "description": "Analyze task complexity and suggest optimal agent routing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Task to analyze for complexity"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for analysis"
                        },
                        "suggest_team": {
                            "type": "boolean",
                            "description": "Suggest optimal agent team composition",
                            "default": True
                        }
                    },
                    "required": ["task_description"]
                }
            },
            
            # Collaborative Artifacts Management
            {
                "name": "create_collaborative_artifact",
                "description": "Create shared artifact during collaboration (code, docs, analysis)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Collaboration session ID"
                        },
                        "artifact_type": {
                            "type": "string",
                            "enum": ["research_summary", "code_patch", "analysis", "plan", "review", "documentation"],
                            "description": "Type of artifact being created"
                        },
                        "title": {
                            "type": "string",
                            "description": "Artifact title"
                        },
                        "content": {
                            "type": "string",
                            "description": "Artifact content"
                        },
                        "contributors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of contributing agents"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "parent_artifact_id": {
                            "type": "string",
                            "description": "Parent artifact ID for versioning"
                        }
                    },
                    "required": ["session_id", "artifact_type", "title", "content", "contributors"]
                }
            },
            
            {
                "name": "get_session_artifacts",
                "description": "Get all artifacts created during a collaboration session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Collaboration session ID"
                        },
                        "artifact_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by artifact types"
                        },
                        "include_versions": {
                            "type": "boolean",
                            "description": "Include all versions of artifacts",
                            "default": False
                        }
                    },
                    "required": ["session_id"]
                }
            },
            
            # Context Lineage for Augment Code
            {
                "name": "log_context_change",
                "description": "Log context changes for Augment Code lineage tracking",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path that was changed"
                        },
                        "change_type": {
                            "type": "string",
                            "enum": ["create", "modify", "delete", "rename"],
                            "description": "Type of change made"
                        },
                        "agent_responsible": {
                            "type": "string",
                            "description": "Agent that made the change"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for tracking"
                        },
                        "context_before": {
                            "type": "object",
                            "description": "Context snapshot before change"
                        },
                        "context_after": {
                            "type": "object",
                            "description": "Context snapshot after change"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why the change was made"
                        }
                    },
                    "required": ["file_path", "change_type", "agent_responsible", "session_id", "reasoning"]
                }
            },
            
            # Enhanced Code Analysis Tools
            {
                "name": "collaborative_code_review",
                "description": "Multi-agent collaborative code review with specialized expertise",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code_content": {
                            "type": "string",
                            "description": "Code to review"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "File path for context"
                        },
                        "review_aspects": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["security", "performance", "maintainability", "testing", "architecture", "style"]},
                            "description": "Aspects to focus review on",
                            "default": ["security", "maintainability", "testing"]
                        },
                        "include_suggestions": {
                            "type": "boolean",
                            "description": "Include improvement suggestions",
                            "default": True
                        },
                        "generate_tests": {
                            "type": "boolean",
                            "description": "Generate test cases for the code",
                            "default": False
                        }
                    },
                    "required": ["code_content"]
                }
            },
            
            {
                "name": "architectural_analysis",
                "description": "Collaborative architectural analysis by domain experts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "codebase_path": {
                            "type": "string",
                            "description": "Path to codebase root"
                        },
                        "analysis_scope": {
                            "type": "string",
                            "enum": ["file", "module", "package", "full"],
                            "description": "Scope of analysis",
                            "default": "module"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas to focus on (e.g., 'scalability', 'security', 'maintainability')"
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "description": "Include architectural recommendations",
                            "default": True
                        },
                        "complexity_analysis": {
                            "type": "boolean",
                            "description": "Include complexity metrics and analysis",
                            "default": True
                        }
                    },
                    "required": ["codebase_path"]
                }
            },
            
            # Session Management
            {
                "name": "start_collaboration_session",
                "description": "Start new collaborative agent session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_type": {
                            "type": "string",
                            "enum": ["task_solving", "research", "code_review", "architecture", "planning"],
                            "description": "Type of collaboration session"
                        },
                        "participants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent roles to include in session"
                        },
                        "initial_context": {
                            "type": "object",
                            "description": "Initial context for the session"
                        },
                        "session_metadata": {
                            "type": "object",
                            "description": "Additional session metadata"
                        }
                    },
                    "required": ["session_type"]
                }
            },
            
            {
                "name": "get_active_sessions",
                "description": "Get list of active collaboration sessions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_completed": {
                            "type": "boolean",
                            "description": "Include completed sessions",
                            "default": False
                        },
                        "session_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by session types"
                        }
                    }
                }
            },
            
            # Cleanup and Maintenance
            {
                "name": "cleanup_expired_contexts",
                "description": "Clean up expired context envelopes and optimize storage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dry_run": {
                            "type": "boolean",
                            "description": "Preview what would be cleaned up without deleting",
                            "default": True
                        },
                        "older_than_hours": {
                            "type": "integer",
                            "description": "Clean contexts older than specified hours (overrides TTL)",
                            "minimum": 1
                        }
                    }
                }
            }
        ]
    }

# Enhanced Tool Handlers
class EnhancedMCPToolHandlers:
    """Enhanced tool handlers for the new MCP tools"""
    
    def __init__(self, storage: EnhancedMCPStorage):
        self.storage = storage
        self.team_manager = EnhancedAgentTeamManager(storage)
    
    async def handle_collaborative_agent_task(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaborative agent task with enhanced routing"""
        try:
            result = await self.team_manager.handle_collaborative_task(arguments)
            
            # Format result for MCP
            return {
                "success": True,
                "session_id": result.get("session_id"),
                "phases_completed": result.get("phases_completed", []),
                "execution_summary": {
                    "duration_minutes": result.get("execution_time_minutes", 0),
                    "contributors": result.get("contributors", []),
                    "artifacts_created": len(result.get("final_recommendations", {})),
                },
                "recommendations": result.get("final_recommendations", {}),
                "research_summary": result.get("research_artifacts", {}).get("synthesis", {}),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def handle_create_context_envelope(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create context envelope with MCP compatibility"""
        try:
            envelope_id = self.storage.create_context_envelope(
                session_id=arguments["session_id"],
                context_type=arguments["context_type"],
                content=arguments["content"],
                metadata=arguments.get("metadata", {}),
                parent_envelope_id=arguments.get("parent_envelope_id"),
                ttl_seconds=arguments.get("ttl_seconds")
            )
            
            return {
                "success": True,
                "envelope_id": envelope_id,
                "session_id": arguments["session_id"],
                "context_type": arguments["context_type"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_get_context_chain(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get context chain for MCP envelope pattern"""
        try:
            envelopes = self.storage.get_session_context_chain(arguments["session_id"])
            
            # Filter by context types if specified
            if arguments.get("context_types"):
                envelopes = [env for env in envelopes if env.context_type in arguments["context_types"]]
            
            # Filter expired if not requested
            if not arguments.get("include_expired", False):
                current_time = time.time()
                envelopes = [
                    env for env in envelopes 
                    if env.ttl_seconds is None or (env.created_at + env.ttl_seconds) > current_time
                ]
            
            return {
                "success": True,
                "session_id": arguments["session_id"],
                "envelope_count": len(envelopes),
                "envelopes": [
                    {
                        "envelope_id": env.envelope_id,
                        "context_type": env.context_type,
                        "created_at": env.created_at,
                        "updated_at": env.updated_at,
                        "access_count": env.access_count,
                        "content_preview": str(env.content)[:200] + "..." if len(str(env.content)) > 200 else str(env.content)
                    }
                    for env in envelopes
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_get_agent_analytics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive agent analytics"""
        try:
            analytics = {}
            
            if arguments.get("session_id"):
                # Get specific session analytics
                analytics = self.storage.get_session_analytics(arguments["session_id"])
            else:
                # Get general analytics
                # This would be implemented to aggregate across multiple sessions
                analytics = {
                    "message": "General analytics not yet implemented",
                    "time_range_hours": arguments.get("time_range_hours", 24)
                }
            
            return {
                "success": True,
                "analytics": analytics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_check_backend_health(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of all LLM backends"""
        try:
            from enhanced_agent_architecture import EnhancedAgentRouter
            
            router = EnhancedAgentRouter(self.storage)
            health_status = router.backend_health
            
            return {
                "success": True,
                "backend_health": health_status,
                "timestamp": time.time(),
                "healthy_backends": [
                    backend for backend, status in health_status.items() 
                    if status.get("available", False)
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_cleanup_expired_contexts(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up expired context envelopes"""
        try:
            if arguments.get("dry_run", True):
                # Count what would be cleaned up
                count = 0  # Would implement counting logic
                return {
                    "success": True,
                    "dry_run": True,
                    "would_cleanup": count,
                    "message": f"Would clean up {count} expired contexts"
                }
            else:
                # Actually clean up
                count = self.storage.cleanup_expired_contexts()
                return {
                    "success": True,
                    "dry_run": False,
                    "cleaned_up": count,
                    "message": f"Cleaned up {count} expired contexts"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }