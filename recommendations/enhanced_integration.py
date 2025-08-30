"""
Integration module for enhanced agentic capabilities with existing MCP server
This module provides the bridge between the advanced agent system and the current codebase
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import logging

# Import the advanced modules
from advanced_agent_collaboration import (
    MultiAgentCollaborationOrchestrator, AdvancedAgent, AgentRole, InteractionMode,
    ADVANCED_COLLABORATION_TOOLS, handle_start_collaboration, 
    handle_get_collaboration_stats, handle_create_custom_agent
)
from enhanced_agent_memory import (
    EnhancedAgentMemory, ContextAwarePromptBuilder, enhance_agent_with_memory
)

logger = logging.getLogger(__name__)

class EnhancedMCPServerIntegration:
    """Integration layer for enhanced agentic capabilities"""
    
    def __init__(self, server):
        self.server = server
        self.collaboration_orchestrator = None
        self.enhanced_agents = {}
        self.memory_systems = {}
        
        # Initialize the enhanced systems
        self._initialize_enhanced_systems()
    
    def _initialize_enhanced_systems(self):
        """Initialize all enhanced agentic systems"""
        
        logger.info("Initializing enhanced agentic systems...")
        
        # Create collaboration orchestrator
        self.collaboration_orchestrator = MultiAgentCollaborationOrchestrator(self.server)
        
        # Enhance existing agents with memory systems
        self._enhance_existing_agents()
        
        # Add new tool handlers
        self._integrate_new_tools()
        
        logger.info("Enhanced agentic systems initialized successfully")
    
    def _enhance_existing_agents(self):
        """Enhance agents from the collaboration orchestrator with memory"""
        
        for agent_id, agent in self.collaboration_orchestrator.agents.items():
            # Create memory database path
            memory_db_path = f"./agent_memories/agent_{agent_id}.db"
            
            # Enhance with memory system
            enhanced_agent = enhance_agent_with_memory(agent, memory_db_path)
            
            # Store references
            self.enhanced_agents[agent_id] = enhanced_agent
            self.memory_systems[agent_id] = enhanced_agent.memory_system
            
            logger.info(f"Enhanced agent {agent_id} with memory system")
    
    def _integrate_new_tools(self):
        """Integrate new tool handlers into the server"""
        
        # Store reference to orchestrator in server
        self.server.collaboration_orchestrator = self.collaboration_orchestrator
        
        # The new tools will be added to get_all_tools() function
        logger.info("Integrated new collaboration tools")


def enhance_existing_agent_team_handlers(server):
    """Enhance existing agent team handlers with new capabilities"""
    
    # Store original handlers
    original_plan_and_code = server.handle_agent_team_plan_and_code if hasattr(server, 'handle_agent_team_plan_and_code') else None
    original_review_and_test = server.handle_agent_team_review_and_test if hasattr(server, 'handle_agent_team_review_and_test') else None
    
    def enhanced_handle_agent_team_plan_and_code(arguments, server):
        """Enhanced version with multi-round collaboration"""
        
        task_desc = arguments.get("task", "").strip()
        if not task_desc:
            raise ValueError("'task' is required")
        
        # Check if we should use advanced collaboration
        use_advanced = arguments.get("use_advanced_collaboration", True)
        
        if use_advanced and hasattr(server, 'collaboration_orchestrator'):
            # Use advanced multi-agent collaboration
            collaboration_args = {
                "task": f"Plan and code: {task_desc}",
                "mode": "collaboration",
                "max_rounds": 3,
                "roles": ["orchestrator", "specialist", "critic"]
            }
            
            return handle_start_collaboration(collaboration_args, server)
        
        # Fall back to original implementation
        elif original_plan_and_code:
            return original_plan_and_code(arguments, server)
        
        else:
            return "Enhanced collaboration not available, and no fallback implementation found"
    
    def enhanced_handle_agent_team_review_and_test(arguments, server):
        """Enhanced version with reflection capabilities"""
        
        diff = arguments.get("diff", "").strip()
        if not diff:
            raise ValueError("'diff' is required")
        
        context = arguments.get("context", "").strip()
        
        # Check if we should use advanced collaboration
        use_advanced = arguments.get("use_advanced_collaboration", True)
        
        if use_advanced and hasattr(server, 'collaboration_orchestrator'):
            # Use advanced multi-agent collaboration with reflection
            collaboration_args = {
                "task": f"Review and test this code change:\n\nDiff: {diff}\nContext: {context}",
                "mode": "debate",  # Use debate mode for critical review
                "max_rounds": 2,
                "roles": ["critic", "specialist", "reflection_agent"]
            }
            
            return handle_start_collaboration(collaboration_args, server)
        
        # Fall back to original implementation
        elif original_review_and_test:
            return original_review_and_test(arguments, server)
        
        else:
            return "Enhanced collaboration not available, and no fallback implementation found"
    
    # Replace handlers
    server.enhanced_handle_agent_team_plan_and_code = enhanced_handle_agent_team_plan_and_code
    server.enhanced_handle_agent_team_review_and_test = enhanced_handle_agent_team_review_and_test


def enhance_llm_analysis_tools(server):
    """Enhance LLM analysis tools with reflection and memory"""
    
    def enhanced_handle_llm_analysis_tool(tool_name, arguments, server):
        """Enhanced LLM analysis with reflection and memory integration"""
        
        code = arguments.get("code", "")
        if not code:
            return "No code provided for analysis"
        
        # Create a temporary agent for this analysis if advanced collaboration is available
        if hasattr(server, 'collaboration_orchestrator'):
            # Use multi-agent approach for complex analysis
            if tool_name == "analyze_code":
                task = f"Perform comprehensive code analysis: {code}"
                collaboration_args = {
                    "task": task,
                    "mode": "collaboration",
                    "max_rounds": 2,
                    "roles": ["specialist", "critic", "reflection_agent"]
                }
                return handle_start_collaboration(collaboration_args, server)
            
            elif tool_name == "suggest_improvements":
                task = f"Suggest improvements for this code with detailed reasoning: {code}"
                collaboration_args = {
                    "task": task,
                    "mode": "collaboration", 
                    "max_rounds": 2,
                    "roles": ["specialist", "critic"]
                }
                return handle_start_collaboration(collaboration_args, server)
        
        # Fallback to original implementation with reflection
        original_result = handle_original_llm_analysis(tool_name, arguments, server)
        
        # Add reflection if we have enhanced capabilities
        if hasattr(server, 'collaboration_orchestrator'):
            reflection_task = f"Reflect on this {tool_name} result and suggest improvements: {original_result}"
            reflection_args = {
                "task": reflection_task,
                "mode": "reflection",
                "max_rounds": 1,
                "roles": ["reflection_agent"]
            }
            
            try:
                reflection_result = handle_start_collaboration(reflection_args, server)
                return f"{original_result}\n\n--- Reflection ---\n{reflection_result}"
            except Exception as e:
                logger.error(f"Reflection failed: {e}")
                return original_result
        
        return original_result
    
    def handle_original_llm_analysis(tool_name, arguments, server):
        """Original LLM analysis implementation as fallback"""
        
        code = arguments.get("code", "")
        
        if tool_name == "analyze_code":
            analysis_type = arguments.get("analysis_type", "explanation")
            prompt = f"Analyze this code for {analysis_type}:\n\n{code}\n\nProvide detailed analysis:"
            
        elif tool_name == "suggest_improvements":
            prompt = f"Suggest specific improvements for this code:\n\n{code}\n\nFocus on readability, performance, and maintainability:"
            
        elif tool_name == "generate_tests":
            framework = arguments.get("framework", "pytest")
            prompt = f"Generate {framework} tests for this code:\n\n{code}\n\nProvide comprehensive test coverage:"
            
        else:
            prompt = f"Analyze this code:\n\n{code}"
        
        try:
            # Use the centralized LLM request method
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    server.make_llm_request_with_retry(prompt, temperature=0.1)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    # Store enhanced handler
    server.enhanced_handle_llm_analysis_tool = enhanced_handle_llm_analysis_tool


def add_enhanced_tools_to_registry():
    """Add enhanced tools to the tool registry"""
    
    enhanced_tools = ADVANCED_COLLABORATION_TOOLS + [
        {
            "name": "enhanced_code_analysis",
            "description": "Multi-agent code analysis with reflection and memory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"},
                    "analysis_type": {"type": "string", "enum": ["bugs", "optimization", "explanation", "refactor", "security"], "default": "explanation"},
                    "use_collaboration": {"type": "boolean", "description": "Use multi-agent collaboration", "default": True},
                    "collaboration_mode": {"type": "string", "enum": ["collaboration", "debate", "reflection"], "default": "collaboration"}
                },
                "required": ["code"]
            }
        },
        {
            "name": "enhanced_agent_planning",
            "description": "Advanced multi-round planning with reflection and tool discovery",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task to plan for"},
                    "constraints": {"type": "string", "description": "Planning constraints"},
                    "planning_horizon": {"type": "string", "enum": ["short", "medium", "long"], "default": "medium"},
                    "include_reflection": {"type": "boolean", "description": "Include reflection rounds", "default": True},
                    "max_iterations": {"type": "integer", "description": "Maximum planning iterations", "default": 3}
                },
                "required": ["task"]
            }
        },
        {
            "name": "agent_memory_query",
            "description": "Query agent memory systems for relevant information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent to query memory from"},
                    "query": {"type": "string", "description": "Memory search query"},
                    "memory_types": {"type": "array", "items": {"type": "string"}, "description": "Types of memory to search"},
                    "time_range": {"type": "string", "description": "Time range for search (e.g., 'last_hour', 'today', 'all_time')", "default": "all_time"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "create_learning_session",
            "description": "Create a learning session where agents iterate and improve on a task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task for agents to learn and improve on"},
                    "success_criteria": {"type": "string", "description": "How to measure success"},
                    "max_learning_rounds": {"type": "integer", "description": "Maximum learning iterations", "default": 5},
                    "feedback_mechanism": {"type": "string", "enum": ["self_reflection", "peer_review", "external_validation"], "default": "self_reflection"}
                },
                "required": ["task", "success_criteria"]
            }
        }
    ]
    
    return enhanced_tools


# New enhanced handlers
def handle_enhanced_code_analysis(arguments, server):
    """Handle enhanced multi-agent code analysis"""
    
    code = arguments.get("code", "").strip()
    if not code:
        raise ValueError("'code' is required")
    
    analysis_type = arguments.get("analysis_type", "explanation")
    use_collaboration = arguments.get("use_collaboration", True)
    collaboration_mode = arguments.get("collaboration_mode", "collaboration")
    
    if use_collaboration and hasattr(server, 'collaboration_orchestrator'):
        # Determine optimal agents for analysis type
        role_mapping = {
            "security": ["specialist", "critic", "reflection_agent"],
            "performance": ["specialist", "critic"],
            "bugs": ["critic", "specialist", "reflection_agent"],
            "optimization": ["specialist", "critic"],
            "refactor": ["specialist", "synthesizer", "critic"],
            "explanation": ["specialist", "synthesizer"]
        }
        
        roles = role_mapping.get(analysis_type, ["specialist", "critic"])
        
        task = f"Perform {analysis_type} analysis on this code:\n\n```\n{code}\n```\n\nProvide comprehensive insights."
        
        collaboration_args = {
            "task": task,
            "mode": collaboration_mode,
            "max_rounds": 2 if len(roles) > 2 else 1,
            "roles": roles
        }
        
        return handle_start_collaboration(collaboration_args, server)
    
    else:
        # Fallback to single-agent analysis
        return server.enhanced_handle_llm_analysis_tool(analysis_type, arguments, server)


def handle_enhanced_agent_planning(arguments, server):
    """Handle enhanced multi-round planning with reflection"""
    
    task = arguments.get("task", "").strip()
    if not task:
        raise ValueError("'task' is required")
    
    constraints = arguments.get("constraints", "")
    planning_horizon = arguments.get("planning_horizon", "medium")
    include_reflection = arguments.get("include_reflection", True)
    max_iterations = arguments.get("max_iterations", 3)
    
    # Build comprehensive planning task
    planning_prompt = f"""
    PLANNING TASK: {task}
    
    Constraints: {constraints if constraints else "No specific constraints"}
    Planning Horizon: {planning_horizon}
    
    Create a comprehensive plan that includes:
    1. Problem analysis and decomposition
    2. Resource requirements and dependencies
    3. Step-by-step execution strategy
    4. Risk assessment and mitigation
    5. Success metrics and validation approach
    6. Timeline and milestones
    """
    
    if hasattr(server, 'collaboration_orchestrator'):
        # Determine collaboration mode based on complexity
        mode = "collaboration"
        rounds = max_iterations
        
        # Include reflection agent if requested
        roles = ["orchestrator", "specialist"]
        if include_reflection:
            roles.append("reflection_agent")
        
        # For complex planning, add more specialized roles
        if planning_horizon == "long" or "complex" in task.lower():
            roles.extend(["critic", "synthesizer"])
            rounds = min(rounds + 1, 5)  # Cap at 5 rounds
        
        collaboration_args = {
            "task": planning_prompt,
            "mode": mode,
            "max_rounds": rounds,
            "roles": roles
        }
        
        return handle_start_collaboration(collaboration_args, server)
    
    else:
        # Fallback to single-agent planning
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    server.make_llm_request_with_retry(planning_prompt, temperature=0.3)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            return f"Planning failed: {str(e)}"


def handle_agent_memory_query(arguments, server):
    """Handle agent memory queries"""
    
    query = arguments.get("query", "").strip()
    if not query:
        raise ValueError("'query' is required")
    
    agent_id = arguments.get("agent_id")
    memory_types = arguments.get("memory_types", ["episodic", "semantic"])
    time_range = arguments.get("time_range", "all_time")
    
    if not hasattr(server, 'collaboration_orchestrator'):
        return "Advanced agent memory system not available"
    
    integration = EnhancedMCPServerIntegration(server)
    
    # If no specific agent, query all agents
    if not agent_id:
        results = {}
        for aid, memory_system in integration.memory_systems.items():
            try:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    memories = loop.run_until_complete(
                        memory_system.retrieve_memories(query, memory_types, limit=5)
                    )
                    if memories:
                        results[aid] = [mem.content for mem in memories]
                finally:
                    loop.close()
            except Exception as e:
                results[aid] = f"Query failed: {str(e)}"
        
        return {"memory_query_results": results, "query": query}
    
    # Query specific agent
    if agent_id in integration.memory_systems:
        memory_system = integration.memory_systems[agent_id]
        
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                memories = loop.run_until_complete(
                    memory_system.retrieve_memories(query, memory_types, limit=10)
                )
                
                results = []
                for memory in memories:
                    results.append({
                        "content": memory.content,
                        "importance": memory.importance,
                        "timestamp": memory.timestamp,
                        "type": memory.memory_type,
                        "tags": memory.tags
                    })
                
                return {
                    "agent_id": agent_id,
                    "query": query,
                    "results": results,
                    "total_found": len(results)
                }
            finally:
                loop.close()
        except Exception as e:
            return f"Memory query failed for agent {agent_id}: {str(e)}"
    
    else:
        return f"Agent {agent_id} not found or has no memory system"


def handle_create_learning_session(arguments, server):
    """Handle creation of learning sessions"""
    
    task = arguments.get("task", "").strip()
    success_criteria = arguments.get("success_criteria", "").strip()
    
    if not task or not success_criteria:
        raise ValueError("Both 'task' and 'success_criteria' are required")
    
    max_learning_rounds = arguments.get("max_learning_rounds", 5)
    feedback_mechanism = arguments.get("feedback_mechanism", "self_reflection")
    
    if not hasattr(server, 'collaboration_orchestrator'):
        return "Advanced learning sessions require collaboration orchestrator"
    
    # Create iterative learning task
    learning_prompt = f"""
    LEARNING TASK: {task}
    
    Success Criteria: {success_criteria}
    Feedback Mechanism: {feedback_mechanism}
    Maximum Learning Rounds: {max_learning_rounds}
    
    This is an iterative learning session. You will:
    1. Attempt the task
    2. Evaluate your performance against success criteria  
    3. Reflect on what worked and what didn't
    4. Identify specific improvements for next iteration
    5. Repeat until success criteria are met or max rounds reached
    
    Focus on continuous improvement and learning from each iteration.
    """
    
    # Use collaboration with reflection-heavy approach
    collaboration_args = {
        "task": learning_prompt,
        "mode": "reflection",  # Use reflection mode for learning
        "max_rounds": max_learning_rounds,
        "roles": ["specialist", "reflection_agent", "critic"]
    }
    
    return handle_start_collaboration(collaboration_args, server)


def update_tool_registry(existing_tools):
    """Update the existing tool registry with enhanced capabilities"""
    
    enhanced_tools = add_enhanced_tools_to_registry()
    
    # Add new tools to existing registry
    existing_tools["tools"].extend(enhanced_tools)
    
    return existing_tools


def update_tool_handlers(server):
    """Update tool handlers with enhanced versions"""
    
    # Create integration instance
    integration = EnhancedMCPServerIntegration(server)
    
    # Enhance existing handlers
    enhance_existing_agent_team_handlers(server)
    enhance_llm_analysis_tools(server)
    
    # Add new enhanced handlers
    server.handle_enhanced_code_analysis = handle_enhanced_code_analysis
    server.handle_enhanced_agent_planning = handle_enhanced_agent_planning
    server.handle_agent_memory_query = handle_agent_memory_query
    server.handle_create_learning_session = handle_create_learning_session
    
    # Update tool call registry
    enhanced_registry = {
        # Advanced collaboration tools
        "start_collaboration": (handle_start_collaboration, True),
        "get_collaboration_stats": (handle_get_collaboration_stats, True),
        "create_custom_agent": (handle_create_custom_agent, True),
        
        # Enhanced analysis tools
        "enhanced_code_analysis": (handle_enhanced_code_analysis, True),
        "enhanced_agent_planning": (handle_enhanced_agent_planning, True),
        "agent_memory_query": (handle_agent_memory_query, True),
        "create_learning_session": (handle_create_learning_session, True),
    }
    
    return enhanced_registry


# Main integration function
def integrate_enhanced_agents(server):
    """Main function to integrate all enhanced agentic capabilities"""
    
    logger.info("Starting integration of enhanced agentic capabilities...")
    
    try:
        # Create and initialize integration
        integration = EnhancedMCPServerIntegration(server)
        
        # Update tool handlers
        enhanced_handlers = update_tool_handlers(server)
        
        # Store integration reference
        server.enhanced_integration = integration
        server.enhanced_handlers = enhanced_handlers
        
        logger.info("Enhanced agentic capabilities integrated successfully!")
        
        return {
            "status": "success",
            "message": "Enhanced agentic capabilities integrated",
            "features": [
                "Multi-round agent collaboration",
                "Dynamic tool discovery and creation", 
                "Self-reflection and learning",
                "Advanced memory systems",
                "Context-aware interactions",
                "Iterative improvement loops"
            ],
            "new_tools_count": len(add_enhanced_tools_to_registry()),
            "enhanced_agents_count": len(integration.enhanced_agents)
        }
        
    except Exception as e:
        logger.error(f"Failed to integrate enhanced agentic capabilities: {e}")
        return {
            "status": "error",
            "message": f"Integration failed: {str(e)}",
            "fallback": "Original functionality preserved"
        }