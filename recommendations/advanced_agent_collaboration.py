"""
Advanced Multi-Agent Collaboration System with Dynamic Tool Selection and Reflection
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import logging
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    TOOL_CREATOR = "tool_creator"
    REFLECTION_AGENT = "reflection_agent"
    CONTEXT_MANAGER = "context_manager"

class InteractionMode(Enum):
    DEBATE = "debate"
    COLLABORATION = "collaboration"
    COMPETITION = "competition"
    REFLECTION = "reflection"
    CONSENSUS_BUILDING = "consensus_building"

class ToolDiscoveryResult(Enum):
    FOUND = "found"
    CREATED = "created"
    COMPOSED = "composed"
    FAILED = "failed"

@dataclass
class AgentMessage:
    agent_id: str
    role: AgentRole
    content: str
    timestamp: float
    message_type: str
    metadata: Dict[str, Any]
    confidence: float
    reasoning: str
    tool_calls: List[Dict[str, Any]]
    context_refs: List[str]

@dataclass
class ReflectionResult:
    agent_id: str
    original_message_id: str
    critique: str
    improvement_suggestions: List[str]
    confidence_adjustment: float
    should_retry: bool
    alternative_approaches: List[str]

@dataclass
class ToolCapability:
    tool_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    success_rate: float
    avg_execution_time: float
    complexity_rating: int
    prerequisites: List[str]
    composition_potential: List[str]

@dataclass
class CollaborationSession:
    session_id: str
    task_description: str
    participants: List[str]
    mode: InteractionMode
    max_rounds: int
    current_round: int
    messages: List[AgentMessage]
    shared_context: Dict[str, Any]
    consensus_threshold: float
    start_time: float
    end_time: Optional[float]

class DynamicToolRegistry:
    """Registry for dynamically discovered and created tools"""
    
    def __init__(self):
        self.tools: Dict[str, ToolCapability] = {}
        self.tool_usage_stats = defaultdict(lambda: {"uses": 0, "success": 0, "avg_time": 0.0})
        self.tool_compositions: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
        
    def register_tool(self, tool: ToolCapability):
        with self.lock:
            self.tools[tool.tool_id] = tool
            logger.info(f"Registered tool: {tool.name}")
    
    def discover_tool_for_task(self, task_description: str, context: Dict[str, Any]) -> Optional[ToolCapability]:
        """Use semantic similarity to find best tool for task"""
        best_match = None
        best_score = 0.0
        
        with self.lock:
            for tool in self.tools.values():
                score = self._calculate_tool_relevance(tool, task_description, context)
                if score > best_score:
                    best_score = score
                    best_match = tool
        
        return best_match if best_score > 0.5 else None
    
    def _calculate_tool_relevance(self, tool: ToolCapability, task: str, context: Dict[str, Any]) -> float:
        """Calculate relevance score between tool and task"""
        # Simple keyword-based scoring (in production, use embeddings)
        task_words = set(task.lower().split())
        tool_words = set((tool.name + " " + tool.description).lower().split())
        
        intersection = task_words.intersection(tool_words)
        union = task_words.union(tool_words)
        
        base_score = len(intersection) / len(union) if union else 0.0
        
        # Boost score based on success rate and usage
        stats = self.tool_usage_stats[tool.tool_id]
        success_factor = stats["success"] / max(stats["uses"], 1)
        
        return base_score * 0.7 + success_factor * 0.3
    
    def compose_tool(self, component_tools: List[str], new_tool_name: str, composition_logic: str) -> ToolCapability:
        """Create a new tool by composing existing tools"""
        new_tool_id = f"composed_{uuid.uuid4().hex[:8]}"
        
        # Calculate combined complexity
        combined_complexity = sum(self.tools[tid].complexity_rating for tid in component_tools if tid in self.tools)
        
        # Create composite tool description
        component_descriptions = [self.tools[tid].description for tid in component_tools if tid in self.tools]
        composite_description = f"Composed tool combining: {', '.join(component_descriptions)}"
        
        new_tool = ToolCapability(
            tool_id=new_tool_id,
            name=new_tool_name,
            description=composite_description,
            parameters={},
            success_rate=0.5,  # Initial estimate
            avg_execution_time=0.0,
            complexity_rating=min(combined_complexity, 10),
            prerequisites=component_tools,
            composition_potential=[]
        )
        
        self.register_tool(new_tool)
        self.tool_compositions[new_tool_id] = component_tools
        
        return new_tool

class AdvancedAgent:
    """Enhanced agent with reflection, tool discovery, and collaboration capabilities"""
    
    def __init__(self, agent_id: str, role: AgentRole, server, tool_registry: DynamicToolRegistry):
        self.agent_id = agent_id
        self.role = role
        self.server = server
        self.tool_registry = tool_registry
        self.memory = deque(maxlen=50)  # Recent conversation memory
        self.long_term_memory: Dict[str, Any] = {}
        self.reflection_history: List[ReflectionResult] = []
        self.expertise_areas: Set[str] = set()
        self.confidence_threshold = 0.7
        self.learning_rate = 0.1
        
        # Performance tracking
        self.task_success_rate = 0.5
        self.avg_response_time = 0.0
        self.collaboration_effectiveness = 0.5
        
    async def process_message(self, message: str, context: Dict[str, Any], 
                            collaboration_session: Optional[CollaborationSession] = None) -> AgentMessage:
        """Process a message with full context awareness and tool discovery"""
        
        start_time = time.time()
        
        # Step 1: Analyze the task and discover/select tools
        required_tools = await self._discover_required_tools(message, context)
        
        # Step 2: Generate initial response
        initial_response = await self._generate_response(message, context, required_tools)
        
        # Step 3: Self-reflect on the response
        if self.role != AgentRole.REFLECTION_AGENT:
            reflection = await self._self_reflect(initial_response, context)
            if reflection.should_retry:
                initial_response = await self._generate_response(
                    message, context, required_tools, reflection
                )
        
        # Step 4: Consider collaboration context
        if collaboration_session:
            initial_response = await self._incorporate_collaboration_context(
                initial_response, collaboration_session
            )
        
        # Step 5: Update learning and memory
        await self._update_learning(message, initial_response, time.time() - start_time)
        
        return AgentMessage(
            agent_id=self.agent_id,
            role=self.role,
            content=initial_response,
            timestamp=time.time(),
            message_type="response",
            metadata={"tools_used": [tool.tool_id for tool in required_tools]},
            confidence=self._calculate_confidence(initial_response, context),
            reasoning=await self._explain_reasoning(message, initial_response, required_tools),
            tool_calls=[{"tool_id": t.tool_id, "params": {}} for t in required_tools],
            context_refs=list(context.keys())
        )
    
    async def _discover_required_tools(self, task: str, context: Dict[str, Any]) -> List[ToolCapability]:
        """Discover and select tools needed for the task"""
        required_tools = []
        
        # Try to find existing tools
        primary_tool = self.tool_registry.discover_tool_for_task(task, context)
        if primary_tool:
            required_tools.append(primary_tool)
        
        # If no suitable tool found and we're a tool creator, create one
        if not required_tools and self.role == AgentRole.TOOL_CREATOR:
            new_tool = await self._create_tool_for_task(task, context)
            if new_tool:
                required_tools.append(new_tool)
        
        # Consider tool composition if we have multiple relevant tools
        if len(required_tools) > 1:
            composed_tool = self.tool_registry.compose_tool(
                [t.tool_id for t in required_tools],
                f"composite_for_{hashlib.md5(task.encode()).hexdigest()[:8]}",
                "Sequential execution of component tools"
            )
            required_tools = [composed_tool]
        
        return required_tools
    
    async def _create_tool_for_task(self, task: str, context: Dict[str, Any]) -> Optional[ToolCapability]:
        """Create a new tool dynamically for the given task"""
        
        # Use LLM to analyze task and generate tool specification
        tool_creation_prompt = f"""
        Analyze this task and create a tool specification:
        Task: {task}
        Context: {json.dumps(context, indent=2)}
        
        Generate a JSON specification for a new tool that could solve this task:
        {{
            "name": "tool_name",
            "description": "what the tool does",
            "parameters": {{"param_name": "param_type"}},
            "complexity_rating": 1-10,
            "implementation_approach": "how to implement"
        }}
        """
        
        try:
            response = await self.server.make_llm_request_with_retry(tool_creation_prompt, temperature=0.3)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                tool_spec = json.loads(json_match.group())
                
                new_tool = ToolCapability(
                    tool_id=f"created_{uuid.uuid4().hex[:8]}",
                    name=tool_spec.get("name", "auto_created_tool"),
                    description=tool_spec.get("description", "Automatically created tool"),
                    parameters=tool_spec.get("parameters", {}),
                    success_rate=0.3,  # Conservative initial estimate
                    avg_execution_time=0.0,
                    complexity_rating=tool_spec.get("complexity_rating", 5),
                    prerequisites=[],
                    composition_potential=[]
                )
                
                self.tool_registry.register_tool(new_tool)
                logger.info(f"Created new tool: {new_tool.name}")
                return new_tool
                
        except Exception as e:
            logger.error(f"Failed to create tool for task: {e}")
        
        return None
    
    async def _generate_response(self, message: str, context: Dict[str, Any], 
                               tools: List[ToolCapability], 
                               reflection: Optional[ReflectionResult] = None) -> str:
        """Generate response using available tools and context"""
        
        # Build context-aware prompt
        role_specific_instructions = self._get_role_instructions()
        
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" for tool in tools
        ]) if tools else "No specific tools available"
        
        reflection_guidance = ""
        if reflection:
            reflection_guidance = f"""
            Previous attempt had issues: {reflection.critique}
            Improvement suggestions: {', '.join(reflection.improvement_suggestions)}
            Alternative approaches: {', '.join(reflection.alternative_approaches)}
            """
        
        # Include memory of recent interactions
        memory_context = ""
        if self.memory:
            recent_interactions = list(self.memory)[-3:]  # Last 3 interactions
            memory_context = "Recent conversation history:\n" + "\n".join([
                f"- {interaction}" for interaction in recent_interactions
            ])
        
        prompt = f"""
        You are an expert {self.role.value} agent with the following capabilities:
        {role_specific_instructions}
        
        Available tools:
        {tools_description}
        
        Context information:
        {json.dumps(context, indent=2)}
        
        {memory_context}
        
        {reflection_guidance}
        
        User request: {message}
        
        Please provide a comprehensive response that:
        1. Directly addresses the user's request
        2. Utilizes available tools appropriately
        3. Shows your reasoning process
        4. Indicates your confidence level
        5. Suggests follow-up actions if needed
        """
        
        response = await self.server.make_llm_request_with_retry(prompt, temperature=0.4)
        
        # Update memory
        self.memory.append(f"User: {message}")
        self.memory.append(f"Agent ({self.role.value}): {response}")
        
        return response
    
    async def _self_reflect(self, response: str, context: Dict[str, Any]) -> ReflectionResult:
        """Reflect on generated response and suggest improvements"""
        
        reflection_prompt = f"""
        You are a critical reviewer analyzing this response. Provide honest feedback:
        
        Response to review: {response}
        Context: {json.dumps(context, indent=2)}
        
        Analyze the response and provide:
        1. What's good about it?
        2. What could be improved?
        3. Are there any errors or gaps?
        4. Alternative approaches?
        5. Should this be retried? (yes/no)
        6. Confidence adjustment (-0.5 to +0.5)
        
        Provide your analysis in JSON format:
        {{
            "critique": "detailed analysis",
            "improvements": ["suggestion1", "suggestion2"],
            "should_retry": true/false,
            "confidence_adjustment": -0.2,
            "alternatives": ["approach1", "approach2"]
        }}
        """
        
        try:
            reflection_response = await self.server.make_llm_request_with_retry(reflection_prompt, temperature=0.2)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', reflection_response, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group())
                
                reflection = ReflectionResult(
                    agent_id=self.agent_id,
                    original_message_id=str(uuid.uuid4()),
                    critique=reflection_data.get("critique", "No specific critique"),
                    improvement_suggestions=reflection_data.get("improvements", []),
                    confidence_adjustment=reflection_data.get("confidence_adjustment", 0.0),
                    should_retry=reflection_data.get("should_retry", False),
                    alternative_approaches=reflection_data.get("alternatives", [])
                )
                
                self.reflection_history.append(reflection)
                return reflection
                
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
        
        # Default reflection
        return ReflectionResult(
            agent_id=self.agent_id,
            original_message_id=str(uuid.uuid4()),
            critique="Unable to perform reflection",
            improvement_suggestions=[],
            confidence_adjustment=0.0,
            should_retry=False,
            alternative_approaches=[]
        )
    
    async def _incorporate_collaboration_context(self, response: str, 
                                               session: CollaborationSession) -> str:
        """Incorporate insights from ongoing collaboration"""
        
        # Get recent messages from other agents
        recent_messages = [msg for msg in session.messages[-5:] if msg.agent_id != self.agent_id]
        
        if not recent_messages:
            return response
        
        collaboration_context = ""
        for msg in recent_messages:
            collaboration_context += f"Agent {msg.agent_id} ({msg.role.value}): {msg.content[:200]}...\n"
        
        enhancement_prompt = f"""
        You are collaborating with other agents. Consider their insights and enhance your response:
        
        Your current response: {response}
        
        Other agents' contributions:
        {collaboration_context}
        
        Enhance your response by:
        1. Building on others' insights
        2. Addressing any gaps they've identified
        3. Providing complementary information
        4. Maintaining coherence with the overall discussion
        
        Provide the enhanced response:
        """
        
        try:
            enhanced_response = await self.server.make_llm_request_with_retry(enhancement_prompt, temperature=0.3)
            return enhanced_response
        except Exception as e:
            logger.error(f"Collaboration enhancement failed: {e}")
            return response
    
    def _get_role_instructions(self) -> str:
        """Get role-specific instructions"""
        instructions = {
            AgentRole.ORCHESTRATOR: "Coordinate multiple agents, break down complex tasks, and synthesize results.",
            AgentRole.SPECIALIST: "Provide deep expertise in your domain, focus on accuracy and technical detail.",
            AgentRole.CRITIC: "Analyze and critique solutions, identify weaknesses and suggest improvements.",
            AgentRole.SYNTHESIZER: "Combine multiple perspectives into coherent, comprehensive solutions.",
            AgentRole.TOOL_CREATOR: "Design and create new tools to solve specific problems.",
            AgentRole.REFLECTION_AGENT: "Reflect on processes and outcomes, facilitate learning and improvement.",
            AgentRole.CONTEXT_MANAGER: "Maintain and organize context information across agent interactions."
        }
        return instructions.get(self.role, "General purpose problem solving")
    
    def _calculate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate confidence in the response"""
        base_confidence = self.task_success_rate
        
        # Adjust based on response length and detail
        length_factor = min(len(response) / 1000, 1.0) * 0.1
        
        # Adjust based on context availability
        context_factor = len(context) / 10.0 * 0.1
        
        return min(base_confidence + length_factor + context_factor, 1.0)
    
    async def _explain_reasoning(self, message: str, response: str, tools: List[ToolCapability]) -> str:
        """Explain the reasoning behind the response"""
        
        reasoning_prompt = f"""
        Explain your reasoning process for this interaction:
        
        Input: {message}
        Output: {response}
        Tools used: {[t.name for t in tools]}
        
        Provide a brief explanation of:
        1. How you interpreted the request
        2. Why you chose this approach
        3. How the tools helped (if any)
        4. What assumptions you made
        
        Keep it concise (2-3 sentences):
        """
        
        try:
            reasoning = await self.server.make_llm_request_with_retry(reasoning_prompt, temperature=0.1)
            return reasoning.strip()
        except Exception as e:
            return f"Standard reasoning process applied. Tools: {[t.name for t in tools]}"
    
    async def _update_learning(self, input_msg: str, response: str, response_time: float):
        """Update agent's learning based on interaction"""
        
        # Update response time tracking
        if self.avg_response_time == 0.0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        
        # Extract and store expertise areas (simple keyword extraction)
        import re
        keywords = re.findall(r'\b[a-zA-Z]{4,}\b', input_msg.lower())
        for keyword in keywords[:3]:  # Store top 3 keywords
            self.expertise_areas.add(keyword)
        
        # Limit expertise areas to prevent memory bloat
        if len(self.expertise_areas) > 20:
            self.expertise_areas = set(list(self.expertise_areas)[-20:])


class MultiAgentCollaborationOrchestrator:
    """Orchestrates multi-round agent collaborations with dynamic tool discovery"""
    
    def __init__(self, server):
        self.server = server
        self.tool_registry = DynamicToolRegistry()
        self.agents: Dict[str, AdvancedAgent] = {}
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.collaboration_patterns = defaultdict(list)
        
        # Initialize default tools
        self._initialize_default_tools()
        
        # Create default agent pool
        self._initialize_agent_pool()
    
    def _initialize_default_tools(self):
        """Initialize with basic tools"""
        basic_tools = [
            ToolCapability("web_search", "Web Search", "Search the web for information", 
                         {}, 0.8, 2.0, 3, [], []),
            ToolCapability("code_analyze", "Code Analysis", "Analyze code for issues", 
                         {}, 0.7, 1.5, 4, [], []),
            ToolCapability("text_process", "Text Processing", "Process and analyze text", 
                         {}, 0.9, 0.8, 2, [], []),
        ]
        
        for tool in basic_tools:
            self.tool_registry.register_tool(tool)
    
    def _initialize_agent_pool(self):
        """Create initial pool of specialized agents"""
        agent_configs = [
            (AgentRole.ORCHESTRATOR, "orchestrator_1"),
            (AgentRole.SPECIALIST, "specialist_code"),
            (AgentRole.SPECIALIST, "specialist_research"),
            (AgentRole.CRITIC, "critic_1"),
            (AgentRole.SYNTHESIZER, "synthesizer_1"),
            (AgentRole.TOOL_CREATOR, "tool_creator_1"),
            (AgentRole.REFLECTION_AGENT, "reflection_1"),
            (AgentRole.CONTEXT_MANAGER, "context_manager_1"),
        ]
        
        for role, agent_id in agent_configs:
            self.agents[agent_id] = AdvancedAgent(agent_id, role, self.server, self.tool_registry)
    
    async def start_collaboration(self, task: str, mode: InteractionMode = InteractionMode.COLLABORATION,
                                max_rounds: int = 5, participant_roles: List[AgentRole] = None) -> str:
        """Start a new multi-agent collaboration session"""
        
        session_id = f"collab_{uuid.uuid4().hex[:12]}"
        
        # Select participants based on task and requested roles
        participants = await self._select_optimal_agents(task, participant_roles or [])
        
        session = CollaborationSession(
            session_id=session_id,
            task_description=task,
            participants=[agent.agent_id for agent in participants],
            mode=mode,
            max_rounds=max_rounds,
            current_round=0,
            messages=[],
            shared_context={"original_task": task},
            consensus_threshold=0.7,
            start_time=time.time(),
            end_time=None
        )
        
        self.active_sessions[session_id] = session
        
        # Execute collaboration
        final_result = await self._execute_collaboration(session_id)
        
        return final_result
    
    async def _select_optimal_agents(self, task: str, preferred_roles: List[AgentRole]) -> List[AdvancedAgent]:
        """Intelligently select agents for the task"""
        
        selected_agents = []
        
        # Always include an orchestrator
        orchestrator = next((agent for agent in self.agents.values() 
                           if agent.role == AgentRole.ORCHESTRATOR), None)
        if orchestrator:
            selected_agents.append(orchestrator)
        
        # Add preferred roles
        for role in preferred_roles:
            agent = next((agent for agent in self.agents.values() 
                         if agent.role == role and agent not in selected_agents), None)
            if agent:
                selected_agents.append(agent)
        
        # Analyze task to determine additional needed roles
        task_analysis_prompt = f"""
        Analyze this task and recommend which agent roles would be most helpful:
        
        Task: {task}
        
        Available roles: {[role.value for role in AgentRole]}
        Current selection: {[agent.role.value for agent in selected_agents]}
        
        Recommend 2-3 additional roles that would be valuable for this task.
        Respond with just the role names, comma-separated:
        """
        
        try:
            recommended_roles = await self.server.make_llm_request_with_retry(task_analysis_prompt, temperature=0.2)
            role_names = [name.strip().lower() for name in recommended_roles.split(",")]
            
            for role_name in role_names:
                try:
                    role = AgentRole(role_name)
                    agent = next((agent for agent in self.agents.values() 
                                if agent.role == role and agent not in selected_agents), None)
                    if agent and len(selected_agents) < 6:  # Limit to 6 agents max
                        selected_agents.append(agent)
                except ValueError:
                    continue  # Invalid role name
                    
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
        
        # Ensure minimum viable team
        if len(selected_agents) < 2:
            # Add a specialist if we don't have enough agents
            specialist = next((agent for agent in self.agents.values() 
                             if agent.role == AgentRole.SPECIALIST and agent not in selected_agents), None)
            if specialist:
                selected_agents.append(specialist)
        
        return selected_agents
    
    async def _execute_collaboration(self, session_id: str) -> str:
        """Execute the multi-round collaboration"""
        
        session = self.active_sessions[session_id]
        participants = [self.agents[agent_id] for agent_id in session.participants]
        
        current_context = session.shared_context.copy()
        
        for round_num in range(session.max_rounds):
            session.current_round = round_num + 1
            
            logger.info(f"Starting collaboration round {session.current_round}/{session.max_rounds}")
            
            # Each agent contributes in this round
            round_messages = []
            
            for agent in participants:
                # Build prompt with current context and previous messages
                round_prompt = self._build_round_prompt(session, agent, current_context)
                
                try:
                    message = await agent.process_message(round_prompt, current_context, session)
                    round_messages.append(message)
                    session.messages.append(message)
                    
                    # Update shared context with agent's contribution
                    current_context[f"agent_{agent.agent_id}_contribution_r{session.current_round}"] = {
                        "content": message.content,
                        "confidence": message.confidence,
                        "tools_used": message.metadata.get("tools_used", [])
                    }
                    
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} failed in round {session.current_round}: {e}")
                    continue
            
            # Check for consensus or early termination
            if await self._check_consensus(session, round_messages):
                logger.info(f"Consensus reached in round {session.current_round}")
                break
            
            # Add inter-round reflection
            if session.current_round < session.max_rounds:
                await self._conduct_inter_round_reflection(session, current_context)
        
        # Synthesize final result
        final_result = await self._synthesize_final_result(session, current_context)
        
        session.end_time = time.time()
        return final_result
    
    def _build_round_prompt(self, session: CollaborationSession, agent: AdvancedAgent, context: Dict[str, Any]) -> str:
        """Build context-aware prompt for each round"""
        
        # Get recent contributions from other agents
        other_contributions = []
        recent_messages = session.messages[-len(session.participants):]
        
        for msg in recent_messages:
            if msg.agent_id != agent.agent_id:
                other_contributions.append(f"Agent {msg.agent_id} ({msg.role.value}): {msg.content[:300]}...")
        
        collaboration_context = "\n\n".join(other_contributions) if other_contributions else "No prior contributions in this round"
        
        round_prompt = f"""
        COLLABORATION ROUND {session.current_round}/{session.max_rounds}
        
        Original Task: {session.task_description}
        Collaboration Mode: {session.mode.value}
        Your Role: {agent.role.value}
        
        Other agents' contributions this round:
        {collaboration_context}
        
        Shared Context:
        {json.dumps({k: v for k, v in context.items() if not k.startswith('agent_')}, indent=2)}
        
        Your task for this round:
        {"Provide your unique expertise and perspective on the task. Build upon or respectfully challenge others' contributions." if session.mode == InteractionMode.COLLABORATION else "Debate and critique the approaches suggested so far. Propose better alternatives." if session.mode == InteractionMode.DEBATE else "Provide your perspective and work towards consensus."}
        
        Focus on:
        1. Your specialized expertise
        2. Identifying gaps in current solutions
        3. Suggesting concrete improvements
        4. Considering implementation feasibility
        """
        
        return round_prompt
    
    async def _check_consensus(self, session: CollaborationSession, round_messages: List[AgentMessage]) -> bool:
        """Check if agents have reached consensus"""
        
        if len(round_messages) < 2:
            return False
        
        # Calculate average confidence
        avg_confidence = sum(msg.confidence for msg in round_messages) / len(round_messages)
        
        # Use LLM to assess consensus
        consensus_prompt = f"""
        Analyze these agent responses to determine if they've reached consensus:
        
        Original task: {session.task_description}
        
        Agent responses:
        """
        
        for msg in round_messages:
            consensus_prompt += f"\n{msg.role.value}: {msg.content[:500]}..."
        
        consensus_prompt += """
        
        Do these responses show consensus? Consider:
        1. Agreement on approach
        2. Complementary rather than conflicting solutions
        3. High confidence levels
        4. Convergence of ideas
        
        Respond with: YES or NO, followed by brief reasoning.
        """
        
        try:
            consensus_response = await self.server.make_llm_request_with_retry(consensus_prompt, temperature=0.1)
            
            has_consensus = consensus_response.strip().upper().startswith("YES")
            confidence_consensus = avg_confidence >= session.consensus_threshold
            
            return has_consensus and confidence_consensus
            
        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
            return False
    
    async def _conduct_inter_round_reflection(self, session: CollaborationSession, context: Dict[str, Any]):
        """Conduct reflection between rounds"""
        
        reflection_agent = next((self.agents[agent_id] for agent_id in session.participants 
                               if self.agents[agent_id].role == AgentRole.REFLECTION_AGENT), None)
        
        if not reflection_agent:
            return
        
        recent_messages = session.messages[-len(session.participants):]
        
        reflection_prompt = f"""
        Reflect on the current progress in this collaboration:
        
        Task: {session.task_description}
        Round: {session.current_round}/{session.max_rounds}
        
        Recent contributions:
        """
        
        for msg in recent_messages:
            reflection_prompt += f"\n{msg.role.value}: {msg.content[:200]}..."
        
        reflection_prompt += """
        
        Analyze:
        1. What progress has been made?
        2. What gaps or conflicts remain?
        3. What should agents focus on in the next round?
        4. Are we on track to solve the original task?
        
        Provide guidance for the next round:
        """
        
        try:
            reflection = await reflection_agent.process_message(reflection_prompt, context)
            
            # Add reflection to shared context
            context["inter_round_reflection"] = {
                "round": session.current_round,
                "reflection": reflection.content,
                "timestamp": time.time()
            }
            
            logger.info(f"Inter-round reflection: {reflection.content[:200]}...")
            
        except Exception as e:
            logger.error(f"Inter-round reflection failed: {e}")
    
    async def _synthesize_final_result(self, session: CollaborationSession, context: Dict[str, Any]) -> str:
        """Synthesize final result from all agent contributions"""
        
        synthesizer = next((self.agents[agent_id] for agent_id in session.participants 
                          if self.agents[agent_id].role == AgentRole.SYNTHESIZER), None)
        
        if not synthesizer:
            # Use any available agent as fallback
            synthesizer = self.agents[session.participants[0]]
        
        # Collect all agent contributions
        all_contributions = []
        for msg in session.messages:
            agent_role = msg.role.value
            contribution = f"{agent_role}: {msg.content}"
            all_contributions.append(contribution)
        
        synthesis_prompt = f"""
        Synthesize a comprehensive final answer from this multi-agent collaboration:
        
        Original Task: {session.task_description}
        Collaboration Duration: {session.current_round} rounds
        Mode: {session.mode.value}
        
        All Agent Contributions:
        {chr(10).join(all_contributions)}
        
        Create a final, comprehensive response that:
        1. Directly answers the original task
        2. Integrates the best insights from all agents
        3. Resolves any conflicts between perspectives
        4. Provides actionable recommendations
        5. Acknowledges different viewpoints where appropriate
        
        Final Answer:
        """
        
        try:
            final_message = await synthesizer.process_message(synthesis_prompt, context)
            
            # Log collaboration statistics
            total_time = time.time() - session.start_time
            logger.info(f"Collaboration completed: {session.current_round} rounds, {len(session.messages)} messages, {total_time:.2f}s")
            
            return final_message.content
            
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            
            # Fallback: simple concatenation
            return f"""
            Multi-agent collaboration result for: {session.task_description}
            
            {chr(10).join([msg.content for msg in session.messages[-len(session.participants):]])}
            
            Note: Automated synthesis failed, showing raw agent outputs.
            """

    def get_collaboration_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a collaboration session"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        stats = {
            "session_id": session_id,
            "task": session.task_description,
            "mode": session.mode.value,
            "rounds_completed": session.current_round,
            "total_messages": len(session.messages),
            "participants": len(session.participants),
            "duration": (session.end_time or time.time()) - session.start_time,
            "agent_contributions": {},
            "tools_discovered": len(self.tool_registry.tools),
            "consensus_reached": session.end_time is not None and session.current_round < session.max_rounds
        }
        
        # Agent-specific statistics
        for agent_id in session.participants:
            agent_messages = [msg for msg in session.messages if msg.agent_id == agent_id]
            stats["agent_contributions"][agent_id] = {
                "messages": len(agent_messages),
                "avg_confidence": sum(msg.confidence for msg in agent_messages) / len(agent_messages) if agent_messages else 0,
                "tools_used": sum(len(msg.tool_calls) for msg in agent_messages)
            }
        
        return stats


# Integration with existing MCP server
def integrate_advanced_agents(server):
    """Integrate advanced agent system with existing MCP server"""
    
    server.collaboration_orchestrator = MultiAgentCollaborationOrchestrator(server)
    
    return server

# New tool handlers for advanced collaboration
def handle_start_collaboration(arguments, server):
    """Start a multi-agent collaboration session"""
    
    task = arguments.get("task", "").strip()
    if not task:
        raise ValueError("'task' is required")
    
    mode = arguments.get("mode", "collaboration")
    max_rounds = int(arguments.get("max_rounds", 5))
    roles = arguments.get("roles", [])
    
    try:
        mode_enum = InteractionMode(mode)
    except ValueError:
        mode_enum = InteractionMode.COLLABORATION
    
    role_enums = []
    for role in roles:
        try:
            role_enums.append(AgentRole(role.lower()))
        except ValueError:
            continue
    
    # Run async collaboration
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            server.collaboration_orchestrator.start_collaboration(
                task, mode_enum, max_rounds, role_enums
            )
        )
        return result
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                server.collaboration_orchestrator.start_collaboration(
                    task, mode_enum, max_rounds, role_enums
                )
            )
            return result
        finally:
            loop.close()

def handle_get_collaboration_stats(arguments, server):
    """Get statistics about a collaboration session"""
    
    session_id = arguments.get("session_id", "").strip()
    if not session_id:
        # Get stats for all active sessions
        return {
            "active_sessions": list(server.collaboration_orchestrator.active_sessions.keys()),
            "total_tools": len(server.collaboration_orchestrator.tool_registry.tools),
            "total_agents": len(server.collaboration_orchestrator.agents)
        }
    
    return server.collaboration_orchestrator.get_collaboration_stats(session_id)

def handle_create_custom_agent(arguments, server):
    """Create a custom agent with specific role and expertise"""
    
    agent_id = arguments.get("agent_id", f"custom_{uuid.uuid4().hex[:8]}")
    role = arguments.get("role", "specialist")
    expertise = arguments.get("expertise", [])
    
    try:
        role_enum = AgentRole(role.lower())
    except ValueError:
        role_enum = AgentRole.SPECIALIST
    
    # Create new agent
    new_agent = AdvancedAgent(agent_id, role_enum, server, server.collaboration_orchestrator.tool_registry)
    
    # Set expertise areas
    new_agent.expertise_areas.update(expertise)
    
    # Register with orchestrator
    server.collaboration_orchestrator.agents[agent_id] = new_agent
    
    return {
        "agent_id": agent_id,
        "role": role_enum.value,
        "expertise": list(new_agent.expertise_areas),
        "message": f"Created custom agent: {agent_id}"
    }

# Additional tool definitions to add to get_all_tools()
ADVANCED_COLLABORATION_TOOLS = [
    {
        "name": "start_collaboration",
        "description": "Start a multi-agent collaboration session with dynamic tool discovery and reflection",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for agents to collaborate on"},
                "mode": {"type": "string", "enum": ["collaboration", "debate", "competition", "consensus_building"], "default": "collaboration"},
                "max_rounds": {"type": "integer", "description": "Maximum collaboration rounds", "default": 5},
                "roles": {"type": "array", "items": {"type": "string"}, "description": "Preferred agent roles to include"}
            },
            "required": ["task"]
        }
    },
    {
        "name": "get_collaboration_stats",
        "description": "Get statistics and insights about collaboration sessions",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Specific session ID, or omit for overview"}
            }
        }
    },
    {
        "name": "create_custom_agent",
        "description": "Create a custom agent with specific role and expertise areas",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Custom agent identifier"},
                "role": {"type": "string", "enum": ["orchestrator", "specialist", "critic", "synthesizer", "tool_creator", "reflection_agent", "context_manager"], "default": "specialist"},
                "expertise": {"type": "array", "items": {"type": "string"}, "description": "Areas of expertise"}
            }
        }
    }
]