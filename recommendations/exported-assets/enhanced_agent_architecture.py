# Enhanced Multi-Agent Architecture with Complexity Routing
import asyncio
import json
import os
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
import hashlib
from uuid import uuid4

class ComplexityLevel(Enum):
    SIMPLE = "simple"           # Local model tasks
    MODERATE = "moderate"       # OpenAI GPT-4o-mini
    COMPLEX = "complex"         # OpenAI GPT-4o or Anthropic Claude
    EXPERT = "expert"           # Anthropic Claude Sonnet for specialized domains

class AgentRole(Enum):
    RESEARCHER = "researcher"
    PLANNER = "planner" 
    CODER = "coder"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"
    DOMAIN_EXPERT = "domain_expert"
    ORCHESTRATOR = "orchestrator"
    SECURITY_EXPERT = "security_expert"
    PERFORMANCE_EXPERT = "performance_expert"

@dataclass
class AgentCapability:
    role: AgentRole
    complexity_level: ComplexityLevel
    specializations: List[str]
    tools: List[str]
    collaboration_patterns: List[str]

@dataclass
class TaskComplexityAnalysis:
    overall_complexity: ComplexityLevel
    reasoning_depth: int  # 1-5 scale
    domain_specialization: List[str]
    multi_step_required: bool
    research_intensive: bool
    collaboration_needed: bool
    estimated_duration: int  # minutes

@dataclass
class CollaborationSession:
    session_id: str
    participants: List[str]
    research_artifacts: Dict[str, Any]
    discussion_history: List[Dict[str, Any]]
    consensus_points: List[str]
    action_items: List[Dict[str, Any]]
    created_at: float
    status: str = "active"

class EnhancedAgentRouter:
    def __init__(self, storage):
        self.storage = storage
        self.complexity_patterns = self._load_complexity_patterns()
        self.agent_capabilities = self._define_agent_capabilities()
        self.backend_health = {}
        self._initialize_backends()

    def _load_complexity_patterns(self) -> Dict[str, ComplexityLevel]:
        """Define patterns that indicate task complexity"""
        return {
            # Simple patterns (local model sufficient)
            'simple_file_ops': ComplexityLevel.SIMPLE,
            'basic_formatting': ComplexityLevel.SIMPLE,
            'syntax_checking': ComplexityLevel.SIMPLE,
            'simple_queries': ComplexityLevel.SIMPLE,
            
            # Moderate patterns (OpenAI GPT-4o-mini)
            'code_explanation': ComplexityLevel.MODERATE,
            'basic_refactoring': ComplexityLevel.MODERATE,
            'test_generation': ComplexityLevel.MODERATE,
            'documentation': ComplexityLevel.MODERATE,
            
            # Complex patterns (GPT-4o or Claude)
            'architecture_design': ComplexityLevel.COMPLEX,
            'multi_file_refactor': ComplexityLevel.COMPLEX,
            'performance_optimization': ComplexityLevel.COMPLEX,
            'system_integration': ComplexityLevel.COMPLEX,
            
            # Expert patterns (Claude Sonnet)
            'security_analysis': ComplexityLevel.EXPERT,
            'formal_verification': ComplexityLevel.EXPERT,
            'research_synthesis': ComplexityLevel.EXPERT,
            'domain_expertise': ComplexityLevel.EXPERT,
        }

    def _define_agent_capabilities(self) -> Dict[AgentRole, AgentCapability]:
        """Define capabilities for each agent role"""
        return {
            AgentRole.RESEARCHER: AgentCapability(
                role=AgentRole.RESEARCHER,
                complexity_level=ComplexityLevel.COMPLEX,
                specializations=['web_search', 'data_analysis', 'source_verification'],
                tools=['web_search', 'firecrawl', 'arxiv_search'],
                collaboration_patterns=['research_synthesis', 'fact_checking']
            ),
            AgentRole.PLANNER: AgentCapability(
                role=AgentRole.PLANNER,
                complexity_level=ComplexityLevel.MODERATE,
                specializations=['task_decomposition', 'workflow_design', 'resource_planning'],
                tools=['project_templates', 'estimation_models'],
                collaboration_patterns=['task_delegation', 'milestone_tracking']
            ),
            AgentRole.CODER: AgentCapability(
                role=AgentRole.CODER,
                complexity_level=ComplexityLevel.MODERATE,
                specializations=['implementation', 'debugging', 'testing'],
                tools=['code_execution', 'file_operations', 'git_operations'],
                collaboration_patterns=['code_review', 'pair_programming']
            ),
            AgentRole.REVIEWER: AgentCapability(
                role=AgentRole.REVIEWER,
                complexity_level=ComplexityLevel.COMPLEX,
                specializations=['quality_assurance', 'best_practices', 'risk_assessment'],
                tools=['static_analysis', 'test_runners', 'security_scanners'],
                collaboration_patterns=['peer_review', 'quality_gates']
            ),
            AgentRole.DOMAIN_EXPERT: AgentCapability(
                role=AgentRole.DOMAIN_EXPERT,
                complexity_level=ComplexityLevel.EXPERT,
                specializations=['specialized_knowledge', 'industry_standards', 'compliance'],
                tools=['domain_databases', 'regulatory_checks', 'standard_references'],
                collaboration_patterns=['expert_consultation', 'knowledge_validation']
            )
        }

    def _initialize_backends(self):
        """Check availability and health of different LLM backends"""
        backends = {
            'local': self._check_local_backend(),
            'openai': self._check_openai_backend(),
            'anthropic': self._check_anthropic_backend()
        }
        self.backend_health = backends

    def _check_local_backend(self) -> Dict[str, Any]:
        """Check local LM Studio availability"""
        try:
            import requests
            base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return {
                'available': response.status_code == 200,
                'latency': response.elapsed.total_seconds(),
                'models': response.json().get('data', []) if response.status_code == 200 else []
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _check_openai_backend(self) -> Dict[str, Any]:
        """Check OpenAI API availability"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {'available': False, 'error': 'No API key'}
        
        try:
            import requests
            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=10)
            return {
                'available': response.status_code == 200,
                'latency': response.elapsed.total_seconds(),
                'has_gpt4': 'gpt-4' in response.text if response.status_code == 200 else False
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _check_anthropic_backend(self) -> Dict[str, Any]:
        """Check Anthropic API availability"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {'available': False, 'error': 'No API key'}
        
        try:
            import requests
            headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
            # Simple ping to check availability
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json={
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1,
                    'messages': [{'role': 'user', 'content': 'ping'}]
                },
                timeout=10
            )
            return {
                'available': response.status_code in [200, 400],  # 400 is ok, means API is responding
                'latency': response.elapsed.total_seconds(),
                'has_sonnet': True  # Assume Sonnet is available if API responds
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def analyze_task_complexity(self, task_description: str, context: Dict[str, Any] = None) -> TaskComplexityAnalysis:
        """Analyze task complexity to determine appropriate routing"""
        context = context or {}
        
        # Keyword-based complexity analysis
        task_lower = task_description.lower()
        complexity_scores = []
        
        # Check against known patterns
        for pattern, complexity in self.complexity_patterns.items():
            if any(keyword in task_lower for keyword in pattern.split('_')):
                complexity_scores.append(complexity.value)
        
        # Analyze specific indicators
        reasoning_indicators = [
            'analyze', 'evaluate', 'compare', 'synthesize', 'recommend',
            'optimize', 'design', 'architect', 'strategy'
        ]
        
        domain_indicators = {
            'security': ['security', 'vulnerability', 'encryption', 'authentication'],
            'performance': ['performance', 'optimization', 'scalability', 'latency'],
            'architecture': ['architecture', 'design pattern', 'system design'],
            'research': ['research', 'literature', 'analysis', 'survey'],
            'compliance': ['compliance', 'regulation', 'audit', 'governance']
        }
        
        reasoning_depth = sum(1 for indicator in reasoning_indicators if indicator in task_lower)
        domain_specializations = [
            domain for domain, keywords in domain_indicators.items()
            if any(keyword in task_lower for keyword in keywords)
        ]
        
        multi_step_indicators = ['plan', 'implement', 'test', 'deploy', 'stages', 'phases']
        multi_step_required = any(indicator in task_lower for indicator in multi_step_indicators)
        
        research_indicators = ['research', 'investigate', 'find', 'discover', 'explore']
        research_intensive = any(indicator in task_lower for indicator in research_indicators)
        
        collaboration_indicators = ['team', 'collaborate', 'review', 'discuss', 'consensus']
        collaboration_needed = any(indicator in task_lower for indicator in collaboration_indicators)
        
        # Determine overall complexity
        if domain_specializations or reasoning_depth >= 3:
            overall_complexity = ComplexityLevel.EXPERT
        elif multi_step_required and research_intensive:
            overall_complexity = ComplexityLevel.COMPLEX
        elif reasoning_depth >= 1 or len(task_lower.split()) > 50:
            overall_complexity = ComplexityLevel.MODERATE
        else:
            overall_complexity = ComplexityLevel.SIMPLE
        
        # Estimate duration based on complexity and scope
        base_duration = 5  # minutes
        if overall_complexity == ComplexityLevel.EXPERT:
            base_duration = 30
        elif overall_complexity == ComplexityLevel.COMPLEX:
            base_duration = 20
        elif overall_complexity == ComplexityLevel.MODERATE:
            base_duration = 10
        
        duration_multipliers = {
            'multi_step': 2.0,
            'research_intensive': 1.5,
            'collaboration': 1.8
        }
        
        estimated_duration = base_duration
        if multi_step_required:
            estimated_duration *= duration_multipliers['multi_step']
        if research_intensive:
            estimated_duration *= duration_multipliers['research_intensive']
        if collaboration_needed:
            estimated_duration *= duration_multipliers['collaboration']
        
        return TaskComplexityAnalysis(
            overall_complexity=overall_complexity,
            reasoning_depth=min(reasoning_depth, 5),
            domain_specialization=domain_specializations,
            multi_step_required=multi_step_required,
            research_intensive=research_intensive,
            collaboration_needed=collaboration_needed,
            estimated_duration=int(estimated_duration)
        )

    def select_optimal_backend(self, complexity: ComplexityLevel, role: AgentRole) -> str:
        """Select the best available backend for the given complexity and role"""
        
        # Priority mapping based on complexity
        backend_priorities = {
            ComplexityLevel.SIMPLE: ['local', 'openai', 'anthropic'],
            ComplexityLevel.MODERATE: ['openai', 'anthropic', 'local'],
            ComplexityLevel.COMPLEX: ['openai', 'anthropic'],
            ComplexityLevel.EXPERT: ['anthropic', 'openai']
        }
        
        # Role-specific backend preferences
        role_preferences = {
            AgentRole.SECURITY_EXPERT: ['anthropic', 'openai'],
            AgentRole.DOMAIN_EXPERT: ['anthropic', 'openai'],
            AgentRole.RESEARCHER: ['anthropic', 'openai', 'local']
        }
        
        # Get priority list
        priorities = role_preferences.get(role, backend_priorities.get(complexity, ['local']))
        
        # Select first available backend
        for backend in priorities:
            if self.backend_health.get(backend, {}).get('available', False):
                return backend
        
        # Fallback to local if nothing else available
        return 'local'

    def create_agent_team(self, task_analysis: TaskComplexityAnalysis) -> List[Dict[str, Any]]:
        """Create an optimal agent team for the given task"""
        team = []
        
        # Always include orchestrator for complex tasks
        if task_analysis.collaboration_needed or task_analysis.multi_step_required:
            team.append({
                'role': AgentRole.ORCHESTRATOR,
                'backend': self.select_optimal_backend(ComplexityLevel.MODERATE, AgentRole.ORCHESTRATOR),
                'primary': True
            })
        
        # Add researchers if research intensive
        if task_analysis.research_intensive:
            team.append({
                'role': AgentRole.RESEARCHER,
                'backend': self.select_optimal_backend(task_analysis.overall_complexity, AgentRole.RESEARCHER),
                'specializations': task_analysis.domain_specialization
            })
        
        # Add domain experts for specialized areas
        for domain in task_analysis.domain_specialization:
            team.append({
                'role': AgentRole.DOMAIN_EXPERT,
                'backend': self.select_optimal_backend(ComplexityLevel.EXPERT, AgentRole.DOMAIN_EXPERT),
                'domain': domain
            })
        
        # Add core roles based on task type
        core_roles = [AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER]
        for role in core_roles:
            team.append({
                'role': role,
                'backend': self.select_optimal_backend(task_analysis.overall_complexity, role)
            })
        
        return team

class CollaborativeWorkflowManager:
    """Manages collaborative workflows between agents"""
    
    def __init__(self, storage, router: EnhancedAgentRouter):
        self.storage = storage
        self.router = router
        self.active_sessions: Dict[str, CollaborationSession] = {}

    async def initiate_collaboration(self, task_description: str, context: Dict[str, Any] = None) -> str:
        """Start a new collaborative session"""
        session_id = f"collab_{uuid4().hex[:12]}"
        
        # Analyze task complexity
        task_analysis = self.router.analyze_task_complexity(task_description, context)
        
        # Create agent team
        agent_team = self.router.create_agent_team(task_analysis)
        
        # Initialize collaboration session
        session = CollaborationSession(
            session_id=session_id,
            participants=[agent['role'].value for agent in agent_team],
            research_artifacts={},
            discussion_history=[],
            consensus_points=[],
            action_items=[],
            created_at=time.time()
        )
        
        self.active_sessions[session_id] = session
        
        # Store session in persistent storage
        self.storage.store_memory(
            key=f"collaboration_{session_id}",
            value=json.dumps(asdict(session)),
            category="collaboration"
        )
        
        return session_id

    async def conduct_research_phase(self, session_id: str, research_queries: List[str]) -> Dict[str, Any]:
        """Conduct collaborative research phase"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        research_results = {}
        
        # Parallel research by multiple researchers
        research_tasks = []
        for i, query in enumerate(research_queries[:3]):  # Limit to 3 queries
            task = self._conduct_individual_research(query, f"researcher_{i}")
            research_tasks.append(task)
        
        # Execute research in parallel
        results = await asyncio.gather(*research_tasks, return_exceptions=True)
        
        # Consolidate results
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                research_results[f"query_{i}"] = result
        
        # Store research artifacts
        session.research_artifacts.update(research_results)
        
        return research_results

    async def _conduct_individual_research(self, query: str, researcher_id: str) -> Dict[str, Any]:
        """Conduct research by individual researcher"""
        # Simulate research call - replace with actual research implementation
        return {
            'researcher_id': researcher_id,
            'query': query,
            'findings': f"Research findings for: {query}",
            'sources': [],
            'confidence': 0.8,
            'timestamp': time.time()
        }

    async def facilitate_agent_discussion(self, session_id: str, topic: str) -> List[Dict[str, Any]]:
        """Facilitate discussion between agents"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        discussion_rounds = []
        
        # Each agent contributes their perspective
        for participant in session.participants:
            contribution = await self._get_agent_contribution(participant, topic, session.research_artifacts)
            discussion_rounds.append(contribution)
            session.discussion_history.append(contribution)
        
        # Synthesize discussion
        synthesis = await self._synthesize_discussion(discussion_rounds)
        session.consensus_points.extend(synthesis.get('consensus_points', []))
        
        return discussion_rounds

    async def _get_agent_contribution(self, agent_role: str, topic: str, research_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Get individual agent's contribution to discussion"""
        # Simulate agent contribution - replace with actual LLM calls
        return {
            'agent': agent_role,
            'topic': topic,
            'perspective': f"{agent_role}'s perspective on {topic}",
            'key_points': [f"Point 1 from {agent_role}", f"Point 2 from {agent_role}"],
            'concerns': [],
            'recommendations': [],
            'timestamp': time.time()
        }

    async def _synthesize_discussion(self, discussion_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize agent discussion into consensus points"""
        # Simulate synthesis - replace with actual LLM synthesis
        return {
            'consensus_points': ['Consensus point 1', 'Consensus point 2'],
            'areas_of_disagreement': [],
            'action_items': ['Action item 1', 'Action item 2']
        }