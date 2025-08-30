# Enhanced Agent Team Handlers with Collaborative Research
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from pathlib import Path

# Import the enhanced architecture components
from enhanced_agent_architecture import (
    ComplexityLevel, AgentRole, TaskComplexityAnalysis, 
    CollaborationSession, EnhancedAgentRouter, CollaborativeWorkflowManager
)
from enhanced_mcp_storage import EnhancedMCPStorage, ContextEnvelope

class EnhancedAgentTeamManager:
    """Enhanced agent team manager with complexity routing and collaboration"""
    
    def __init__(self, storage: EnhancedMCPStorage):
        self.storage = storage
        self.router = EnhancedAgentRouter(storage)
        self.workflow_manager = CollaborativeWorkflowManager(storage, self.router)
        self.active_teams = {}
        
    async def handle_collaborative_task(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced task handler with collaborative research and discussion phases"""
        task_desc = arguments.get("task", "").strip()
        if not task_desc:
            raise ValueError("'task' is required")
        
        target_files = arguments.get("target_files", [])
        constraints = arguments.get("constraints", "").strip()
        research_required = arguments.get("research_required", True)
        discussion_rounds = arguments.get("discussion_rounds", 2)
        apply_changes = arguments.get("apply_changes", False)
        
        # Analyze task complexity
        task_analysis = self.router.analyze_task_complexity(task_desc, {
            'target_files': target_files,
            'constraints': constraints
        })
        
        # Create collaboration session
        session_id = await self.workflow_manager.initiate_collaboration(task_desc, {
            'task_analysis': asdict(task_analysis),
            'target_files': target_files,
            'constraints': constraints
        })
        
        # Create context envelope for this task
        context_envelope_id = self.storage.create_context_envelope(
            session_id=session_id,
            context_type='task',
            content={
                'task_description': task_desc,
                'target_files': target_files,
                'constraints': constraints,
                'task_analysis': asdict(task_analysis)
            },
            metadata={
                'complexity': task_analysis.overall_complexity.value,
                'estimated_duration': task_analysis.estimated_duration,
                'collaboration_needed': task_analysis.collaboration_needed
            },
            ttl_seconds=3600  # 1 hour TTL
        )
        
        # Execute multi-phase collaboration
        result = await self._execute_collaborative_phases(
            session_id, context_envelope_id, task_analysis,
            research_required, discussion_rounds, apply_changes
        )
        
        # Store final result
        self.storage.create_artifact(
            session_id=session_id,
            artifact_type='final_result',
            title=f"Task Result: {task_desc[:50]}...",
            content=json.dumps(result),
            contributors=result.get('contributors', []),
            tags=['final', 'collaborative']
        )
        
        return result

    async def _execute_collaborative_phases(self, session_id: str, context_envelope_id: str,
                                          task_analysis: TaskComplexityAnalysis,
                                          research_required: bool, discussion_rounds: int,
                                          apply_changes: bool) -> Dict[str, Any]:
        """Execute the multi-phase collaborative workflow"""
        
        phases_completed = []
        research_artifacts = {}
        discussion_outcomes = []
        final_recommendations = {}
        
        try:
            # Phase 1: Individual Research (if required)
            if research_required and task_analysis.research_intensive:
                research_artifacts = await self._conduct_research_phase(
                    session_id, context_envelope_id, task_analysis
                )
                phases_completed.append("research")
            
            # Phase 2: Agent Discussion Rounds
            for round_num in range(discussion_rounds):
                discussion_outcome = await self._conduct_discussion_round(
                    session_id, context_envelope_id, task_analysis,
                    research_artifacts, round_num
                )
                discussion_outcomes.append(discussion_outcome)
                phases_completed.append(f"discussion_round_{round_num + 1}")
            
            # Phase 3: Synthesis and Final Recommendations
            final_recommendations = await self._synthesize_final_recommendations(
                session_id, context_envelope_id, task_analysis,
                research_artifacts, discussion_outcomes
            )
            phases_completed.append("synthesis")
            
            # Phase 4: Implementation (if requested)
            if apply_changes:
                implementation_result = await self._implement_recommendations(
                    session_id, context_envelope_id, final_recommendations
                )
                final_recommendations['implementation'] = implementation_result
                phases_completed.append("implementation")
            
            return {
                'session_id': session_id,
                'phases_completed': phases_completed,
                'research_artifacts': research_artifacts,
                'discussion_outcomes': discussion_outcomes,
                'final_recommendations': final_recommendations,
                'contributors': self._extract_contributors(session_id),
                'execution_time_minutes': self._calculate_execution_time(session_id)
            }
            
        except Exception as e:
            # Log the error and return partial results
            self.storage.log_agent_performance(
                agent_role='orchestrator',
                backend_used='system',
                task_type='collaborative_task',
                complexity_level=task_analysis.overall_complexity.value,
                execution_time_ms=int((time.time() - time.time()) * 1000),
                success=False,
                error_type=type(e).__name__
            )
            
            return {
                'session_id': session_id,
                'error': str(e),
                'phases_completed': phases_completed,
                'partial_results': {
                    'research_artifacts': research_artifacts,
                    'discussion_outcomes': discussion_outcomes
                }
            }

    async def _conduct_research_phase(self, session_id: str, context_envelope_id: str,
                                    task_analysis: TaskComplexityAnalysis) -> Dict[str, Any]:
        """Conduct collaborative research with multiple specialized researchers"""
        
        # Generate research queries based on task analysis
        research_queries = await self._generate_research_queries(task_analysis)
        
        # Create research context envelope
        research_envelope_id = self.storage.create_context_envelope(
            session_id=session_id,
            context_type='research',
            content={
                'research_queries': research_queries,
                'domain_specializations': task_analysis.domain_specialization
            },
            parent_envelope_id=context_envelope_id,
            ttl_seconds=1800  # 30 minutes TTL
        )
        
        research_results = {}
        
        # Assign specialized researchers for different domains
        researchers = await self._assign_domain_researchers(task_analysis.domain_specialization)
        
        # Conduct parallel research
        research_tasks = []
        for i, (query, researcher) in enumerate(zip(research_queries, researchers)):
            task = self._conduct_specialized_research(
                session_id, research_envelope_id, query, researcher, i
            )
            research_tasks.append(task)
        
        # Execute research tasks
        individual_results = await asyncio.gather(*research_tasks, return_exceptions=True)
        
        # Consolidate research findings
        for i, result in enumerate(individual_results):
            if not isinstance(result, Exception):
                research_results[f"research_{i}"] = result
                
                # Store as collaborative artifact
                self.storage.create_artifact(
                    session_id=session_id,
                    artifact_type='research_finding',
                    title=f"Research Finding {i + 1}",
                    content=json.dumps(result),
                    contributors=[result.get('researcher_id', 'unknown')],
                    tags=['research', result.get('domain', 'general')]
                )
        
        # Synthesize research findings
        synthesis = await self._synthesize_research_findings(
            session_id, research_envelope_id, research_results
        )
        
        return {
            'individual_findings': research_results,
            'synthesis': synthesis,
            'research_envelope_id': research_envelope_id,
            'total_researchers': len(researchers)
        }

    async def _conduct_discussion_round(self, session_id: str, context_envelope_id: str,
                                      task_analysis: TaskComplexityAnalysis,
                                      research_artifacts: Dict[str, Any],
                                      round_num: int) -> Dict[str, Any]:
        """Conduct a round of agent discussion and collaboration"""
        
        # Create discussion context envelope
        discussion_envelope_id = self.storage.create_context_envelope(
            session_id=session_id,
            context_type='discussion',
            content={
                'round_number': round_num,
                'research_artifacts': research_artifacts,
                'discussion_topics': self._generate_discussion_topics(task_analysis, research_artifacts)
            },
            parent_envelope_id=context_envelope_id,
            ttl_seconds=900  # 15 minutes TTL
        )
        
        # Get agent team for this complexity level
        agent_team = self.router.create_agent_team(task_analysis)
        
        # Conduct structured discussion
        discussion_contributions = []
        
        for agent_config in agent_team:
            agent_role = agent_config['role']
            backend = agent_config['backend']
            
            # Generate agent's contribution
            contribution = await self._get_agent_discussion_contribution(
                session_id, discussion_envelope_id, agent_role, backend,
                task_analysis, research_artifacts, round_num
            )
            
            discussion_contributions.append(contribution)
            
            # Log the interaction
            self.storage.log_agent_interaction(
                session_id=session_id,
                source_agent=agent_role.value,
                target_agent='discussion_moderator',
                interaction_type='discussion_contribution',
                content=contribution['content'][:500],  # Truncate for storage
                context_envelope_id=discussion_envelope_id,
                success=True,
                duration_ms=contribution.get('generation_time_ms', 0)
            )
        
        # Synthesize discussion round
        round_synthesis = await self._synthesize_discussion_round(
            session_id, discussion_envelope_id, discussion_contributions
        )
        
        return {
            'round_number': round_num,
            'contributions': discussion_contributions,
            'synthesis': round_synthesis,
            'discussion_envelope_id': discussion_envelope_id,
            'participant_count': len(agent_team)
        }

    async def _get_agent_discussion_contribution(self, session_id: str, discussion_envelope_id: str,
                                               agent_role: AgentRole, backend: str,
                                               task_analysis: TaskComplexityAnalysis,
                                               research_artifacts: Dict[str, Any],
                                               round_num: int) -> Dict[str, Any]:
        """Get individual agent's contribution to the discussion"""
        
        start_time = time.time()
        
        # Build context for the agent
        context = {
            'role': agent_role.value,
            'task_complexity': task_analysis.overall_complexity.value,
            'research_findings': research_artifacts.get('synthesis', {}),
            'round_number': round_num,
            'specialization': task_analysis.domain_specialization
        }
        
        # Generate role-specific prompt
        prompt = self._build_agent_discussion_prompt(agent_role, context)
        
        try:
            # Make LLM request using selected backend
            response = await self._make_backend_request(backend, prompt, agent_role)
            
            contribution = {
                'agent_role': agent_role.value,
                'backend_used': backend,
                'content': response,
                'key_points': self._extract_key_points(response),
                'recommendations': self._extract_recommendations(response),
                'concerns': self._extract_concerns(response),
                'generation_time_ms': int((time.time() - start_time) * 1000),
                'success': True
            }
            
            # Log performance
            self.storage.log_agent_performance(
                agent_role=agent_role.value,
                backend_used=backend,
                task_type='discussion_contribution',
                complexity_level=task_analysis.overall_complexity.value,
                execution_time_ms=contribution['generation_time_ms'],
                success=True,
                quality_score=0.8  # Default quality score
            )
            
            return contribution
            
        except Exception as e:
            error_contribution = {
                'agent_role': agent_role.value,
                'backend_used': backend,
                'error': str(e),
                'generation_time_ms': int((time.time() - start_time) * 1000),
                'success': False
            }
            
            # Log error
            self.storage.log_agent_performance(
                agent_role=agent_role.value,
                backend_used=backend,
                task_type='discussion_contribution',
                complexity_level=task_analysis.overall_complexity.value,
                execution_time_ms=error_contribution['generation_time_ms'],
                success=False,
                error_type=type(e).__name__
            )
            
            return error_contribution

    def _build_agent_discussion_prompt(self, agent_role: AgentRole, context: Dict[str, Any]) -> str:
        """Build role-specific discussion prompt"""
        
        base_prompt = f"""You are a {agent_role.value} participating in a collaborative problem-solving session.

Task Context:
- Complexity Level: {context['task_complexity']}
- Discussion Round: {context['round_number']}
- Specialization Areas: {', '.join(context.get('specialization', []))}

Research Findings:
{json.dumps(context.get('research_findings', {}), indent=2)[:1000]}

As a {agent_role.value}, provide your specialized perspective focusing on:
"""

        role_specific_guidance = {
            AgentRole.PLANNER: """
- Task breakdown and sequencing
- Resource requirements and timeline
- Risk assessment and mitigation strategies
- Dependencies and coordination needs""",
            
            AgentRole.CODER: """
- Implementation approach and architecture
- Technical feasibility and constraints
- Code quality and testing considerations
- Integration and deployment aspects""",
            
            AgentRole.REVIEWER: """
- Quality assurance and validation
- Potential issues and edge cases
- Best practices and standards compliance
- Security and performance implications""",
            
            AgentRole.RESEARCHER: """
- Additional research needs and gaps
- Evidence evaluation and source credibility
- Methodological considerations
- Related work and precedents""",
            
            AgentRole.DOMAIN_EXPERT: """
- Domain-specific requirements and constraints
- Industry standards and regulations
- Specialized knowledge and insights
- Compliance and best practice recommendations""",
            
            AgentRole.SECURITY_EXPERT: """
- Security vulnerabilities and threats
- Authentication and authorization requirements
- Data protection and privacy concerns
- Compliance with security standards"""
        }
        
        prompt = base_prompt + role_specific_guidance.get(agent_role, "- Your specialized expertise and recommendations")
        
        prompt += """

Provide a structured response with:
1. Key insights from your perspective
2. Specific recommendations
3. Potential concerns or risks
4. Questions for other team members

Keep your response focused and actionable."""
        
        return prompt

    async def _make_backend_request(self, backend: str, prompt: str, agent_role: AgentRole) -> str:
        """Make LLM request to the specified backend"""
        
        if backend == 'anthropic':
            return await self._make_anthropic_request(prompt, agent_role)
        elif backend == 'openai':
            return await self._make_openai_request(prompt, agent_role)
        else:  # local backend
            return await self._make_local_request(prompt, agent_role)

    async def _make_anthropic_request(self, prompt: str, agent_role: AgentRole) -> str:
        """Make request to Anthropic Claude"""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Select model based on agent role
            model = "claude-3-5-sonnet-20241022" if agent_role in [
                AgentRole.DOMAIN_EXPERT, AgentRole.SECURITY_EXPERT
            ] else "claude-3-haiku-20240307"
            
            message = await client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Error making Anthropic request: {str(e)}"

    async def _make_openai_request(self, prompt: str, agent_role: AgentRole) -> str:
        """Make request to OpenAI"""
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Select model based on complexity
            model = "gpt-4o" if agent_role in [
                AgentRole.DOMAIN_EXPERT, AgentRole.SECURITY_EXPERT
            ] else "gpt-4o-mini"
            
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error making OpenAI request: {str(e)}"

    async def _make_local_request(self, prompt: str, agent_role: AgentRole) -> str:
        """Make request to local LM Studio"""
        try:
            import aiohttp
            
            base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": os.getenv("MODEL_NAME", "openai/gpt-oss-20b"),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                ) as response:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            return f"Error making local request: {str(e)}"

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from agent response"""
        # Simple extraction - could be enhanced with NLP
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '• ', '* ')) or line[0:2].isdigit():
                key_points.append(line.lstrip('- •*0123456789. '))
        
        return key_points[:5]  # Limit to top 5 points

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from agent response"""
        # Look for recommendation sections
        recommendations = []
        in_recommendations = False
        
        for line in text.split('\n'):
            line = line.strip().lower()
            if 'recommend' in line or 'suggest' in line:
                in_recommendations = True
            elif in_recommendations and line.startswith(('- ', '• ', '* ')):
                recommendations.append(line.lstrip('- •* '))
        
        return recommendations[:3]  # Limit to top 3

    def _extract_concerns(self, text: str) -> List[str]:
        """Extract concerns from agent response"""
        concerns = []
        in_concerns = False
        
        for line in text.split('\n'):
            line_lower = line.strip().lower()
            if any(word in line_lower for word in ['concern', 'risk', 'issue', 'problem']):
                in_concerns = True
            elif in_concerns and line.strip().startswith(('- ', '• ', '* ')):
                concerns.append(line.strip().lstrip('- •* '))
        
        return concerns[:3]  # Limit to top 3

    # Additional helper methods would go here...
    # (Implementation continues with synthesis methods, research query generation, etc.)

    def _calculate_execution_time(self, session_id: str) -> float:
        """Calculate total execution time for a session"""
        session_envelope = self.storage.get_session_context_chain(session_id)
        if not session_envelope:
            return 0.0
        
        start_time = min(env.created_at for env in session_envelope)
        end_time = max(env.updated_at for env in session_envelope)
        
        return (end_time - start_time) / 60.0  # Convert to minutes

    def _extract_contributors(self, session_id: str) -> List[str]:
        """Extract list of contributors for a session"""
        # This would query the interactions table to find unique agents
        # Simplified implementation
        return ["planner", "researcher", "coder", "reviewer"]