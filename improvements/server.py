# Updates to integrate into your existing enhanced_lm_studio_mcp_server.py

# 1. Add imports at the top of the file:
"""
from router_agent_system import RouterAgent, Backend, TaskProfile
from enhanced_routing_integration import EnhancedRoutingMixin
"""

# 2. Update the server class to inherit from the mixin:
class EnhancedLMStudioMCPServer(EnhancedRoutingMixin):
    def __init__(self):
        # Call parent init first
        super().__init__()
        
        # Existing initialization code
        self.base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
        self.working_directory = os.getcwd()

        # Initialize persistent storage: legacy facade wrapped by V2 selector when enabled
        self.storage = StorageSelector(_LegacyEnhanced())
        
        # Performance monitoring settings
        self.performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.2"))  # seconds
        
        # Router state (can be removed as it's in the mixin now)
        # self._router_last_call = {"lmstudio": 0.0, "openai": 0.0, "anthropic": 0.0}
        # self._router_min_interval = 1.0 / float(os.getenv("ROUTER_RATE_LIMIT_TPS", "5"))

        logger.info("Enhanced MCP Server initialized with intelligent routing")

    # The route_chat method is now provided by the mixin, so you can remove the old one
    # Remove the old decide_backend_via_router method as well

    # Update make_llm_request_with_retry to use the new routing system:
    async def make_llm_request_with_retry(self, prompt: str, temperature: float = 0.35, 
                                         retries: int = 2, backoff: float = 0.5,
                                         intent: Optional[str] = None,
                                         role: Optional[str] = None) -> str:
        """Centralized LLM request with intelligent routing"""
        
        # If router is available, use it
        if self._router_initialized:
            try:
                return await self.route_chat(
                    prompt,
                    intent=intent,
                    role=role,
                    temperature=temperature,
                    max_retries=retries
                )
            except Exception as e:
                logger.warning(f"Router failed, falling back: {e}")
                
        # Fallback to direct LMStudio call
        import requests
        attempt = 0
        last_err = None
        
        while attempt <= retries:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 2000
                    },
                    timeout=(30, 120),
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"].get("content", "")
                        return _sanitize_llm_output(content)
                    return "Error: No response from model"
                last_err = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_err = str(e)
            attempt += 1
            await asyncio.sleep(backoff * attempt)
            
        return f"Error: LLM request failed after {retries+1} attempts: {last_err}"


# 3. Update handle_llm_analysis_tool to pass role/intent:
def handle_llm_analysis_tool(tool_name, arguments, server):
    """Handle LLM-based analysis tools with intelligent routing"""
    code = arguments.get("code", "")
    compact = bool(arguments.get("compact", False))
    code_only = bool(arguments.get("code_only", False))
    
    # Determine intent and role based on tool
    intent_map = {
        "analyze_code": "code-analysis",
        "explain_code": "code-explanation", 
        "suggest_improvements": "code-improvement",
        "generate_tests": "test-generation"
    }
    
    role_map = {
        "analyze_code": "Analyst",
        "explain_code": "Teacher",
        "suggest_improvements": "Reviewer",
        "generate_tests": "Test Engineer"
    }
    
    intent = intent_map.get(tool_name, "general")
    role = role_map.get(tool_name, "Assistant")

    # Build prompts as before...
    if tool_name == "analyze_code":
        analysis_type = arguments.get("analysis_type", "explanation")
        limit = 8 if compact else 12
        prompt = (
            "You are a concise senior engineer. Provide a short, non-repetitive analysis (<= "
            f"{limit} bullets) focused on {analysis_type}.\n\nCode:\n{code}\n"
        )
    # ... rest of prompt building ...

    # Call with routing hints
    try:
        raw = asyncio.get_event_loop().run_until_complete(
            server.make_llm_request_with_retry(
                prompt, 
                temperature=0.1, 
                retries=2, 
                backoff=0.5,
                intent=intent,
                role=role
            )
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            raw = loop.run_until_complete(
                server.make_llm_request_with_retry(
                    prompt, 
                    temperature=0.1, 
                    retries=2, 
                    backoff=0.5,
                    intent=intent,
                    role=role
                )
            )
        finally:
            loop.close()
    except Exception as e:
        return f"Error: {str(e)}"

    if tool_name == "generate_tests" and code_only:
        return _extract_code_from_text(raw)
    return raw


# 4. Add new tools to get_all_tools():
def get_all_tools():
    """Return all available tools with enhanced capabilities"""
    base = {
        "tools": [
            # ... existing tools ...
            
            # Router diagnostics and testing
            {
                "name": "router_diagnostics",
                "description": "Show router analytics, performance metrics, and recent routing decisions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 20, "description": "Number of recent decisions to show"}
                    }
                }
            },
            {
                "name": "router_test", 
                "description": "Test router decision for a specific task without executing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description to route"},
                        "role": {"type": "string", "description": "Agent role (optional)"},
                        "intent": {"type": "string", "description": "Task intent (optional)"},
                        "complexity": {"type": "string", "enum": ["low", "medium", "high", "auto"], "default": "auto"}
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "router_update_profile",
                "description": "Update backend profile (performance, capabilities)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "backend": {"type": "string", "enum": ["lmstudio", "openai", "anthropic"]},
                        "updates": {"type": "object", "description": "Profile updates"}
                    },
                    "required": ["backend", "updates"]
                }
            }
        ]
    }
    return _merged_tools(base)


# 5. Add handler for router_update_profile:
def handle_router_update_profile(arguments, server):
    """Update backend profile dynamically"""
    backend_str = arguments.get("backend", "").lower()
    updates = arguments.get("updates", {})
    
    if not backend_str or not updates:
        raise ValidationError("backend and updates are required")
        
    if not server._router_initialized:
        return "Router not initialized"
        
    # Map string to enum
    backend_map = {
        "lmstudio": Backend.LMSTUDIO,
        "openai": Backend.OPENAI,
        "anthropic": Backend.ANTHROPIC
    }
    
    backend = backend_map.get(backend_str)
    if not backend:
        raise ValidationError(f"Invalid backend: {backend_str}")
        
    # Update profile
    profile = server._router_agent.backend_profiles.get(backend)
    if not profile:
        return f"Backend {backend_str} not found"
        
    # Apply updates
    updated_fields = []
    for key, value in updates.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
            updated_fields.append(f"{key}={value}")
            
    return f"Updated {backend_str} profile: {', '.join(updated_fields)}"


# 6. Update the registry in handle_tool_call to include new handlers:
registry = {
    # ... existing tools ...
    
    # Router tools
    "router_diagnostics": (handle_router_diagnostics, True),
    "router_test": (handle_router_test, True), 
    "router_update_profile": (handle_router_update_profile, True),
}


# 7. Update agent team functions to use router for agent creation:
def _create_agent(role: str, goal: str, backstory: str, task_desc: str):
    """Create agent with intelligent backend selection"""
    try:
        from crewai import Agent
        
        # Get backend recommendation from enhanced module first
        backend = _enh_decide_backend_for_role(role, task_desc)
        
        # Build LLM for the selected backend
        llm = _build_llm_for_backend(backend)
        
        kwargs = {"allow_delegation": False, "verbose": False}
        if llm is not None:
            kwargs["llm"] = llm
            
        logger.info("Creating agent '%s' -> backend=%s model=%s", 
                   role, backend, getattr(llm, 'model', 'n/a'))
        
        return Agent(role=role, goal=goal, backstory=backstory, **kwargs)
        
    except Exception as e:
        logger.warning("Failed to create agent '%s': %s", role, e)
        return None