import os
import asyncio
import logging
from typing import Optional, Dict, Any
import json
import time

logger = logging.getLogger(__name__)

# Import the RouterAgent from previous artifact
from router_agent_system import RouterAgent, Backend, TaskProfile


class EnhancedRoutingMixin:
    """Mixin to enhance the server with intelligent routing"""
    
    def __init__(self):
        super().__init__()
        self._router_agent: Optional[RouterAgent] = None
        self._router_initialized = False
        self._router_fallback_mode = False
        
        # Initialize router in background to not block startup
        asyncio.create_task(self._initialize_router())
        
    async def _initialize_router(self):
        """Initialize router agent asynchronously"""
        try:
            self._router_agent = RouterAgent(self.storage, self.base_url)
            self._router_initialized = True
            logger.info("Router agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize router agent: {e}")
            self._router_fallback_mode = True
            
    async def route_chat(self, prompt: str, *, 
                        intent: Optional[str] = None,
                        complexity: Optional[str] = "auto",
                        role: Optional[str] = None,
                        preferred_backend: Optional[str] = None,
                        temperature: float = 0.2,
                        max_retries: int = 2) -> str:
        """Enhanced routing with intelligent agent-based selection"""
        
        # If preferred backend specified, try it first
        if preferred_backend:
            try:
                return await self._execute_on_backend(
                    prompt, preferred_backend, temperature, intent, role
                )
            except Exception as e:
                logger.warning(f"Preferred backend {preferred_backend} failed: {e}")
                # Continue to intelligent routing
                
        # Use router agent if available
        if self._router_initialized and not self._router_fallback_mode:
            try:
                return await self._route_via_agent(
                    prompt, intent, complexity, role, temperature, max_retries
                )
            except Exception as e:
                logger.error(f"Router agent failed: {e}")
                self._router_fallback_mode = True
                
        # Fallback to heuristic routing
        return await self._heuristic_route(
            prompt, intent, complexity, role, temperature
        )
        
    async def _route_via_agent(self, prompt: str, intent: Optional[str],
                              complexity: str, role: Optional[str],
                              temperature: float, max_retries: int) -> str:
        """Route using the intelligent router agent"""
        
        # Analyze task
        task = await self._router_agent.analyze_task(prompt, intent, role)
        if complexity != "auto":
            task.complexity = complexity
            
        # Get routing decision
        decision = await self._router_agent.decide_routing(task)
        
        logger.info(
            f"Router decision: {decision.backend.value} ({decision.model}) "
            f"confidence={decision.confidence:.2f} reason='{decision.reasoning}'"
        )
        
        # Try primary backend
        start_time = time.time()
        try:
            result = await self._execute_on_backend(
                prompt,
                decision.backend.value,
                temperature,
                intent,
                role,
                model_override=decision.model
            )
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self._router_agent._update_performance(
                decision.backend, latency_ms, success=True
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Primary backend {decision.backend.value} failed: {e}")
            
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self._router_agent._update_performance(
                decision.backend, latency_ms, success=False
            )
            
            # Try fallback chain
            for fallback in decision.fallback_chain[:max_retries]:
                try:
                    logger.info(f"Trying fallback: {fallback.value}")
                    return await self._execute_on_backend(
                        prompt,
                        fallback.value,
                        temperature,
                        intent,
                        role
                    )
                except Exception as e2:
                    logger.warning(f"Fallback {fallback.value} failed: {e2}")
                    continue
                    
            # All backends failed
            raise Exception(f"All backends failed. Last error: {e}")
            
    async def _execute_on_backend(self, prompt: str, backend: str,
                                 temperature: float, intent: Optional[str],
                                 role: Optional[str],
                                 model_override: Optional[str] = None) -> str:
        """Execute request on specific backend"""
        
        backend_lower = backend.lower()
        
        # Rate limiting
        self._router_wait(backend_lower)
        
        max_tokens = int(os.getenv("ROUTER_MAX_TOKENS", "2000"))
        
        if backend_lower == "openai" and os.getenv("OPENAI_API_KEY"):
            from server import _http_client
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            data = _http_client.post_sync(
                f"{base}/chat/completions",
                json_data={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                operation_type="simple"
            )
            return _sanitize_llm_output(
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
                
        elif backend_lower == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from server import _http_client
            base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            model = model_override or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

            data = _http_client.post_sync(
                f"{base}/messages",
                json_data={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                },
                headers={
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                    "content-type": "application/json",
                },
                operation_type="simple"
            )
            parts = data.get("content", [])
            if parts and isinstance(parts, list):
                txt = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
                return _sanitize_llm_output(txt)
            return _sanitize_llm_output("")
                
        else:
            # LMStudio/local backend
            model = model_override or self.model_name
            return await self.make_llm_request_with_retry(
                prompt, temperature=temperature, retries=2, backoff=0.5
            )
            
    async def _heuristic_route(self, prompt: str, intent: Optional[str],
                              complexity: str, role: Optional[str],
                              temperature: float) -> str:
        """Fallback heuristic routing (existing logic)"""
        text = (prompt or "").lower()
        
        # Complexity detection
        complex_markers = [
            "architecture", "design", "vulnerability", "formal",
            "proof", "theorem", "deep analysis", "multi-step"
        ]
        is_complex = (
            any(w in text for w in complex_markers) or
            complexity == "high" or
            intent in {"analysis", "architecture", "validation"}
        )
        
        # Backend selection
        backend = "lmstudio"  # default
        
        if is_complex:
            if os.getenv("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                backend = "openai"
                
        logger.info(f"Heuristic routing selected: {backend}")
        
        try:
            return await self._execute_on_backend(
                prompt, backend, temperature, intent, role
            )
        except Exception as e:
            logger.warning(f"Heuristic primary backend failed: {e}")
            # Try fallbacks
            fallback_order = ["lmstudio", "openai", "anthropic"]
            for fb in fallback_order:
                if fb != backend:
                    try:
                        return await self._execute_on_backend(
                            prompt, fb, temperature, intent, role
                        )
                    except:
                        continue
            return f"Error: All backends failed"
            
    async def get_router_diagnostics(self, limit: int = 20) -> Dict[str, Any]:
        """Get router diagnostics and recent decisions"""
        if not self._router_initialized:
            return {
                "status": "not_initialized",
                "fallback_mode": self._router_fallback_mode
            }
            
        analytics = await self._router_agent.get_router_analytics(hours=24)
        
        # Get recent decisions
        recent_decisions = self.storage.retrieve_memory(
            category="router_decisions",
            limit=limit
        )
        
        decisions_formatted = []
        for mem in recent_decisions:
            try:
                data = json.loads(mem["value"])
                decisions_formatted.append({
                    "time": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(data["timestamp"])
                    ),
                    "backend": data["backend"],
                    "model": data["model"],
                    "confidence": f"{data['confidence']:.0%}",
                    "reasoning": data["reasoning"]
                })
            except:
                continue
                
        return {
            "status": "active",
            "fallback_mode": self._router_fallback_mode,
            "analytics": analytics,
            "recent_decisions": decisions_formatted
        }


# Update tool handlers for router diagnostics
def handle_router_diagnostics(arguments, server):
    """Enhanced router diagnostics with analytics"""
    limit = int(arguments.get("limit", 20))
    
    # Get diagnostics from enhanced router
    try:
        diagnostics = asyncio.get_event_loop().run_until_complete(
            server.get_router_diagnostics(limit)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            diagnostics = loop.run_until_complete(
                server.get_router_diagnostics(limit)
            )
        finally:
            loop.close()
    except Exception as e:
        return f"Error getting diagnostics: {e}"
        
    # Format output
    output = ["=== Router Diagnostics ===\n"]
    
    output.append(f"Status: {diagnostics['status']}")
    output.append(f"Fallback Mode: {diagnostics.get('fallback_mode', False)}")
    
    if "analytics" in diagnostics:
        analytics = diagnostics["analytics"]
        output.append(f"\nTotal Decisions (24h): {analytics['total_decisions']}")
        output.append(f"Avg Confidence: {analytics['avg_confidence']:.0%}")
        output.append(f"Cache Size: {analytics['cache_size']}")
        
        # Backend distribution
        if analytics["backend_distribution"]:
            output.append("\nBackend Usage:")
            for backend, count in analytics["backend_distribution"].items():
                pct = (count / analytics['total_decisions'] * 100) if analytics['total_decisions'] > 0 else 0
                output.append(f"  {backend}: {count} ({pct:.0f}%)")
                
        # Model distribution
        if analytics["model_distribution"]:
            output.append("\nModel Usage:")
            for model, count in sorted(
                analytics["model_distribution"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:  # Top 5
                output.append(f"  {model}: {count}")
                
        # Backend performance
        if analytics["backend_performance"]:
            output.append("\nBackend Performance:")
            for backend, perf in analytics["backend_performance"].items():
                if perf["total_requests"] > 0:
                    output.append(
                        f"  {backend}: "
                        f"success={perf['success_rate']:.0%} "
                        f"avg_latency={perf['avg_latency_ms']:.0f}ms "
                        f"requests={perf['total_requests']}"
                    )
                    
    # Recent decisions
    if diagnostics.get("recent_decisions"):
        output.append(f"\nRecent Decisions (last {limit}):")
        for decision in diagnostics["recent_decisions"]:
            output.append(
                f"  [{decision['time']}] {decision['backend']} "
                f"({decision['model']}) conf={decision['confidence']} - "
                f"{decision['reasoning']}"
            )
            
    return "\n".join(output)


# Tool to manually test router decisions
def handle_router_test(arguments, server):
    """Test router decision for a given task"""
    task_description = arguments.get("task", "").strip()
    role = arguments.get("role")
    intent = arguments.get("intent")
    complexity = arguments.get("complexity", "auto")
    
    if not task_description:
        raise ValidationError("task is required")
        
    # Get router decision
    try:
        if not server._router_initialized:
            return "Router not initialized"
            
        task = asyncio.get_event_loop().run_until_complete(
            server._router_agent.analyze_task(task_description, intent, role)
        )
        if complexity != "auto":
            task.complexity = complexity
            
        decision = asyncio.get_event_loop().run_until_complete(
            server._router_agent.decide_routing(task)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            task = loop.run_until_complete(
                server._router_agent.analyze_task(task_description, intent, role)
            )
            if complexity != "auto":
                task.complexity = complexity
            decision = loop.run_until_complete(
                server._router_agent.decide_routing(task)
            )
        finally:
            loop.close()
    except Exception as e:
        return f"Error: {e}"
        
    # Format output
    output = [
        "=== Router Test Results ===",
        f"\nTask Analysis:",
        f"  Complexity: {task.complexity}",
        f"  Estimated Tokens: {task.estimated_tokens}",
        f"  Requires Vision: {task.requires_vision}",
        f"  Requires Tools: {task.requires_tools}",
        f"\nRouting Decision:",
        f"  Backend: {decision.backend.value}",
        f"  Model: {decision.model}",
        f"  Confidence: {decision.confidence:.0%}",
        f"  Reasoning: {decision.reasoning}",
        f"  Fallback Chain: {[b.value for b in decision.fallback_chain]}",
    ]
    
    return "\n".join(output)