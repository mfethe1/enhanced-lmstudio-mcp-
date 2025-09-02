from __future__ import annotations

import json
import time
from typing import Dict, Any


def handle_get_router_diagnostics(arguments: Dict[str, Any], server) -> str:
    """Show recent router decisions, latencies, and backend usage metrics.
    Returns JSON with router_log entries, backend usage counts, and average latencies.
    """
    limit = int(arguments.get("limit", 20))
    
    # Get recent router log entries
    router_log = getattr(server, '_router_log', [])
    recent_entries = router_log[-limit:] if router_log else []
    
    # Calculate backend usage metrics
    backend_counts = {}
    latency_sums = {}
    latency_counts = {}
    success_counts = {}
    error_counts = {}
    
    for entry in router_log:
        backend = entry.get("backend", "unknown")
        success = entry.get("success", False)
        latency = entry.get("latency_ms", 0)
        
        # Count backend usage
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        
        # Track latencies
        if latency > 0:
            latency_sums[backend] = latency_sums.get(backend, 0) + latency
            latency_counts[backend] = latency_counts.get(backend, 0) + 1
        
        # Track success/error rates
        if success:
            success_counts[backend] = success_counts.get(backend, 0) + 1
        else:
            error_counts[backend] = error_counts.get(backend, 0) + 1
    
    # Calculate average latencies
    avg_latencies = {}
    for backend in latency_sums:
        if latency_counts[backend] > 0:
            avg_latencies[backend] = round(latency_sums[backend] / latency_counts[backend], 2)
    
    # Calculate success rates
    success_rates = {}
    for backend in backend_counts:
        total = backend_counts[backend]
        successes = success_counts.get(backend, 0)
        success_rates[backend] = round(successes / total * 100, 1) if total > 0 else 0.0
    
    # Get router configuration
    router_config = {
        "router_use_agent": getattr(server, '_router_use_agent', True),
        "router_min_interval": getattr(server, '_router_min_interval', 0.2),
        "anthropic_smart_switch": True,  # Assume enabled if not explicitly disabled
        "available_backends": []
    }
    
    # Check which backends are available
    import os
    if os.getenv("OPENAI_API_KEY"):
        router_config["available_backends"].append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        router_config["available_backends"].append("anthropic")
    if os.getenv("OPENAI_API_BASE") or os.getenv("LMSTUDIO_API_BASE"):
        router_config["available_backends"].append("lmstudio")
    
    # Compile diagnostics report
    diagnostics = {
        "timestamp": time.time(),
        "total_requests": len(router_log),
        "recent_entries": recent_entries,
        "backend_usage": {
            "counts": backend_counts,
            "avg_latency_ms": avg_latencies,
            "success_rates_pct": success_rates,
            "error_counts": error_counts
        },
        "router_config": router_config,
        "recommendations": []
    }
    
    # Add recommendations based on metrics
    for backend, error_count in error_counts.items():
        total_count = backend_counts.get(backend, 0)
        if total_count > 0 and error_count / total_count > 0.2:
            diagnostics["recommendations"].append(f"High error rate for {backend}: {error_count}/{total_count}")
    
    for backend, avg_lat in avg_latencies.items():
        if avg_lat > 10000:  # > 10 seconds
            diagnostics["recommendations"].append(f"High latency for {backend}: {avg_lat}ms average")
    
    if not router_config["available_backends"]:
        diagnostics["recommendations"].append("No backend API keys configured - only LM Studio local will be available")
    
    return json.dumps(diagnostics, indent=2)
