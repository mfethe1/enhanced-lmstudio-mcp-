"""
Lightweight metrics module with Prometheus support (optional).
If prometheus_client is unavailable, falls back to no-op so code paths remain safe.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except Exception:
    # Fallback no-op shims
    _PROM_AVAILABLE = False
    class _Noop:
        def labels(self, *_, **__):
            return self
        def inc(self, *_):
            return None
        def observe(self, *_):
            return None
        def set(self, *_):
            return None
    Counter = Histogram = Gauge = _Noop  # type: ignore
    def generate_latest(*_args, **_kwargs):
        return b""
    CONTENT_TYPE_LATEST = "text/plain"

# Single registry for the process
_REGISTRY: Optional[CollectorRegistry] = None

# Metrics objects
REQUESTS_TOTAL = None
REQUEST_LATENCY = None
EMPTY_RESPONSES = None
CIRCUIT_STATE = None


def _get_registry():
    global _REGISTRY
    if not _PROM_AVAILABLE:
        return None
    if _REGISTRY is None:
        _REGISTRY = CollectorRegistry()
    return _REGISTRY


def init_metrics():
    global REQUESTS_TOTAL, REQUEST_LATENCY, EMPTY_RESPONSES, CIRCUIT_STATE
    if REQUESTS_TOTAL is not None:
        return
    reg = _get_registry()
    labelnames = ["backend"]
    REQUESTS_TOTAL = Counter("mcp_requests_total", "Total requests by backend and outcome", labelnames + ["outcome"], registry=reg) if _PROM_AVAILABLE else Counter()
    REQUEST_LATENCY = Histogram("mcp_request_latency_seconds", "Request latency by backend", labelnames, registry=reg) if _PROM_AVAILABLE else Histogram()
    EMPTY_RESPONSES = Counter("mcp_empty_responses_total", "Empty responses by backend", labelnames, registry=reg) if _PROM_AVAILABLE else Counter()
    CIRCUIT_STATE = Gauge("mcp_circuit_breaker_state", "Circuit breaker state per backend (0 closed, 0.5 half-open, 1 open)", labelnames, registry=reg) if _PROM_AVAILABLE else Gauge()


# Initialize eagerly if enabled
if os.getenv("METRICS_ENABLED", "1").lower() in {"1", "true", "yes", "on"}:
    init_metrics()


def record_backend_result(backend: str, success: bool, latency_ms: int, empty: bool = False):
    if REQUESTS_TOTAL is None or REQUEST_LATENCY is None or EMPTY_RESPONSES is None:
        return
    outcome = "success" if success else "failure"
    try:
        REQUESTS_TOTAL.labels(backend=backend, outcome=outcome).inc()
        REQUEST_LATENCY.labels(backend=backend).observe(max(0.0, latency_ms / 1000.0))
        if empty:
            EMPTY_RESPONSES.labels(backend=backend).inc()
    except Exception:
        pass


def set_circuit_state(backend: str, state: str):
    if CIRCUIT_STATE is None:
        return
    mapping = {"CLOSED": 0.0, "HALF_OPEN": 0.5, "OPEN": 1.0}
    val = mapping.get(state, 0.0)
    try:
        CIRCUIT_STATE.labels(backend=backend).set(val)
    except Exception:
        pass


def metrics_payload_bytes() -> bytes:
    reg = _get_registry()
    try:
        return generate_latest(reg) if reg else b""
    except Exception:
        return b""
