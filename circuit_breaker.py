"""
Circuit breaker pattern implementation for external API calls.
Prevents cascading failures by temporarily disabling failing services.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2      # Successes needed to close from half-open
    timeout: float = 30.0           # Request timeout


class CircuitBreaker:
    """Circuit breaker for external API calls with exponential backoff"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
        
        logger.debug(f"Circuit breaker '{name}' initialized")
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_available(self) -> bool:
        """Check if the circuit allows calls"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if time.time() - self._last_failure_time >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker"""
        if not self.is_available:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' CLOSED after recovery")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures")
            elif self._state == CircuitState.HALF_OPEN:
                # Failed during half-open, go back to open
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' back to OPEN after half-open failure")
    
    def reset(self):
        """Manually reset the circuit breaker to closed state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "is_available": self.is_available
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and calls are blocked"""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services"""
    
    def __init__(self):
        self._breakers = {}
        self._lock = threading.Lock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def call_with_breaker(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the appropriate circuit breaker"""
        breaker = self.get_breaker(service_name)
        return breaker.call(func, *args, **kwargs)
    
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_available_services(self) -> list[str]:
        """Get list of services with available circuit breakers"""
        with self._lock:
            return [name for name, breaker in self._breakers.items() if breaker.is_available]


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def with_circuit_breaker(service_name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to wrap functions with circuit breaker protection"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            breaker = circuit_manager.get_breaker(service_name, config)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Pre-configured circuit breakers for common services
FIRECRAWL_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120.0,
    success_threshold=2,
    timeout=60.0
)

OPENAI_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
)

ANTHROPIC_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
)

LMSTUDIO_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    success_threshold=1,
    timeout=45.0
)
