"""
Production configuration management for LM Studio MCP server.
Handles environment-specific settings and optimizations.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration settings"""
    
    # Core settings
    expose_public_only: bool = True
    log_level: str = "WARNING"
    proactive_research_enabled: bool = False
    
    # Performance settings
    http_connect_timeout: int = 2
    http_read_timeout_simple: int = 8
    http_read_timeout_complex: int = 45
    
    # Model monitoring
    model_monitor_interval: int = 300  # 5 minutes
    model_alert_threshold: int = 1
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    firecrawl_failure_threshold: int = 3
    openai_failure_threshold: int = 5
    anthropic_failure_threshold: int = 5
    lmstudio_failure_threshold: int = 3
    
    # Metrics and monitoring
    metrics_enabled: bool = True
    metrics_port: int = 9099
    performance_alerts: bool = True
    performance_threshold: float = 0.2
    
    @classmethod
    def from_environment(cls) -> 'ProductionConfig':
        """Create configuration from environment variables"""
        return cls(
            expose_public_only=_env_bool('EXPOSE_PUBLIC_ONLY', True),
            log_level=os.getenv('LOG_LEVEL', 'WARNING').upper(),
            proactive_research_enabled=_env_bool('PROACTIVE_RESEARCH_ENABLED', False),
            
            http_connect_timeout=_env_int('HTTP_CONNECT_TIMEOUT', 2),
            http_read_timeout_simple=_env_int('HTTP_READ_TIMEOUT_SIMPLE', 8),
            http_read_timeout_complex=_env_int('HTTP_READ_TIMEOUT_COMPLEX', 45),
            
            model_monitor_interval=_env_int('MODEL_MONITOR_INTERVAL', 300),
            model_alert_threshold=_env_int('MODEL_ALERT_THRESHOLD', 1),
            
            circuit_breaker_enabled=_env_bool('CIRCUIT_BREAKER_ENABLED', True),
            firecrawl_failure_threshold=_env_int('FIRECRAWL_FAILURE_THRESHOLD', 3),
            openai_failure_threshold=_env_int('OPENAI_FAILURE_THRESHOLD', 5),
            anthropic_failure_threshold=_env_int('ANTHROPIC_FAILURE_THRESHOLD', 5),
            lmstudio_failure_threshold=_env_int('LMSTUDIO_FAILURE_THRESHOLD', 3),
            
            metrics_enabled=_env_bool('METRICS_ENABLED', True),
            metrics_port=_env_int('METRICS_PORT', 9099),
            performance_alerts=_env_bool('PERFORMANCE_ALERTS', True),
            performance_threshold=_env_float('PERFORMANCE_THRESHOLD', 0.2),
        )
    
    def apply_logging_config(self):
        """Apply logging configuration"""
        log_level_int = getattr(logging, self.log_level, logging.WARNING)
        logging.getLogger().setLevel(log_level_int)
        
        # Configure specific loggers for production
        if self.log_level in ('WARNING', 'ERROR', 'CRITICAL'):
            # Reduce noise from external libraries
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        logger.info(f"Logging configured for production: {self.log_level}")
    
    def get_circuit_breaker_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker configurations for each service"""
        return {
            'firecrawl': {
                'failure_threshold': self.firecrawl_failure_threshold,
                'recovery_timeout': 120.0,
                'success_threshold': 2,
                'timeout': 60.0
            },
            'openai': {
                'failure_threshold': self.openai_failure_threshold,
                'recovery_timeout': 60.0,
                'success_threshold': 2,
                'timeout': 30.0
            },
            'anthropic': {
                'failure_threshold': self.anthropic_failure_threshold,
                'recovery_timeout': 60.0,
                'success_threshold': 2,
                'timeout': 30.0
            },
            'lmstudio': {
                'failure_threshold': self.lmstudio_failure_threshold,
                'recovery_timeout': 30.0,
                'success_threshold': 1,
                'timeout': 45.0
            }
        }
    
    def validate(self) -> list[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if self.http_connect_timeout <= 0:
            issues.append("HTTP connect timeout must be positive")
        
        if self.http_read_timeout_simple <= 0:
            issues.append("HTTP read timeout (simple) must be positive")
        
        if self.http_read_timeout_complex <= 0:
            issues.append("HTTP read timeout (complex) must be positive")
        
        if self.model_monitor_interval < 60:
            issues.append("Model monitor interval should be at least 60 seconds")
        
        if self.performance_threshold <= 0:
            issues.append("Performance threshold must be positive")
        
        if self.log_level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            issues.append(f"Invalid log level: {self.log_level}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'expose_public_only': self.expose_public_only,
            'log_level': self.log_level,
            'proactive_research_enabled': self.proactive_research_enabled,
            'http_connect_timeout': self.http_connect_timeout,
            'http_read_timeout_simple': self.http_read_timeout_simple,
            'http_read_timeout_complex': self.http_read_timeout_complex,
            'model_monitor_interval': self.model_monitor_interval,
            'model_alert_threshold': self.model_alert_threshold,
            'circuit_breaker_enabled': self.circuit_breaker_enabled,
            'metrics_enabled': self.metrics_enabled,
            'metrics_port': self.metrics_port,
            'performance_alerts': self.performance_alerts,
            'performance_threshold': self.performance_threshold,
        }


def _env_bool(key: str, default: bool) -> bool:
    """Parse boolean from environment variable"""
    value = os.getenv(key, '').lower()
    if value in ('1', 'true', 'yes', 'on', 'enabled'):
        return True
    elif value in ('0', 'false', 'no', 'off', 'disabled'):
        return False
    else:
        return default


def _env_int(key: str, default: int) -> int:
    """Parse integer from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    """Parse float from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# Global production configuration instance
production_config = ProductionConfig.from_environment()


def get_production_config() -> ProductionConfig:
    """Get the global production configuration"""
    return production_config


def validate_production_config() -> bool:
    """Validate production configuration and log any issues"""
    issues = production_config.validate()
    
    if issues:
        logger.error("Production configuration issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Production configuration validated successfully")
    return True


def log_production_config():
    """Log current production configuration"""
    config_dict = production_config.to_dict()
    logger.info("Production configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")


def create_production_mcp_json() -> Dict[str, Any]:
    """Create optimized mcp.json configuration for production"""
    return {
        "mcpServers": {
            "enhanced-lmstudio-mcp": {
                "command": "python",
                "args": ["server.py"],
                "env": {
                    "EXPOSE_PUBLIC_ONLY": "1",
                    "PROACTIVE_RESEARCH_ENABLED": "0",
                    "LOG_LEVEL": "WARNING",
                    "HTTP_CONNECT_TIMEOUT": "2",
                    "HTTP_READ_TIMEOUT_SIMPLE": "8",
                    "HTTP_READ_TIMEOUT_COMPLEX": "45",
                    "MODEL_MONITOR_INTERVAL": "300",
                    "CIRCUIT_BREAKER_ENABLED": "1",
                    "METRICS_ENABLED": "1",
                    "PERFORMANCE_ALERTS": "1",
                    "PERFORMANCE_THRESHOLD": "0.2"
                }
            }
        }
    }
