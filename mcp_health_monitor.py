#!/usr/bin/env python3
"""
MCP Server Health Monitor and Diagnostics System
Provides comprehensive health checks, monitoring, and diagnostics for the MCP server
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import queue

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_health.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError as e:
    logger.warning(f"aiohttp dependency missing: {e}")
    aiohttp = None

try:
    import psutil
except ImportError as e:
    logger.warning(f"psutil dependency missing: {e}")
    psutil = None

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    from prometheus_client.exposition import start_http_server
    prometheus_client_available = True
except ImportError as e:
    logger.warning(f"prometheus_client dependency missing: {e}")
    prometheus_client_available = False
    Counter = Gauge = Histogram = None
    start_http_server = None

class HealthStatus:
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class MCPHealthMonitor:
    """Comprehensive health monitoring system for MCP server"""
    
    def __init__(self, config_path: str = "mcp.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.health_history = []
        self.metrics = {}
        self.alerts = queue.Queue()
        self.monitoring = False
        self._init_metrics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration"""
        try:
            with open(self.config_path, 'r') as f:
                mcp_config = json.load(f)
                
            # Extract Jarvis server config
            jarvis_config = mcp_config.get('mcpServers', {}).get('jarvis', {})
            env = jarvis_config.get('env', {})
            
            return {
                'server': {
                    'command': jarvis_config.get('command', 'python'),
                    'args': jarvis_config.get('args', []),
                    'type': jarvis_config.get('type', 'stdio')
                },
                'providers': {
                    'openai': {
                        'enabled': bool(os.getenv('OPENAI_API_KEY')),
                        'base_url': env.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                        'models': {
                            'primary': env.get('OPENAI_MODEL', 'gpt-5-chat-latest'),
                            'reasoning': env.get('OPENAI_REASONING_MODEL', 'gpt-5-chat-latest'),
                            'coding': env.get('OPENAI_CODING_MODEL', 'gpt-5-chat-latest'),
                            'fallback': env.get('OPENAI_FALLBACK_MODEL', 'gpt-4o')
                        }
                    },
                    'anthropic': {
                        'enabled': bool(os.getenv('ANTHROPIC_API_KEY')),
                        'base_url': env.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
                        'models': {
                            'primary': env.get('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929'),
                            'complex': env.get('ANTHROPIC_MODEL_COMPLEX', 'claude-opus-4-1-20250805'),
                            'overseer': env.get('ANTHROPIC_MODEL_OVERSEER', 'claude-opus-4-1-20250805')
                        }
                    },
                    'lmstudio': {
                        'enabled': env.get('USE_LOCAL_FOR_SIMPLE', '1') == '1',
                        'base_url': env.get('LMSTUDIO_API_BASE', 'http://localhost:1234/v1'),
                        'model': env.get('LMSTUDIO_MODEL', 'openai/gpt-oss-20b')
                    }
                },
                'circuit_breaker': {
                    'enabled': env.get('CIRCUIT_BREAKER_ENABLED', '1') == '1',
                    'thresholds': {
                        'lmstudio': int(env.get('CIRCUIT_LMSTUDIO_THRESHOLD', '10')),
                        'openai': int(env.get('CIRCUIT_OPENAI_THRESHOLD', '6')),
                        'anthropic': int(env.get('CIRCUIT_ANTHROPIC_THRESHOLD', '6'))
                    },
                    'recovery_times': {
                        'lmstudio': int(env.get('CIRCUIT_LMSTUDIO_RECOVERY', '180')),
                        'openai': int(env.get('CIRCUIT_OPENAI_RECOVERY', '90')),
                        'anthropic': int(env.get('CIRCUIT_ANTHROPIC_RECOVERY', '90'))
                    }
                },
                'timeouts': {
                    'connect': int(env.get('HTTP_CONNECT_TIMEOUT', '10')),
                    'read_simple': int(env.get('HTTP_READ_TIMEOUT_SIMPLE', '60')),
                    'read_complex': int(env.get('HTTP_READ_TIMEOUT_COMPLEX', '180'))
                },
                'monitoring': {
                    'metrics_enabled': env.get('METRICS_EXPORTER', '1') == '1',
                    'metrics_bind': env.get('METRICS_BIND', '127.0.0.1:9099')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _init_metrics(self):
        """Initialize Prometheus metrics if available"""
        if prometheus_client_available:
            self.metrics['health_score'] = Gauge('mcp_health_score', 'Overall health score (0-100)')
            self.metrics['provider_status'] = Gauge('mcp_provider_status', 'Provider status', ['provider'])
            self.metrics['response_time'] = Histogram('mcp_response_time_seconds', 'Response time', ['provider', 'model'])
            self.metrics['error_count'] = Counter('mcp_errors_total', 'Total errors', ['provider', 'error_type'])
            self.metrics['request_count'] = Counter('mcp_requests_total', 'Total requests', ['provider', 'model'])
            self.metrics['memory_usage'] = Gauge('mcp_memory_usage_bytes', 'Memory usage in bytes')
            self.metrics['cpu_usage'] = Gauge('mcp_cpu_usage_percent', 'CPU usage percentage')
    
    async def check_provider_health(self, provider: str) -> Tuple[str, Dict[str, Any]]:
        """Check health of a specific provider"""
        provider_config = self.config['providers'].get(provider, {})
        
        if not provider_config.get('enabled'):
            return HealthStatus.UNHEALTHY, {'reason': 'Provider disabled'}
        
        results = {
            'provider': provider,
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check API connectivity
        base_url = provider_config.get('base_url')
        if not base_url:
            return HealthStatus.CRITICAL, {'reason': 'No base URL configured'}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Provider-specific health checks
                if provider == 'openai':
                    status, details = await self._check_openai_health(session, provider_config)
                elif provider == 'anthropic':
                    status, details = await self._check_anthropic_health(session, provider_config)
                elif provider == 'lmstudio':
                    status, details = await self._check_lmstudio_health(session, provider_config)
                else:
                    status = HealthStatus.UNHEALTHY
                    details = {'reason': 'Unknown provider'}
                
                results['status'] = status
                results['details'] = details
                
                # Update metrics
                if self.metrics.get('provider_status'):
                    self.metrics['provider_status'].labels(provider=provider).set(
                        1 if status == HealthStatus.HEALTHY else 0
                    )
                
                return status, results
                
        except Exception as e:
            logger.error(f"Health check failed for {provider}: {e}")
            if self.metrics.get('error_count'):
                self.metrics['error_count'].labels(provider=provider, error_type='health_check').inc()
            return HealthStatus.CRITICAL, {'reason': str(e)}
    
    async def _check_openai_health(self, session: aiohttp.ClientSession, 
                                  config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Check OpenAI API health"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return HealthStatus.CRITICAL, {'reason': 'No API key configured'}
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            # Test with a minimal request
            start_time = time.time()
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json={
                    'model': config['models']['fallback'],  # Use fallback for health checks
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'max_tokens': 1
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                latency = time.time() - start_time
                
                if self.metrics.get('response_time'):
                    self.metrics['response_time'].labels(
                        provider='openai',
                        model=config['models']['fallback']
                    ).observe(latency)
                
                if response.status == 200:
                    return HealthStatus.HEALTHY, {
                        'latency': latency,
                        'models_available': list(config['models'].values())
                    }
                elif response.status == 429:
                    return HealthStatus.DEGRADED, {
                        'reason': 'Rate limited',
                        'status_code': response.status
                    }
                else:
                    return HealthStatus.UNHEALTHY, {
                        'reason': f'API error: {response.status}',
                        'status_code': response.status
                    }
                    
        except asyncio.TimeoutError:
            return HealthStatus.DEGRADED, {'reason': 'Request timeout'}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {'reason': str(e)}
    
    async def _check_anthropic_health(self, session: aiohttp.ClientSession,
                                     config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Check Anthropic API health"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return HealthStatus.CRITICAL, {'reason': 'No API key configured'}
        
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        
        try:
            start_time = time.time()
            async with session.post(
                f"{config['base_url']}/v1/messages",
                headers=headers,
                json={
                    'model': config['models']['primary'],
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'max_tokens': 1
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                latency = time.time() - start_time
                
                if self.metrics.get('response_time'):
                    self.metrics['response_time'].labels(
                        provider='anthropic',
                        model=config['models']['primary']
                    ).observe(latency)
                
                if response.status == 200:
                    return HealthStatus.HEALTHY, {
                        'latency': latency,
                        'models_available': list(config['models'].values())
                    }
                elif response.status == 429:
                    return HealthStatus.DEGRADED, {
                        'reason': 'Rate limited',
                        'status_code': response.status
                    }
                else:
                    return HealthStatus.UNHEALTHY, {
                        'reason': f'API error: {response.status}',
                        'status_code': response.status
                    }
                    
        except asyncio.TimeoutError:
            return HealthStatus.DEGRADED, {'reason': 'Request timeout'}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {'reason': str(e)}
    
    async def _check_lmstudio_health(self, session: aiohttp.ClientSession,
                                   config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Check LM Studio health"""
        try:
            # First check if the server is running
            async with session.get(
                f"{config['base_url']}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    return HealthStatus.UNHEALTHY, {
                        'reason': 'LM Studio server not responding'
                    }
                
                models_data = await response.json()
                available_models = [m['id'] for m in models_data.get('data', [])]
                
                # Check if configured model is loaded
                if config['model'] not in available_models:
                    return HealthStatus.DEGRADED, {
                        'reason': f"Configured model {config['model']} not loaded",
                        'available_models': available_models
                    }
                
                # Test inference
                start_time = time.time()
                async with session.post(
                    f"{config['base_url']}/chat/completions",
                    json={
                        'model': config['model'],
                        'messages': [{'role': 'user', 'content': 'test'}],
                        'max_tokens': 1
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as test_response:
                    latency = time.time() - start_time
                    
                    if self.metrics.get('response_time'):
                        self.metrics['response_time'].labels(
                            provider='lmstudio',
                            model=config['model']
                        ).observe(latency)
                    
                    if test_response.status == 200:
                        return HealthStatus.HEALTHY, {
                            'latency': latency,
                            'model': config['model'],
                            'available_models': available_models
                        }
                    else:
                        return HealthStatus.DEGRADED, {
                            'reason': f'Inference error: {test_response.status}'
                        }
                        
        except aiohttp.ClientError:
            return HealthStatus.UNHEALTHY, {'reason': 'LM Studio offline'}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {'reason': str(e)}
    
    async def check_system_resources(self) -> Tuple[str, Dict[str, Any]]:
        """Check system resource usage"""
        if not psutil:
            return HealthStatus.HEALTHY, {'reason': 'psutil not available'}
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update metrics
            if self.metrics.get('cpu_usage'):
                self.metrics['cpu_usage'].set(cpu_percent)
            if self.metrics.get('memory_usage'):
                self.metrics['memory_usage'].set(memory.used)
            
            # Determine health based on thresholds
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent}% used")
            
            if issues:
                return HealthStatus.DEGRADED, {
                    'issues': issues,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                }
            
            return HealthStatus.HEALTHY, {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthStatus.UNHEALTHY, {'reason': str(e)}
    
    async def check_circuit_breakers(self) -> Tuple[str, Dict[str, Any]]:
        """Check circuit breaker states"""
        if not self.config['circuit_breaker']['enabled']:
            return HealthStatus.HEALTHY, {'reason': 'Circuit breakers disabled'}
        
        # This would integrate with the actual circuit breaker implementation
        # For now, return a placeholder
        return HealthStatus.HEALTHY, {
            'circuit_breakers': {
                'openai': 'CLOSED',
                'anthropic': 'CLOSED',
                'lmstudio': 'CLOSED'
            }
        }
    
    async def perform_full_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        logger.info("Starting full health check...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': HealthStatus.HEALTHY,
            'health_score': 100,
            'components': {}
        }
        
        # Check all providers
        provider_scores = []
        for provider in ['openai', 'anthropic', 'lmstudio']:
            status, details = await self.check_provider_health(provider)
            results['components'][provider] = {
                'status': status,
                'details': details
            }
            
            # Calculate score
            if status == HealthStatus.HEALTHY:
                provider_scores.append(100)
            elif status == HealthStatus.DEGRADED:
                provider_scores.append(70)
            elif status == HealthStatus.UNHEALTHY:
                provider_scores.append(30)
            else:  # CRITICAL
                provider_scores.append(0)
        
        # Check system resources
        sys_status, sys_details = await self.check_system_resources()
        results['components']['system'] = {
            'status': sys_status,
            'details': sys_details
        }
        
        # Check circuit breakers
        cb_status, cb_details = await self.check_circuit_breakers()
        results['components']['circuit_breakers'] = {
            'status': cb_status,
            'details': cb_details
        }
        
        # Calculate overall health score
        if provider_scores:
            results['health_score'] = sum(provider_scores) / len(provider_scores)
        
        # Determine overall status
        if results['health_score'] >= 80:
            results['overall_status'] = HealthStatus.HEALTHY
        elif results['health_score'] >= 60:
            results['overall_status'] = HealthStatus.DEGRADED
        elif results['health_score'] >= 30:
            results['overall_status'] = HealthStatus.UNHEALTHY
        else:
            results['overall_status'] = HealthStatus.CRITICAL
        
        # Update metrics
        if self.metrics.get('health_score'):
            self.metrics['health_score'].set(results['health_score'])
        
        # Store in history
        self.health_history.append(results)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        logger.info(f"Health check complete: {results['overall_status']} (score: {results['health_score']:.1f})")
        
        return results
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous health monitoring"""
        self.monitoring = True
        logger.info(f"Starting health monitoring with {interval}s interval")
        
        # Start metrics server if enabled
        if self.config['monitoring']['metrics_enabled'] and prometheus_client_available:
            host, port = self.config['monitoring']['metrics_bind'].split(':')
            start_http_server(int(port), addr=host)
            logger.info(f"Metrics server started on {host}:{port}")
        
        while self.monitoring:
            try:
                results = await self.perform_full_health_check()
                
                # Check for critical issues
                if results['overall_status'] == HealthStatus.CRITICAL:
                    self.alerts.put({
                        'level': 'CRITICAL',
                        'message': 'System health critical',
                        'details': results
                    })
                elif results['overall_status'] == HealthStatus.UNHEALTHY:
                    self.alerts.put({
                        'level': 'WARNING',
                        'message': 'System health degraded',
                        'details': results
                    })
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        logger.info("Health monitoring stopped")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get current health report"""
        if not self.health_history:
            return {'status': 'No health data available'}
        
        latest = self.health_history[-1]
        
        # Calculate trends if we have enough history
        if len(self.health_history) >= 5:
            recent_scores = [h['health_score'] for h in self.health_history[-5:]]
            trend = 'improving' if recent_scores[-1] > recent_scores[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'current_status': latest['overall_status'],
            'health_score': latest['health_score'],
            'trend': trend,
            'last_check': latest['timestamp'],
            'components': latest['components'],
            'alerts': list(self.alerts.queue) if not self.alerts.empty() else []
        }

async def main():
    """Main entry point for health monitoring"""
    monitor = MCPHealthMonitor()
    
    # Command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'check':
            # Perform single health check
            results = await monitor.perform_full_health_check()
            print(json.dumps(results, indent=2))
            
        elif command == 'monitor':
            # Start continuous monitoring
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            try:
                await monitor.start_monitoring(interval)
            except KeyboardInterrupt:
                monitor.stop_monitoring()
                print("\nMonitoring stopped")
                
        elif command == 'report':
            # Get health report
            report = monitor.get_health_report()
            print(json.dumps(report, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: mcp_health_monitor.py [check|monitor [interval]|report]")
    else:
        # Default: perform single check
        results = await monitor.perform_full_health_check()
        
        print("\n" + "="*60)
        print("🏥 MCP SERVER HEALTH CHECK")
        print("="*60 + "\n")
        
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Health Score: {results['health_score']:.1f}/100")
        print(f"Timestamp: {results['timestamp']}\n")
        
        print("Component Status:")
        for component, data in results['components'].items():
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.CRITICAL: "🔴"
            }.get(data['status'], "❓")
            
            print(f"  {status_emoji} {component.upper()}: {data['status']}")
            if 'reason' in data.get('details', {}):
                print(f"     → {data['details']['reason']}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Health monitor failed: {e}")
        traceback.print_exc()
        sys.exit(1)