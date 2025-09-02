#!/usr/bin/env python3
"""
Test suite for reliability improvements in LM Studio MCP Server
Tests timeout handling, circuit breakers, and enhanced HTTP client functionality
"""

import asyncio
import json
import os
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
sys.path.append('.')

# Import server components
from server import EnhancedLMStudioMCPServer, RobustHTTPClient, CircuitBreaker, _circuit_breakers

class TestReliabilityImprovements:
    """Test suite for reliability improvements"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset circuit breakers
        for cb in _circuit_breakers.values():
            cb.failure_count = 0
            cb.state = "CLOSED"
            cb.last_failure_time = 0
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Initially closed
        assert cb.can_execute() == True
        assert cb.state == "CLOSED"
        
        # Record failures
        for i in range(2):
            cb.record_failure()
            assert cb.can_execute() == True  # Still closed
        
        # Third failure should open circuit
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() == False
        
        # Wait for recovery timeout
        time.sleep(1.1)
        assert cb.can_execute() == True  # Should be half-open
        assert cb.state == "HALF_OPEN"
        
        # Success should close circuit
        cb.record_success()
        assert cb.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_enhanced_http_client_timeouts(self):
        """Test adaptive timeout functionality"""
        client = RobustHTTPClient()
        
        # Test timeout calculation
        simple_timeout = client.get_timeout("simple")
        complex_timeout = client.get_timeout("complex")
        
        assert simple_timeout[0] == 10  # connect timeout
        assert simple_timeout[1] == 60  # simple read timeout
        assert complex_timeout[1] == 180  # complex read timeout
    
    @pytest.mark.asyncio
    async def test_lmstudio_request_with_circuit_breaker(self):
        """Test LM Studio request with circuit breaker integration"""
        server = EnhancedLMStudioMCPServer()
        
        # Mock the HTTP client to simulate failures
        with patch.object(server.http_client, 'post_async', side_effect=Exception("Connection failed")):
            # Make multiple requests to trigger circuit breaker
            for i in range(4):  # Exceed failure threshold
                result = await server._lmstudio_request_with_retry("test prompt", retries=0)
                assert "Error:" in result
            
            # Circuit breaker should now be open
            cb = _circuit_breakers["lmstudio"]
            assert cb.state == "OPEN"
            
            # Next request should be blocked by circuit breaker
            result = await server._lmstudio_request_with_retry("test prompt", retries=0)
            assert "circuit breaker is OPEN" in result
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_selection(self):
        """Test adaptive timeout based on operation complexity"""
        server = EnhancedLMStudioMCPServer()
        
        # Mock successful response
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        
        with patch.object(server.http_client, 'post_async', return_value=mock_response) as mock_post:
            # Simple operation
            await server._lmstudio_request_with_retry("simple prompt")
            mock_post.assert_called_with(
                f"{server.base_url}/v1/chat/completions",
                json_data={
                    "model": server.model_name,
                    "messages": [{"role": "user", "content": "simple prompt"}],
                    "temperature": 0.35,
                    "max_tokens": 4000
                },
                headers={"Content-Type": "application/json"},
                operation_type="simple"
            )
            
            # Complex operation
            complex_prompt = "analyze" + " " * 2000  # Long prompt with analyze keyword
            await server._lmstudio_request_with_retry(complex_prompt)
            mock_post.assert_called_with(
                f"{server.base_url}/v1/chat/completions",
                json_data={
                    "model": server.model_name,
                    "messages": [{"role": "user", "content": complex_prompt}],
                    "temperature": 0.35,
                    "max_tokens": 4000
                },
                headers={"Content-Type": "application/json"},
                operation_type="complex"
            )
    
    @pytest.mark.asyncio
    async def test_async_rate_limiting(self):
        """Test async rate limiting doesn't block event loop"""
        server = EnhancedLMStudioMCPServer()
        server._router_min_interval = 0.1  # 100ms interval
        
        start_time = time.time()
        
        # Make two rapid calls
        await server._router_wait("lmstudio")
        await server._router_wait("lmstudio")
        
        elapsed = time.time() - start_time
        assert elapsed >= 0.1  # Should have waited at least 100ms
        assert elapsed < 0.2   # But not too long (async should be efficient)
    
    def test_configuration_improvements(self):
        """Test improved configuration settings"""
        # Test that new environment variables are properly set
        expected_configs = {
            "ROUTER_RATE_LIMIT_TPS": "12",
            "SMART_PLAN_IMMEDIATE_TIMEOUT_SEC": "150", 
            "ROUTER_BG_TIMEOUT_SEC": "300",
            "HTTP_CONNECT_TIMEOUT": "10",
            "HTTP_READ_TIMEOUT_SIMPLE": "60",
            "HTTP_READ_TIMEOUT_COMPLEX": "180",
            "HTTP_MAX_RETRIES": "3",
            "HTTP_BACKOFF_FACTOR": "1.5"
        }
        
        # Read mcp.json to verify settings
        with open("recommendations/mcp.json", "r") as f:
            config = json.load(f)
        
        env_vars = config["mcpServers"]["lmstudio-mcp"]["env"]
        
        for key, expected_value in expected_configs.items():
            assert key in env_vars, f"Missing config: {key}"
            assert env_vars[key] == expected_value, f"Wrong value for {key}: got {env_vars[key]}, expected {expected_value}"

def run_reliability_tests():
    """Run all reliability tests"""
    print("🧪 Running Reliability Improvement Tests...")
    
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    if exit_code == 0:
        print("✅ All reliability tests passed!")
        return True
    else:
        print("❌ Some reliability tests failed!")
        return False

if __name__ == "__main__":
    # Set test environment variables
    os.environ.update({
        "HTTP_CONNECT_TIMEOUT": "10",
        "HTTP_READ_TIMEOUT_SIMPLE": "60", 
        "HTTP_READ_TIMEOUT_COMPLEX": "180",
        "HTTP_MAX_RETRIES": "3",
        "HTTP_BACKOFF_FACTOR": "1.5",
        "ROUTER_RATE_LIMIT_TPS": "12"
    })
    
    success = run_reliability_tests()
    sys.exit(0 if success else 1)
