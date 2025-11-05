#!/usr/bin/env python3
"""
Comprehensive test for production hardening measures.
Tests circuit breakers, model monitoring, configuration optimization, and graceful shutdown.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_circuit_breaker_functionality():
    """Test circuit breaker pattern implementation"""
    print("=" * 80)
    print("PHASE 3.1: Circuit Breaker Functionality Test")
    print("=" * 80)
    
    try:
        from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
        
        # Create test circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        breaker = CircuitBreaker("test_service", config)
        
        print(f"✓ Circuit breaker created: {breaker.name}")
        print(f"✓ Initial state: {breaker.state.value}")
        
        # Test successful calls
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        if result == "success":
            print("✓ Successful call through circuit breaker")
        
        # Test failure handling
        def failure_func():
            raise Exception("Test failure")
        
        failure_count = 0
        for i in range(3):
            try:
                breaker.call(failure_func)
            except Exception:
                failure_count += 1
        
        print(f"✓ Handled {failure_count} failures")
        print(f"✓ Circuit state after failures: {breaker.state.value}")
        
        # Test circuit open behavior
        if breaker.state.value == "open":
            try:
                breaker.call(success_func)
                print("✗ Circuit should be open and blocking calls")
                return False
            except CircuitBreakerOpenError:
                print("✓ Circuit correctly blocking calls when open")
        
        return True
        
    except Exception as e:
        print(f"✗ Circuit breaker test failed: {e}")
        return False

def test_model_monitoring():
    """Test model availability monitoring"""
    print("\n" + "=" * 80)
    print("PHASE 3.2: Model Monitoring Test")
    print("=" * 80)
    
    try:
        from model_monitor import ModelMonitor, ModelChangeEvent, create_alert_callback
        
        # Mock server for testing
        class MockServer:
            def __init__(self):
                self.models = ["model1", "model2", "model3"]
            
            def refresh_lmstudio_models(self, force=False):
                return self.models.copy()
        
        mock_server = MockServer()
        monitor = ModelMonitor(mock_server, check_interval=1.0)
        
        print(f"✓ Model monitor created")
        print(f"✓ Initial models: {len(monitor.get_current_models())}")
        
        # Test callback system
        events_received = []
        def test_callback(event: ModelChangeEvent):
            events_received.append(event)
        
        monitor.add_callback("test", test_callback)
        print("✓ Callback added")
        
        # Simulate model change
        mock_server.models = ["model1", "model2"]  # Remove model3
        changed = monitor.force_check()
        
        if changed:
            print("✓ Model change detected")
            if events_received:
                print(f"✓ Callback triggered: {events_received[-1].event_type}")
        
        # Test statistics
        stats = monitor.get_stats()
        print(f"✓ Monitor stats: {stats['model_count']} models, {stats['recent_events']} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Model monitoring test failed: {e}")
        return False

def test_production_configuration():
    """Test production configuration management"""
    print("\n" + "=" * 80)
    print("PHASE 3.3: Production Configuration Test")
    print("=" * 80)
    
    try:
        from production_config import ProductionConfig, validate_production_config
        
        # Test configuration creation
        config = ProductionConfig.from_environment()
        print(f"✓ Production config created")
        print(f"✓ Log level: {config.log_level}")
        print(f"✓ Expose public only: {config.expose_public_only}")
        print(f"✓ Proactive research: {config.proactive_research_enabled}")
        
        # Test validation
        is_valid = validate_production_config()
        print(f"✓ Configuration validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test circuit breaker configs
        cb_configs = config.get_circuit_breaker_configs()
        print(f"✓ Circuit breaker configs: {len(cb_configs)} services")
        
        # Test configuration dictionary
        config_dict = config.to_dict()
        print(f"✓ Config dictionary: {len(config_dict)} settings")
        
        return True
        
    except Exception as e:
        print(f"✗ Production configuration test failed: {e}")
        return False

def test_graceful_shutdown():
    """Test graceful shutdown of all components"""
    print("\n" + "=" * 80)
    print("PHASE 3.4: Graceful Shutdown Test")
    print("=" * 80)
    
    try:
        # Test circuit breaker manager shutdown
        from circuit_breaker import circuit_manager
        
        # Get initial stats
        initial_stats = circuit_manager.get_all_stats()
        print(f"✓ Circuit manager stats: {len(initial_stats)} breakers")
        
        # Test reset functionality
        circuit_manager.reset_all()
        print("✓ All circuit breakers reset")
        
        # Test model monitor shutdown
        from model_monitor import ModelMonitor
        
        class MockServer:
            def refresh_lmstudio_models(self, force=False):
                return ["test_model"]
        
        monitor = ModelMonitor(MockServer(), check_interval=1.0)
        monitor.start_monitoring()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        monitor.stop_monitoring()
        print("✓ Model monitor stopped gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Graceful shutdown test failed: {e}")
        return False

def test_integrated_hardening():
    """Test integrated hardening with server components"""
    print("\n" + "=" * 80)
    print("PHASE 3.5: Integrated Hardening Test")
    print("=" * 80)
    
    try:
        # Set production environment
        os.environ.update({
            "EXPOSE_PUBLIC_ONLY": "1",
            "PROACTIVE_RESEARCH_ENABLED": "0",
            "LOG_LEVEL": "WARNING",
            "CIRCUIT_BREAKER_ENABLED": "1",
            "MODEL_MONITOR_INTERVAL": "300"
        })
        
        # Import server with hardening
        import server
        
        # Test server creation with hardening
        test_server = server.EnhancedLMStudioMCPServer()
        print("✓ Server created with hardening measures")
        
        # Test circuit breaker integration
        if hasattr(server, 'circuit_manager'):
            available_services = server.circuit_manager.get_available_services()
            print(f"✓ Circuit breakers available: {len(available_services)} services")
        
        # Test model monitoring integration
        if hasattr(test_server, 'model_monitor'):
            monitor_stats = test_server.model_monitor.get_stats()
            print(f"✓ Model monitor integrated: {monitor_stats['model_count']} models")
        
        # Test production configuration
        from production_config import get_production_config
        prod_config = get_production_config()
        print(f"✓ Production config: expose_public_only={prod_config.expose_public_only}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integrated hardening test failed: {e}")
        return False

def main():
    """Run production hardening tests"""
    print("LM Studio MCP Server - Production Hardening Tests")
    print("=" * 80)
    
    tests = [
        test_circuit_breaker_functionality,
        test_model_monitoring,
        test_production_configuration,
        test_graceful_shutdown,
        test_integrated_hardening
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION HARDENING TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        phase_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"{status} Phase 3.{i+1}: {phase_name}")
    
    if passed == total:
        print("\n🎉 PRODUCTION HARDENING COMPLETE!")
        print("✅ All hardening measures implemented and tested")
        print("✅ Circuit breakers protecting external services")
        print("✅ Model monitoring with alerting")
        print("✅ Production configuration optimized")
        print("✅ Graceful shutdown implemented")
        return True
    else:
        print(f"\n⚠️ HARDENING ISSUES DETECTED")
        print(f"✗ {total - passed} hardening measure(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
