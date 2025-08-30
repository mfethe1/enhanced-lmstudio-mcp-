#!/usr/bin/env python3
"""
Router Testing and Monitoring Suite
Test the intelligent routing system and monitor its performance
"""

import asyncio
import json
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import os


class RouterTestSuite:
    """Test suite for router agent validation"""
    
    def __init__(self, server):
        self.server = server
        self.results = []
        
    async def run_test_battery(self):
        """Run comprehensive test battery"""
        print("🚀 Starting Router Test Battery\n")
        
        # Test 1: Role-based routing
        await self.test_role_routing()
        
        # Test 2: Complexity-based routing
        await self.test_complexity_routing()
        
        # Test 3: Fallback behavior
        await self.test_fallback_behavior()
        
        # Test 4: Cache effectiveness
        await self.test_cache_effectiveness()
        
        # Test 5: Performance under load
        await self.test_performance_load()
        
        # Generate report
        self.generate_report()
        
    async def test_role_routing(self):
        """Test that roles route to appropriate backends"""
        print("📋 Test 1: Role-based Routing")
        
        test_cases = [
            ("Security Reviewer", "Review this code for vulnerabilities", "security"),
            ("Coder", "Write a Python function to sort a list", "implementation"),
            ("Researcher", "Analyze the implications of quantum computing", "research"),
            ("Performance Analyzer", "Find bottlenecks in this algorithm", "optimization"),
        ]
        
        for role, task, expected_intent in test_cases:
            start = time.time()
            
            result = await self.server.route_chat(
                task,
                role=role,
                intent=expected_intent
            )
            
            elapsed = time.time() - start
            
            # Get the routing decision from logs
            decision = self._extract_last_routing_decision()
            
            self.results.append({
                "test": "role_routing",
                "role": role,
                "backend": decision.get("backend", "unknown"),
                "confidence": decision.get("confidence", 0),
                "latency": elapsed,
                "success": len(result) > 0 and not result.startswith("Error")
            })
            
            print(f"  ✓ {role} → {decision.get('backend', 'unknown')} "
                  f"(confidence: {decision.get('confidence', 0):.0%}, "
                  f"latency: {elapsed:.2f}s)")
                  
    async def test_complexity_routing(self):
        """Test complexity-based routing"""
        print("\n📊 Test 2: Complexity-based Routing")
        
        test_cases = [
            ("Explain what a variable is", "low"),
            ("Design a microservices architecture", "high"),
            ("Write a binary search function", "medium"),
            ("Prove the halting problem is undecidable", "high"),
        ]
        
        for task, expected_complexity in test_cases:
            result = await self.server.router_test({
                "task": task,
                "complexity": "auto"  # Let router detect
            })
            
            # Parse result
            detected_complexity = self._extract_field(result, "Complexity:")
            backend = self._extract_field(result, "Backend:")
            
            self.results.append({
                "test": "complexity_routing",
                "task": task[:50],
                "expected": expected_complexity,
                "detected": detected_complexity,
                "backend": backend,
                "match": detected_complexity == expected_complexity
            })
            
            status = "✓" if detected_complexity == expected_complexity else "✗"
            print(f"  {status} '{task[:50]}...' → {detected_complexity} "
                  f"(expected: {expected_complexity}) → {backend}")
                  
    async def test_fallback_behavior(self):
        """Test fallback chain behavior"""
        print("\n🔄 Test 3: Fallback Behavior")
        
        # Simulate primary backend failure by using invalid API key
        original_key = os.getenv("ANTHROPIC_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = "invalid_key_to_force_failure"
        
        try:
            result = await self.server.route_chat(
                "Analyze this complex philosophical argument about consciousness",
                role="Philosopher",
                complexity="high"
            )
            
            success = len(result) > 0 and not result.startswith("Error")
            
            self.results.append({
                "test": "fallback",
                "scenario": "primary_failure",
                "success": success,
                "result_preview": result[:100] if success else result
            })
            
            print(f"  {'✓' if success else '✗'} Primary failure handled: {success}")
            
        finally:
            # Restore key
            os.environ["ANTHROPIC_API_KEY"] = original_key
            
    async def test_cache_effectiveness(self):
        """Test decision caching"""
        print("\n💾 Test 4: Cache Effectiveness")
        
        task = "Write a function to calculate fibonacci numbers"
        
        # First call (cache miss)
        start1 = time.time()
        result1 = await self.server.route_chat(task)
        time1 = time.time() - start1
        
        # Second call (should be cache hit)
        start2 = time.time()
        result2 = await self.server.route_chat(task)
        time2 = time.time() - start2
        
        cache_speedup = time1 / time2 if time2 > 0 else 0
        
        self.results.append({
            "test": "cache",
            "first_call_time": time1,
            "second_call_time": time2,
            "speedup": cache_speedup,
            "cache_worked": time2 < time1 * 0.5  # At least 2x speedup
        })
        
        print(f"  ✓ First call: {time1:.2f}s, Second call: {time2:.2f}s")
        print(f"  ✓ Cache speedup: {cache_speedup:.1f}x")
        
    async def test_performance_load(self):
        """Test performance under concurrent load"""
        print("\n🏃 Test 5: Performance Under Load")
        
        tasks = [
            ("Write hello world in Python", "Coder", "low"),
            ("Explain quantum entanglement", "Physicist", "high"),
            ("Review this security patch", "Security Reviewer", "high"),
            ("Optimize database query", "DBA", "medium"),
            ("Design REST API", "Architect", "medium"),
        ] * 2  # 10 concurrent requests
        
        start = time.time()
        
        # Run concurrently
        results = await asyncio.gather(*[
            self.server.route_chat(task, role=role, complexity=complexity)
            for task, role, complexity in tasks
        ], return_exceptions=True)
        
        total_time = time.time() - start
        successful = sum(1 for r in results 
                        if not isinstance(r, Exception) and 
                        not str(r).startswith("Error"))
        
        self.results.append({
            "test": "load",
            "total_requests": len(tasks),
            "successful": successful,
            "total_time": total_time,
            "avg_time": total_time / len(tasks),
            "success_rate": successful / len(tasks)
        })
        
        print(f"  ✓ Processed {len(tasks)} requests in {total_time:.2f}s")
        print(f"  ✓ Success rate: {successful}/{len(tasks)} "
              f"({successful/len(tasks):.0%})")
        print(f"  ✓ Average time: {total_time/len(tasks):.2f}s per request")
        
    def _extract_last_routing_decision(self) -> Dict[str, Any]:
        """Extract last routing decision from storage"""
        try:
            decisions = self.server.storage.retrieve_memory(
                category="router_decisions",
                limit=1
            )
            if decisions:
                return json.loads(decisions[0]["value"])
        except:
            pass
        return {}
        
    def _extract_field(self, text: str, field: str) -> str:
        """Extract field value from text output"""
        lines = text.split('\n')
        for line in lines:
            if field in line:
                return line.split(field)[-1].strip()
        return "unknown"
        
    def generate_report(self):
        """Generate test report with visualizations"""
        print("\n📈 Test Report Summary\n")
        
        # Summary stats
        total_tests = len(self.results)
        successful = sum(1 for r in self.results 
                        if r.get("success") or r.get("match") or r.get("cache_worked"))
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Success Rate: {successful/total_tests:.0%}\n")
        
        # Group results by test
        by_test = defaultdict(list)
        for result in self.results:
            by_test[result["test"]].append(result)
            
        # Per-test summaries
        for test_name, results in by_test.items():
            print(f"### {test_name.replace('_', ' ').title()}")
            
            if test_name == "role_routing":
                backends = defaultdict(int)
                for r in results:
                    backends[r.get("backend", "unknown")] += 1
                for backend, count in backends.items():
                    print(f"  - {backend}: {count} requests")
                    
            elif test_name == "complexity_routing":
                matches = sum(1 for r in results if r.get("match"))
                print(f"  - Accuracy: {matches}/{len(results)} ({matches/len(results):.0%})")
                
            elif test_name == "cache":
                r = results[0]
                print(f"  - Speedup: {r['speedup']:.1f}x")
                print(f"  - Cache effective: {'Yes' if r['cache_worked'] else 'No'}")
                
            elif test_name == "load":
                r = results[0]
                print(f"  - Success rate: {r['success_rate']:.0%}")
                print(f"  - Avg latency: {r['avg_time']:.2f}s")
                
            print()


class RouterMonitor:
    """Real-time monitoring for router performance"""
    
    def __init__(self, server):
        self.server = server
        
    async def monitor_live(self, duration_seconds: int = 60):
        """Monitor router performance in real-time"""
        print(f"📊 Monitoring router for {duration_seconds} seconds...\n")
        
        start_time = time.time()
        snapshots = []
        
        while time.time() - start_time < duration_seconds:
            # Get current analytics
            analytics = await self.server._router_agent.get_router_analytics(hours=1)
            
            snapshot = {
                "timestamp": time.time(),
                "total_decisions": analytics["total_decisions"],
                "avg_confidence": analytics["avg_confidence"],
                "cache_size": analytics["cache_size"],
                "backend_distribution": analytics["backend_distribution"],
                "backend_performance": analytics["backend_performance"]
            }
            snapshots.append(snapshot)
            
            # Display current stats
            self._display_snapshot(snapshot)
            
            # Wait before next snapshot
            await asyncio.sleep(5)
            
        # Generate summary
        self._generate_monitoring_summary(snapshots)
        
    def _display_snapshot(self, snapshot: Dict[str, Any]):
        """Display current snapshot"""
        print(f"\r⏰ {time.strftime('%H:%M:%S')} | ", end="")
        print(f"Decisions: {snapshot['total_decisions']} | ", end="")
        print(f"Confidence: {snapshot['avg_confidence']:.0%} | ", end="")
        print(f"Cache: {snapshot['cache_size']} | ", end="")
        
        # Backend distribution
        backends = []
        for backend, count in snapshot['backend_distribution'].items():
            backends.append(f"{backend}: {count}")
        print(" | ".join(backends), end="")
        
    def _generate_monitoring_summary(self, snapshots: List[Dict[str, Any]]):
        """Generate monitoring summary"""
        print("\n\n📊 Monitoring Summary\n")
        
        if not snapshots:
            print("No data collected")
            return
            
        # Decision rate
        duration = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]
        total_decisions = snapshots[-1]["total_decisions"] - snapshots[0]["total_decisions"]
        decision_rate = total_decisions / duration * 60 if duration > 0 else 0
        
        print(f"Decision Rate: {decision_rate:.1f} per minute")
        
        # Average confidence trend
        confidences = [s["avg_confidence"] for s in snapshots if s["avg_confidence"] > 0]
        if confidences:
            print(f"Confidence Range: {min(confidences):.0%} - {max(confidences):.0%}")
            
        # Backend usage
        print("\nBackend Usage:")
        final_dist = snapshots[-1]["backend_distribution"]
        total = sum(final_dist.values())
        for backend, count in final_dist.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  - {backend}: {count} ({pct:.1f}%)")


# Example usage
async def test_router_system():
    """Complete test of the router system"""
    from enhanced_lm_studio_mcp_server import EnhancedLMStudioMCPServer
    
    print("🚀 Initializing Enhanced LM Studio MCP Server with Router...\n")
    server = EnhancedLMStudioMCPServer()
    
    # Wait for router initialization
    await asyncio.sleep(2)
    
    if not server._router_initialized:
        print("❌ Router failed to initialize")
        return
        
    print("✅ Router initialized successfully\n")
    
    # Run test suite
    test_suite = RouterTestSuite(server)
    await test_suite.run_test_battery()
    
    # Optional: Run monitoring
    # monitor = RouterMonitor(server)
    # await monitor.monitor_live(duration_seconds=60)


if __name__ == "__main__":
    asyncio.run(test_router_system())