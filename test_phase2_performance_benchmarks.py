"""
Performance Benchmarks for Phase 2 Components

Validates that timeout values are appropriate and performance targets are met.

Author: Jarvis MCP Team
Date: 2025-01-16
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

# Add parent directory to path
sys.path.insert(0, ".")

from server import EnhancedLMStudioMCPServer
from handlers import agent_teams, swarm


class Phase2PerformanceBenchmarks:
    """Performance benchmarks for Phase 2 components"""
    
    def __init__(self):
        self.server = EnhancedLMStudioMCPServer()
        self.results = {
            "ephemeral_agents": {},
            "file_locking": {},
            "workflows": {},
            "swarm": {}
        }
        self.test_base_dir = Path("C:/Users/mfeth/.mcp-servers/lmstudio-mcp")
    
    def print_section(self, title):
        """Print section header"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def print_metric(self, name, value, unit, target=None, status=None):
        """Print performance metric"""
        if status is None and target is not None:
            status = "[OK]" if value <= target else "[SLOW]"
        elif status is None:
            status = "[OK]"
        
        print(f"{status} {name}: {value:.2f}{unit}", end="")
        if target:
            print(f" (target: <{target}{unit})")
        else:
            print()
    
    def benchmark_ephemeral_agents(self, iterations=10):
        """Benchmark ephemeral agent operations"""
        self.print_section("BENCHMARK 1: EPHEMERAL AGENTS")
        
        creation_times = []
        release_times = []
        
        print(f"\nRunning {iterations} iterations...")
        
        for i in range(iterations):
            # Benchmark creation
            start = time.time()
            args = {
                "role": "planner",
                "task_description": f"Benchmark task {i}",
                "priority": 1
            }
            result_str = agent_teams.handle_request_ephemeral_agent(args, self.server)
            result = json.loads(result_str)
            creation_time = (time.time() - start) * 1000  # Convert to ms
            creation_times.append(creation_time)
            
            # Benchmark release (if agent was created)
            agent_id = result.get("id")
            if agent_id and not agent_id.startswith("req-"):
                start = time.time()
                release_args = {"agent_id": agent_id}
                agent_teams.handle_release_ephemeral_agent(release_args, self.server)
                release_time = (time.time() - start) * 1000
                release_times.append(release_time)
        
        # Calculate statistics
        print("\n--- Creation Performance ---")
        self.print_metric("Mean", mean(creation_times), "ms", target=100)
        self.print_metric("Median", median(creation_times), "ms")
        self.print_metric("Std Dev", stdev(creation_times) if len(creation_times) > 1 else 0, "ms")
        self.print_metric("Min", min(creation_times), "ms")
        self.print_metric("Max", max(creation_times), "ms")
        
        if release_times:
            print("\n--- Release Performance ---")
            self.print_metric("Mean", mean(release_times), "ms", target=50)
            self.print_metric("Median", median(release_times), "ms")
        
        self.results["ephemeral_agents"] = {
            "creation_mean_ms": mean(creation_times),
            "creation_median_ms": median(creation_times),
            "release_mean_ms": mean(release_times) if release_times else 0
        }
    
    def benchmark_file_locking(self, iterations=10):
        """Benchmark file locking operations"""
        self.print_section("BENCHMARK 2: FILE LOCKING")
        
        test_file = self.test_base_dir / "benchmark_lock_file.txt"
        test_file.touch(exist_ok=True)
        
        acquire_times = []
        release_times = []
        
        print(f"\nRunning {iterations} iterations...")
        
        for i in range(iterations):
            # Benchmark acquire
            start = time.time()
            args = {
                "file_path": str(test_file),
                "owner_id": f"benchmark-{i}",
                "timeout_seconds": 5
            }
            result_str = agent_teams.handle_acquire_file_lock(args, self.server)
            result = json.loads(result_str)
            acquire_time = (time.time() - start) * 1000
            acquire_times.append(acquire_time)
            
            # Benchmark release
            if result.get("status") == "success":
                start = time.time()
                release_args = {
                    "file_path": str(test_file),
                    "owner_id": f"benchmark-{i}"
                }
                agent_teams.handle_release_file_lock(release_args, self.server)
                release_time = (time.time() - start) * 1000
                release_times.append(release_time)
        
        # Calculate statistics
        print("\n--- Acquire Performance ---")
        self.print_metric("Mean", mean(acquire_times), "ms", target=10)
        self.print_metric("Median", median(acquire_times), "ms")
        self.print_metric("Std Dev", stdev(acquire_times) if len(acquire_times) > 1 else 0, "ms")
        self.print_metric("Min", min(acquire_times), "ms")
        self.print_metric("Max", max(acquire_times), "ms")
        
        print("\n--- Release Performance ---")
        self.print_metric("Mean", mean(release_times), "ms", target=5)
        self.print_metric("Median", median(release_times), "ms")
        
        self.results["file_locking"] = {
            "acquire_mean_ms": mean(acquire_times),
            "acquire_median_ms": median(acquire_times),
            "release_mean_ms": mean(release_times)
        }
    
    def benchmark_workflows(self, iterations=5):
        """Benchmark workflow execution"""
        self.print_section("BENCHMARK 3: WORKFLOWS")
        
        parallel_times = []
        
        print(f"\nRunning {iterations} iterations...")
        
        for i in range(iterations):
            # Benchmark parallel workflow
            start = time.time()
            tasks = [
                {"task_id": f"task-{j}", "description": f"Task {j}"}
                for j in range(5)
            ]
            args = {
                "tasks": tasks,
                "error_strategy": "CONTINUE",
                "aggregation_strategy": "ALL"
            }
            agent_teams.handle_execute_parallel_workflow(args, self.server)
            parallel_time = (time.time() - start) * 1000
            parallel_times.append(parallel_time)
        
        # Calculate statistics
        print("\n--- Parallel Workflow Performance ---")
        self.print_metric("Mean", mean(parallel_times), "ms", target=5000)
        self.print_metric("Median", median(parallel_times), "ms")
        self.print_metric("Std Dev", stdev(parallel_times) if len(parallel_times) > 1 else 0, "ms")
        self.print_metric("Min", min(parallel_times), "ms")
        self.print_metric("Max", max(parallel_times), "ms")
        
        self.results["workflows"] = {
            "parallel_mean_ms": mean(parallel_times),
            "parallel_median_ms": median(parallel_times)
        }
    
    def benchmark_swarm(self, iterations=5):
        """Benchmark swarm operations"""
        self.print_section("BENCHMARK 4: SWARM PATTERN")
        
        create_times = []
        execute_times = []
        
        print(f"\nRunning {iterations} iterations...")
        
        for i in range(iterations):
            # Benchmark swarm creation
            start = time.time()
            agents = [
                {"agent_id": f"planner-{i}", "specialization": "planner", "max_tasks": 5},
                {"agent_id": f"coder-{i}", "specialization": "coder", "max_tasks": 5}
            ]
            args = {
                "agents": agents,
                "swarm_id": f"benchmark-swarm-{i}"
            }
            swarm.handle_create_swarm(args, self.server)
            create_time = (time.time() - start) * 1000
            create_times.append(create_time)
            
            # Benchmark task execution
            start = time.time()
            task = {"task_id": f"task-{i}", "description": "Benchmark task"}
            exec_args = {
                "swarm_id": f"benchmark-swarm-{i}",
                "task": task,
                "specialization": "planner",
                "timeout": 30.0
            }
            swarm.handle_execute_swarm_task(exec_args, self.server)
            execute_time = (time.time() - start) * 1000
            execute_times.append(execute_time)
        
        # Calculate statistics
        print("\n--- Swarm Creation Performance ---")
        self.print_metric("Mean", mean(create_times), "ms", target=100)
        self.print_metric("Median", median(create_times), "ms")
        
        print("\n--- Task Execution Performance ---")
        self.print_metric("Mean", mean(execute_times), "ms", target=1000)
        self.print_metric("Median", median(execute_times), "ms")
        self.print_metric("Std Dev", stdev(execute_times) if len(execute_times) > 1 else 0, "ms")
        
        self.results["swarm"] = {
            "create_mean_ms": mean(create_times),
            "execute_mean_ms": mean(execute_times),
            "execute_median_ms": median(execute_times)
        }
    
    def generate_report(self):
        """Generate performance report"""
        self.print_section("PERFORMANCE SUMMARY")
        
        print("\n--- Ephemeral Agents ---")
        print(f"Creation: {self.results['ephemeral_agents']['creation_mean_ms']:.2f}ms (target: <100ms)")
        print(f"Release: {self.results['ephemeral_agents']['release_mean_ms']:.2f}ms (target: <50ms)")
        
        print("\n--- File Locking ---")
        print(f"Acquire: {self.results['file_locking']['acquire_mean_ms']:.2f}ms (target: <10ms)")
        print(f"Release: {self.results['file_locking']['release_mean_ms']:.2f}ms (target: <5ms)")
        
        print("\n--- Workflows ---")
        print(f"Parallel (5 tasks): {self.results['workflows']['parallel_mean_ms']:.2f}ms (target: <5000ms)")
        
        print("\n--- Swarm Pattern ---")
        print(f"Create: {self.results['swarm']['create_mean_ms']:.2f}ms (target: <100ms)")
        print(f"Execute: {self.results['swarm']['execute_mean_ms']:.2f}ms (target: <1000ms)")
        
        # Check if all targets met
        targets_met = (
            self.results['ephemeral_agents']['creation_mean_ms'] < 100 and
            self.results['file_locking']['acquire_mean_ms'] < 10 and
            self.results['workflows']['parallel_mean_ms'] < 5000 and
            self.results['swarm']['execute_mean_ms'] < 1000
        )
        
        print("\n" + "="*70)
        if targets_met:
            print("[SUCCESS] All performance targets met!")
        else:
            print("[WARNING] Some performance targets not met")
        print("="*70)
        
        return targets_met
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("\n" + "="*70)
        print("  PHASE 2 PERFORMANCE BENCHMARKS")
        print("  Validating timeout values and performance targets")
        print("="*70)
        
        self.benchmark_ephemeral_agents(iterations=10)
        self.benchmark_file_locking(iterations=10)
        self.benchmark_workflows(iterations=5)
        self.benchmark_swarm(iterations=5)
        
        return self.generate_report()


def main():
    """Main entry point"""
    benchmarks = Phase2PerformanceBenchmarks()
    success = benchmarks.run_all_benchmarks()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

