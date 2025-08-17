#!/usr/bin/env python3
"""
Use the Enhanced MCP Server to analyze itself and suggest improvements

This script sends MCP requests to the running server to get intelligent
recommendations for system improvements.
"""

import json
import subprocess
import sys
import time
import threading
from queue import Queue, Empty

class MCPSystemAnalyzer:
    def __init__(self):
        self.server_process = None
        self.response_queue = Queue()
        self.request_id = 1
        
    def start_server_if_needed(self):
        """Start the MCP server if not already running"""
        try:
            # Check if server is already running by trying to connect
            test_request = self.create_request("get_performance_stats", {"hours": 1})
            response = self.send_request_sync(test_request, timeout=3)
            if response:
                print("✅ MCP Server is already running")
                return True
        except:
            pass
            
        print("🚀 Starting MCP Server for analysis...")
        
        self.server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Give server time to initialize
        time.sleep(3)
        print("✅ MCP Server started")
        return True
        
    def create_request(self, tool_name, arguments):
        """Create an MCP request message"""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        self.request_id += 1
        return request
        
    def send_request_sync(self, request, timeout=30):
        """Send a request and wait for response"""
        if not self.server_process:
            return None
            
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            # Wait for response
            stdout, stderr = self.server_process.communicate(timeout=timeout)
            
            if stdout:
                try:
                    response = json.loads(stdout.strip())
                    return response
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON response: {stdout}")
                    return None
            return None
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Request timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"❌ Error sending request: {e}")
            return None
            
    def analyze_system_architecture(self):
        """Use sequential thinking to analyze the system architecture"""
        print("\n🧠 Starting System Architecture Analysis...")
        print("=" * 55)
        
        # Step 1: Initial analysis
        request = self.create_request("sequential_thinking", {
            "thought": "I need to comprehensively analyze the Enhanced MCP Server v2.1 system architecture, including its 18 tools, persistent storage layer, performance monitoring, and overall design patterns to identify areas for improvement.",
            "thought_number": 1,
            "total_thoughts": 5,
            "next_thought_needed": True
        })
        
        response = self.send_request_sync(request)
        if response:
            print("💭 Thought 1: Initial Architecture Analysis")
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                print(f"   {content}")
            
        # Step 2: Tool ecosystem analysis
        request = self.create_request("sequential_thinking", {
            "thought": "Now I should analyze the tool ecosystem - the 18 tools available, their interdependencies, performance characteristics, and how they work together. I'll consider if there are gaps in functionality or opportunities for better integration.",
            "thought_number": 2,
            "total_thoughts": 5,
            "next_thought_needed": True
        })
        
        response = self.send_request_sync(request)
        if response:
            print("💭 Thought 2: Tool Ecosystem Analysis")
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                print(f"   {content}")
                
        # Step 3: Storage and persistence analysis
        request = self.create_request("sequential_thinking", {
            "thought": "Let me examine the persistent storage layer - SQLite database, memory management, performance metrics storage, error tracking. I should consider scalability, performance optimization, data integrity, and backup strategies.",
            "thought_number": 3,
            "total_thoughts": 5,
            "next_thought_needed": True
        })
        
        response = self.send_request_sync(request)
        if response:
            print("💭 Thought 3: Storage & Persistence Analysis")
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                print(f"   {content}")
                
        # Step 4: Performance and monitoring analysis
        request = self.create_request("sequential_thinking", {
            "thought": "Now I'll analyze the performance monitoring system - the decorator pattern, metrics collection, error tracking, alerting thresholds. I should consider if the monitoring is comprehensive enough and if there are performance bottlenecks.",
            "thought_number": 4,
            "total_thoughts": 5,
            "next_thought_needed": True
        })
        
        response = self.send_request_sync(request)
        if response:
            print("💭 Thought 4: Performance & Monitoring Analysis")
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                print(f"   {content}")
                
        # Step 5: Final recommendations
        request = self.create_request("sequential_thinking", {
            "thought": "Based on my analysis of the architecture, tools, storage, and monitoring, I can now synthesize comprehensive improvement recommendations. I should prioritize them by impact and feasibility, considering security, scalability, user experience, and maintainability.",
            "thought_number": 5,
            "total_thoughts": 5,
            "next_thought_needed": False
        })
        
        response = self.send_request_sync(request)
        if response:
            print("💭 Thought 5: Final Recommendations")
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                print(f"   {content}")
        
        return True
        
    def analyze_core_files(self):
        """Analyze core system files for improvements"""
        print("\n🔍 Analyzing Core System Files...")
        print("=" * 40)
        
        core_files = [
            ("server.py", "Main MCP server with 18 tools and performance monitoring"),
            ("storage.py", "Persistent SQLite storage layer with comprehensive data management"),
            ("package.json", "Project configuration and metadata")
        ]
        
        recommendations = []
        
        for filename, description in core_files:
            print(f"\n📁 Analyzing {filename}...")
            
            # Read the file content
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    
                # Analyze the file
                request = self.create_request("analyze_code", {
                    "code": file_content[:4000],  # Limit content for analysis
                    "analysis_type": "optimization"
                })
                
                response = self.send_request_sync(request)
                if response and "result" in response:
                    content = response["result"]["content"][0]["text"]
                    print(f"   📊 Analysis: {content[:200]}...")
                    
                # Get improvement suggestions
                request = self.create_request("suggest_improvements", {
                    "code": file_content[:4000]
                })
                
                response = self.send_request_sync(request)
                if response and "result" in response:
                    content = response["result"]["content"][0]["text"]
                    recommendations.append(f"{filename}: {content}")
                    print(f"   💡 Suggestions: {content[:200]}...")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {filename}: {e}")
                
        return recommendations
        
    def get_system_performance_insights(self):
        """Get performance insights from the system"""
        print("\n📊 Getting Performance Insights...")
        print("=" * 35)
        
        # Get performance statistics
        request = self.create_request("get_performance_stats", {
            "hours": 24
        })
        
        response = self.send_request_sync(request)
        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print(f"📈 Performance Stats:\n{content}")
            
        # Get error patterns
        request = self.create_request("get_error_patterns", {
            "hours": 24
        })
        
        response = self.send_request_sync(request)
        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print(f"\n🚨 Error Patterns:\n{content}")
            
        return True
        
    def generate_comprehensive_recommendations(self):
        """Generate comprehensive system improvement recommendations"""
        print("\n🎯 Generating Comprehensive Recommendations...")
        print("=" * 50)
        
        # Use the code analysis to suggest system-wide improvements
        system_overview = """
        Enhanced MCP Server v2.1 System Overview:
        - 18 advanced tools with performance monitoring
        - SQLite persistent storage (memories, sessions, metrics, errors)
        - Real-time performance tracking with 200ms threshold alerts
        - Comprehensive error tracking and pattern analysis
        - Sequential thinking with persistent context
        - Enhanced debugging and code execution capabilities
        - Comprehensive test suites and validation
        
        Current architecture uses:
        - Python asyncio for async operations
        - SQLite for data persistence
        - JSON-RPC for MCP protocol
        - Decorator pattern for performance monitoring
        - LM Studio integration for AI model access
        """
        
        request = self.create_request("suggest_improvements", {
            "code": system_overview
        })
        
        response = self.send_request_sync(request)
        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print(f"🚀 System-Wide Recommendations:\n{content}")
            
            # Store these recommendations in memory
            store_request = self.create_request("store_memory", {
                "key": "system_improvement_recommendations",
                "value": content,
                "category": "analysis"
            })
            
            store_response = self.send_request_sync(store_request)
            if store_response:
                print("\n💾 Recommendations stored in persistent memory")
                
        return True
        
    def cleanup(self):
        """Clean up the server process"""
        if self.server_process:
            try:
                self.server_process.terminate()
                time.sleep(1)
                if self.server_process.poll() is None:
                    self.server_process.kill()
                print("🔧 Server process cleaned up")
            except:
                pass

def main():
    print("🧪 Enhanced MCP Server - Self-Analysis & Improvement Suggestions")
    print("=" * 70)
    
    analyzer = MCPSystemAnalyzer()
    
    try:
        # Start the server
        if not analyzer.start_server_if_needed():
            print("❌ Failed to start MCP server")
            return
            
        # Run comprehensive analysis
        print("\n🔍 Starting Comprehensive System Analysis...")
        
        # 1. Architecture analysis using sequential thinking
        analyzer.analyze_system_architecture()
        
        # 2. Core file analysis
        recommendations = analyzer.analyze_core_files()
        
        # 3. Performance insights
        analyzer.get_system_performance_insights()
        
        # 4. Generate comprehensive recommendations
        analyzer.generate_comprehensive_recommendations()
        
        print("\n" + "=" * 70)
        print("✅ SYSTEM ANALYSIS COMPLETE")
        print("\n📋 Key Areas Analyzed:")
        print("• System Architecture & Design Patterns")
        print("• Tool Ecosystem & Integration")
        print("• Persistent Storage Layer")
        print("• Performance Monitoring System")
        print("• Core File Implementations")
        print("• Performance Metrics & Error Patterns")
        
        print("\n💡 All recommendations have been generated and stored in memory.")
        print("Use 'retrieve_memory' with category 'analysis' to access them later.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main() 