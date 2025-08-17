#!/usr/bin/env python3
"""
Quick Start Script for Enhanced LM Studio MCP Server

This script provides an easy way to start the enhanced MCP server with
proper configuration checking and helpful setup guidance.
"""

import os
import sys
import subprocess
import json
import requests
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    try:
        import requests
        import aiohttp
    except ImportError as e:
        missing_deps.append(str(e).split("'")[1])
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_lm_studio():
    """Check if LM Studio is running and accessible"""
    print("🔍 Checking LM Studio connection...")
    
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
    
    try:
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            if models.get("data"):
                print(f"✅ LM Studio is running at {lm_studio_url}")
                print(f"📊 Available models: {len(models['data'])}")
                return True
            else:
                print(f"⚠️  LM Studio is running but no models are loaded")
                return False
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to LM Studio at {lm_studio_url}")
        print("💡 Make sure LM Studio is running and the API server is enabled")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Connection to LM Studio timed out")
        return False

def create_default_config():
    """Create default configuration if it doesn't exist"""
    config_file = Path("config.json")
    if not config_file.exists():
        print("📝 Creating default configuration...")
        
        config = {
            "lm_studio": {
                "url": "http://localhost:1234",
                "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                "timeout": 120
            },
            "server": {
                "max_memory_entries": 1000,
                "code_execution_timeout": 30,
                "temp_file_cleanup": True
            },
            "safety": {
                "sandbox_execution": True,
                "max_file_size": "10MB"
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created {config_file}")

def display_startup_info():
    """Display startup information and capabilities"""
    print("\n🚀 Enhanced LM Studio MCP Server v2.0")
    print("=" * 50)
    print("🛠  Available Tool Categories:")
    print("   • Sequential Thinking & Problem Solving")
    print("   • Advanced Code Analysis & Explanations") 
    print("   • Safe Code Execution & Testing")
    print("   • File System Operations")
    print("   • Memory & Context Management")
    print("   • Enhanced Debugging Tools")
    print()
    print("📋 Total Tools: 16 advanced capabilities")
    print("🧠 Memory: Persistent across sessions")
    print("🔒 Safety: Sandboxed execution environment")
    print()

def start_server():
    """Start the enhanced MCP server"""
    print("🎯 Starting Enhanced MCP Server...")
    print("📨 Server will communicate via stdin/stdout (MCP protocol)")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the server
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ server.py not found in current directory")
        return False
    
    return True

def main():
    """Main startup routine"""
    print("🧪 Enhanced MCP Server Quick Start")
    print("=" * 40)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Check LM Studio
    if not check_lm_studio():
        print("\n💡 Setup Instructions:")
        print("1. Start LM Studio")
        print("2. Enable the API server (Settings > Developer > API Server)")
        print("3. Load a compatible model (DeepSeek R1 recommended)")
        print("4. Verify the server is running on port 1234")
        response = input("\nTry again? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
        # Re-check
        if not check_lm_studio():
            print("❌ Still cannot connect to LM Studio")
            sys.exit(1)
    
    # Step 3: Create config if needed
    create_default_config()
    
    # Step 4: Display info
    display_startup_info()
    
    # Step 5: Start server
    start_server()

if __name__ == "__main__":
    main() 