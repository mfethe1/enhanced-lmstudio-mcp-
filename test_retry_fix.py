#!/usr/bin/env python3
"""
Test script to verify the chat_with_tools retry fix works with real LM Studio.
This script will test the retry logic by making requests and showing the transcript.
"""

import json
import os
from types import SimpleNamespace
from server import handle_chat_with_tools

def test_chat_with_tools_retry():
    """Test the enhanced retry logic with real LM Studio"""
    
    # Create fake server object
    server = SimpleNamespace()
    server.base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234").rstrip("/")
    server.model_name = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
    
    print(f"Testing chat_with_tools with LM Studio at: {server.base_url}")
    print(f"Using model: {server.model_name}")
    print("-" * 60)
    
    # Test 1: Simple request that should work
    print("TEST 1: Simple request")
    try:
        result = handle_chat_with_tools({
            "instruction": "Say hello and tell me what model you are.",
            "max_iters": 1,
            "temperature": 0.3,
            "max_tokens": 100
        }, server)
        
        print(f"✅ Success: {result['content'][:100]}...")
        print(f"📊 Transcript entries: {len(result.get('transcript', []))}")
        if result.get('transcript'):
            for entry in result['transcript']:
                print(f"   📝 {entry}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
    
    # Test 2: Request with tools (more likely to trigger empty responses)
    print("TEST 2: Request with tools")
    try:
        result = handle_chat_with_tools({
            "instruction": "List the files in the current directory and tell me about one of them.",
            "allowed_tools": ["read_file_content", "search_files"],
            "max_iters": 2,
            "temperature": 0.2,
            "tool_choice": "auto"
        }, server)
        
        print(f"✅ Success: {result['content'][:150]}...")
        print(f"📊 Transcript entries: {len(result.get('transcript', []))}")
        if result.get('transcript'):
            for entry in result['transcript']:
                print(f"   📝 {entry}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
    
    # Test 3: Request that might trigger retries (high temperature, complex request)
    print("TEST 3: Complex request (may trigger retries)")
    try:
        result = handle_chat_with_tools({
            "instruction": "Analyze the server.py file and suggest improvements to the chat_with_tools function.",
            "allowed_tools": ["read_file_content"],
            "max_iters": 1,
            "temperature": 0.1,  # Very low temperature might cause empty responses
            "max_tokens": 50,    # Very low token limit might cause issues
            "tool_choice": "auto"
        }, server)
        
        if "ALL PROVIDERS FAILED" in result.get('content', ''):
            print(f"⚠️  Expected failure (testing retry logic): {result['content'][:100]}...")
            print(f"📊 Retry attempts logged: {len([t for t in result.get('transcript', []) if 'empty response' in str(t)])}")
        else:
            print(f"✅ Success: {result['content'][:100]}...")
        
        print(f"📊 Total transcript entries: {len(result.get('transcript', []))}")
        if result.get('transcript'):
            for entry in result['transcript']:
                print(f"   📝 {entry}")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()

if __name__ == "__main__":
    print("🔧 Testing Enhanced Chat_with_Tools Retry Logic")
    print("=" * 60)
    test_chat_with_tools_retry()
    print("=" * 60)
    print("✅ Test completed! Check the transcript entries above to see retry logic in action.")
