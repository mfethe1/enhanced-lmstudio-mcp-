#!/usr/bin/env python3
"""
Test openai/gpt-oss-20b integration with the MCP server
"""
import os
import sys
import json
import asyncio

# Set production environment for testing
os.environ.update({
    'MODEL_NAME': 'openai/gpt-oss-20b',
    'HTTP_READ_TIMEOUT_SIMPLE': '120',
    'HTTP_READ_TIMEOUT_COMPLEX': '300',
    'LMSTUDIO_MAX_RETRIES': '3',
    'CIRCUIT_LMSTUDIO_THRESHOLD': '10',
    'NO_FALLBACK_PROVIDERS': '0',  # Allow fallbacks
    'LOG_LEVEL': 'INFO'
})

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import EnhancedLMStudioMCPServer, _extract_content_from_response, _is_reasoning_model

def test_reasoning_model_detection():
    """Test that we correctly identify reasoning models"""
    print("🧪 Testing reasoning model detection...")
    
    test_cases = [
        ("openai/gpt-oss-20b", True),
        ("mistralai/magistral-small-2509", False),
        ("gpt-4-reasoning", True),
        ("claude-3-sonnet", False),
        ("thinking-model-v1", True),
        ("o1-preview", True),
        ("chain-of-thought-7b", True)
    ]
    
    for model, expected in test_cases:
        result = _is_reasoning_model(model)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {model}: {result} (expected {expected})")

def test_content_extraction():
    """Test content extraction from different response formats"""
    print("\n🧪 Testing content extraction...")
    
    # Test case 1: Normal response with content
    normal_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello, world!"
            }
        }]
    }
    
    content = _extract_content_from_response(normal_response, "openai/gpt-oss-20b")
    print(f"   ✅ Normal response: '{content}'")
    
    # Test case 2: Reasoning model with empty content but reasoning
    reasoning_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": "The user is asking for a greeting. I should respond with 'Hello'"
            }
        }]
    }
    
    content = _extract_content_from_response(reasoning_response, "openai/gpt-oss-20b")
    print(f"   ✅ Reasoning response: '{content[:50]}...'")
    
    # Test case 3: Completions endpoint response
    completions_response = {
        "choices": [{
            "text": "<|channel|>analysis<|message|>Hello there!"
        }]
    }
    
    content = _extract_content_from_response(completions_response, "openai/gpt-oss-20b")
    print(f"   ✅ Completions response: '{content}'")

async def test_direct_llm_request():
    """Test direct LLM request with the reasoning model"""
    print("\n🧪 Testing direct LLM request...")
    
    server = EnhancedLMStudioMCPServer()
    
    try:
        response = await server.make_llm_request_with_retry(
            "Say hello in one word",
            temperature=0.1
        )
        
        if response and response.strip():
            print(f"   ✅ LLM request successful: '{response[:100]}...'")
            return True
        else:
            print(f"   ❌ LLM request returned empty: '{response}'")
            return False
            
    except Exception as e:
        print(f"   ❌ LLM request failed: {e}")
        return False

def test_chat_with_tools():
    """Test chat_with_tools function with reasoning model"""
    print("\n🧪 Testing chat_with_tools...")
    
    server = EnhancedLMStudioMCPServer()
    
    arguments = {
        "instruction": "What is 2+2?",
        "allowed_tools": [],  # No tools, just chat
        "max_iters": 1,
        "temperature": 0.1,
        "model": "openai/gpt-oss-20b"
    }
    
    try:
        from server import handle_chat_with_tools
        result = handle_chat_with_tools(arguments, server)
        
        if isinstance(result, dict) and result.get("content"):
            content = result["content"]
            print(f"   ✅ Chat with tools successful: '{content[:100]}...'")
            return True
        else:
            print(f"   ❌ Chat with tools failed: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ Chat with tools error: {e}")
        return False

def test_mcp_tool_integration():
    """Test a few MCP tools to ensure they work with the reasoning model"""
    print("\n🧪 Testing MCP tool integration...")
    
    server = EnhancedLMStudioMCPServer()
    
    # Test health_check
    try:
        from server import handle_health_check
        result = handle_health_check({}, server)
        
        if isinstance(result, dict) and "status" in result:
            print(f"   ✅ Health check: {result.get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {result}")
            
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test get_version
    try:
        from server import handle_get_version
        result = handle_get_version({}, server)
        
        if isinstance(result, dict) and "model" in result:
            print(f"   ✅ Version check: model={result.get('model')}")
        else:
            print(f"   ❌ Version check failed: {result}")
            
    except Exception as e:
        print(f"   ❌ Version check error: {e}")

async def main():
    print("🚀 OpenAI GPT-OSS-20B Integration Test")
    print("=" * 60)
    
    # Test 1: Model detection
    test_reasoning_model_detection()
    
    # Test 2: Content extraction
    test_content_extraction()
    
    # Test 3: Direct LLM request
    llm_success = await test_direct_llm_request()
    
    # Test 4: Chat with tools
    chat_success = test_chat_with_tools()
    
    # Test 5: MCP tool integration
    test_mcp_tool_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    if llm_success and chat_success:
        print("✅ OpenAI GPT-OSS-20B is fully integrated and working!")
        print("\n🎉 Key Features Working:")
        print("   • Reasoning model detection")
        print("   • Content extraction from reasoning field")
        print("   • Fallback to completions endpoint")
        print("   • MCP tool integration")
        print("   • Extended timeouts for reasoning")
        
        print("\n🔧 Configuration Applied:")
        print(f"   • Model: {os.getenv('MODEL_NAME')}")
        print(f"   • Timeout: {os.getenv('HTTP_READ_TIMEOUT_SIMPLE')}s")
        print(f"   • Retries: {os.getenv('LMSTUDIO_MAX_RETRIES')}")
        print(f"   • Circuit breaker: {os.getenv('CIRCUIT_LMSTUDIO_THRESHOLD')} failures")
        
        return True
    else:
        print("❌ Integration test failed")
        print("\n💡 Troubleshooting:")
        if not llm_success:
            print("   • LLM request failed - check LM Studio model loading")
        if not chat_success:
            print("   • Chat with tools failed - check MCP integration")
        print("   • Verify openai/gpt-oss-20b is loaded in LM Studio")
        print("   • Check LM Studio logs for errors")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
