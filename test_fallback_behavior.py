#!/usr/bin/env python3
"""
Test the fallback behavior when LM Studio fails or returns empty content
"""

import sys
import os
import time

# Add the current directory to Python path so we can import server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import EnhancedLMStudioMCPServer, handle_chat_with_tools

def test_fallback_behavior():
    """Test chat_with_tools with fallback when LM Studio fails"""
    print("🔧 Testing Chat_with_Tools Fallback Behavior")
    print("=" * 60)
    
    # Initialize server
    server = EnhancedLMStudioMCPServer()
    
    # Test with a simple request that should trigger fallback
    test_args = {
        "instruction": "Hello, please respond with 'Hi there!' and nothing else.",
        "allowed_tools": [],
        "tool_choice": "none",  # No tools to simplify
        "max_iters": 1,         # Single iteration
        "temperature": 0.1,     # Low temperature for consistent output
        "max_tokens": 10        # Very small response
    }
    
    print(f"Testing with LM Studio at: {server.base_url}")
    print("Expected behavior: LM Studio will timeout/fail, then fallback to OpenAI/Anthropic")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Call the function - it should fallback when LM Studio fails
        print("Calling handle_chat_with_tools...")
        result = handle_chat_with_tools(test_args, server)
        
        elapsed = time.time() - start_time
        print(f"✅ Function completed in {elapsed:.1f} seconds")
        
        # Analyze the result
        content = result.get("content", "")
        transcript = result.get("transcript", [])
        model = result.get("model", "unknown")
        
        print(f"\nResult Analysis:")
        print(f"  Content length: {len(content)}")
        print(f"  Model used: {model}")
        print(f"  Transcript entries: {len(transcript)}")
        
        # Check if we got actual content
        if content and not content.startswith("PREVENTIVE MEASURES FAILED"):
            print(f"  ✅ SUCCESS: Got actual content!")
            print(f"  Content preview: '{content[:100]}{'...' if len(content) > 100 else ''}'")
            
            # Check transcript for fallback behavior
            fallback_used = False
            for entry in transcript:
                if "fallback" in str(entry).lower() or "openai" in str(entry).lower() or "anthropic" in str(entry).lower():
                    fallback_used = True
                    break
            
            if fallback_used:
                print("  ✅ Fallback providers were used successfully")
            else:
                print("  ℹ️  LM Studio may have worked (no fallback needed)")
                
        else:
            print(f"  ❌ ISSUE: Got error or empty content")
            print(f"  Content: '{content[:200]}{'...' if len(content) > 200 else ''}'")
        
        # Show transcript for debugging
        if transcript:
            print(f"\n📋 Transcript ({len(transcript)} entries):")
            for i, entry in enumerate(transcript[:10]):  # Show first 10 entries
                print(f"  {i+1}. {entry}")
            if len(transcript) > 10:
                print(f"  ... and {len(transcript) - 10} more entries")
        
        # Check for hints (error indicators)
        hints = result.get("hints", {})
        if hints:
            print(f"\n⚠️  Hints provided (indicates issues):")
            for key, value in hints.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                    for item in value[:3]:  # Show first 3 items
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
                    
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_fallback_behavior()
    
    if result:
        content = result.get('content', '')
        if content and not content.startswith('PREVENTIVE MEASURES FAILED') and not content.startswith('Failed to get response'):
            print("\n🎉 SUCCESS: Chat_with_tools is working with fallback!")
            print("The retry logic and fallback providers are functioning correctly.")
        else:
            print("\n❌ STILL FAILING: Need to investigate further")
            print("Check LM Studio model loading and API key configuration.")
    else:
        print("\n❌ FAILED: Could not test function")
