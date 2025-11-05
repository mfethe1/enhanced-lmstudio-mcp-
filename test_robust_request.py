#!/usr/bin/env python3
"""
Test the _make_robust_lm_studio_request function directly
"""

import sys
import os

# Add the current directory to Python path so we can import server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import EnhancedLMStudioMCPServer, _make_robust_lm_studio_request

def test_robust_request():
    """Test the robust request function directly"""
    print("🔧 Testing _make_robust_lm_studio_request Function")
    print("=" * 60)
    
    # Initialize server
    server = EnhancedLMStudioMCPServer()
    
    # Test parameters
    model = "openai/gpt-oss-20b"
    messages = [{"role": "user", "content": "Hello, respond with just 'Hi there!'"}]
    temperature = 0.1
    max_tokens = 10
    top_p = None
    tools = []
    safe_tool_choice = "none"
    has_tools = False
    transcript = []
    
    print(f"Testing with LM Studio at: {server.base_url}")
    print(f"Using model: {model}")
    print("-" * 60)
    
    try:
        # Call the robust request function
        print("Calling _make_robust_lm_studio_request...")
        data = _make_robust_lm_studio_request(
            server, model, messages, temperature, max_tokens, top_p,
            tools, safe_tool_choice, has_tools, transcript
        )
        
        print(f"✅ Function returned: {type(data)}")
        
        if data is None:
            print("❌ Function returned None - all retries failed")
            print("Transcript:")
            for i, entry in enumerate(transcript):
                print(f"  {i+1}. {entry}")
        else:
            print("✅ Function returned data!")
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                print(f"Content: '{content}'")
                if content:
                    print("🎉 SUCCESS: Got non-empty response!")
                else:
                    print("❌ WARNING: Response is empty")
            else:
                print("❌ No choices in response")
                
            print(f"Transcript ({len(transcript)} entries):")
            for i, entry in enumerate(transcript):
                print(f"  {i+1}. {entry}")
                
        return data
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_robust_request()
    
    if result and result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content", "").strip()
        if content:
            print("\n🎉 SUCCESS: Robust request function is working!")
        else:
            print("\n❌ ISSUE: Function works but returns empty content")
    else:
        print("\n❌ FAILED: Robust request function is not working")
