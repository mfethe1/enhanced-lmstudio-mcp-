#!/usr/bin/env python3
"""
Test the HTTP client timeout behavior directly
"""

import sys
import os
import time

# Set very short timeouts
os.environ["HTTP_CONNECT_TIMEOUT"] = "2"
os.environ["HTTP_READ_TIMEOUT_SIMPLE"] = "5"

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import _http_client

def test_http_timeout():
    """Test HTTP client timeout behavior"""
    print("🔧 Testing HTTP Client Timeout Behavior")
    print("=" * 50)
    print("Timeouts: connect=2s, read=5s")
    print("-" * 50)
    
    # Test payload
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print("Testing HTTP request to LM Studio...")
    start_time = time.time()
    
    try:
        data = _http_client.post_sync(
            "http://localhost:1234/v1/chat/completions",
            json_data=payload,
            headers={"Content-Type": "application/json"},
            operation_type="simple"
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Request completed in {elapsed:.1f} seconds")
        
        if data:
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                print(f"Response: '{content}'")
                if content:
                    print("✅ Got non-empty response!")
                else:
                    print("❌ Got empty response")
            else:
                print("❌ No choices in response")
        else:
            print("❌ No data returned")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Request failed after {elapsed:.1f} seconds: {e}")
        
        # Check if it's a timeout
        if "timeout" in str(e).lower():
            print("✅ Timeout working correctly!")
            return True
        else:
            print("⚠️  Different error (not timeout)")
            return False
    
    return True

if __name__ == "__main__":
    success = test_http_timeout()
    if success:
        print("\n🎯 HTTP client behavior understood")
    else:
        print("\n❌ HTTP client has issues")
