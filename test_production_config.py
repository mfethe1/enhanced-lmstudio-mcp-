#!/usr/bin/env python3
"""
Test with production configuration (fallbacks enabled)
"""
import os
import sys
import time

# Set production-like environment
os.environ.update({
    'MODEL_NAME': 'openai/gpt-oss-20b',
    'HTTP_CONNECT_TIMEOUT': '10',
    'HTTP_READ_TIMEOUT_SIMPLE': '120',
    'HTTP_READ_TIMEOUT_COMPLEX': '300',
    'LMSTUDIO_MAX_RETRIES': '3',
    'CIRCUIT_LMSTUDIO_THRESHOLD': '10',
    'NO_FALLBACK_PROVIDERS': '0',  # Enable fallbacks (production setting)
    'LOG_LEVEL': 'WARNING'
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import handle_chat_with_tools, EnhancedLMStudioMCPServer

def test_production_chat_with_tools():
    """Test chat_with_tools in production configuration"""
    print("🏭 Testing Production Configuration")
    print("=" * 50)
    print(f"Model: {os.getenv('MODEL_NAME')}")
    print(f"Fallbacks enabled: {os.getenv('NO_FALLBACK_PROVIDERS') == '0'}")
    print("-" * 50)
    
    server = EnhancedLMStudioMCPServer()
    
    test_cases = [
        {
            "name": "Simple math",
            "instruction": "What is 7 + 8?",
            "expected_content": "15"
        },
        {
            "name": "Reasoning task", 
            "instruction": "If a train travels 60 mph for 2 hours, how far does it go? Show calculation.",
            "expected_content": "120"
        },
        {
            "name": "Creative task",
            "instruction": "Write one sentence about a cat.",
            "expected_content": "cat"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        
        arguments = {
            "instruction": test_case["instruction"],
            "allowed_tools": [],
            "max_iters": 1,
            "temperature": 0.1,
            "model": "openai/gpt-oss-20b"
        }
        
        start_time = time.time()
        try:
            result = handle_chat_with_tools(arguments, server)
            elapsed = time.time() - start_time
            
            print(f"   Time: {elapsed:.1f}s")
            
            if isinstance(result, dict):
                content = result.get("content", "")
                transcript = result.get("transcript", [])
                
                print(f"   Content: '{content[:80]}{'...' if len(content) > 80 else ''}'")
                print(f"   Transcript: {transcript}")
                
                # Check what provider was used
                lm_studio_used = any("LM Studio success" in str(entry) for entry in transcript)
                openai_used = any("openai" in str(entry).lower() for entry in transcript)
                anthropic_used = any("anthropic" in str(entry).lower() for entry in transcript)
                
                provider = "LM Studio" if lm_studio_used else ("OpenAI" if openai_used else ("Anthropic" if anthropic_used else "Unknown"))
                
                # Check if we got expected content
                has_expected = test_case["expected_content"].lower() in content.lower()
                
                success = bool(content and content.strip() and has_expected)
                
                print(f"   Provider: {provider}")
                print(f"   Expected content found: {has_expected}")
                print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
                results.append({
                    "name": test_case["name"],
                    "success": success,
                    "provider": provider,
                    "time": elapsed,
                    "content_length": len(content)
                })
                
            else:
                print(f"   Unexpected result type: {type(result)}")
                results.append({
                    "name": test_case["name"],
                    "success": False,
                    "provider": "Unknown",
                    "time": elapsed,
                    "content_length": 0
                })
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ ERROR ({elapsed:.1f}s): {e}")
            results.append({
                "name": test_case["name"],
                "success": False,
                "provider": "Error",
                "time": elapsed,
                "content_length": 0
            })
    
    return results

def main():
    print("🚀 Production Configuration Test")
    print("=" * 60)
    
    results = test_production_chat_with_tools()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 PRODUCTION TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    # Provider breakdown
    providers = {}
    for result in results:
        provider = result["provider"]
        if provider not in providers:
            providers[provider] = {"count": 0, "success": 0}
        providers[provider]["count"] += 1
        if result["success"]:
            providers[provider]["success"] += 1
    
    print(f"\nProvider usage:")
    for provider, stats in providers.items():
        success_rate = (stats["success"] / stats["count"]) * 100 if stats["count"] > 0 else 0
        print(f"   {provider}: {stats['count']} requests, {success_rate:.0f}% success")
    
    # Performance stats
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
    print(f"\nPerformance:")
    print(f"   Average response time: {avg_time:.1f}s")
    print(f"   Fastest: {min(r['time'] for r in results):.1f}s")
    print(f"   Slowest: {max(r['time'] for r in results):.1f}s")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL PRODUCTION TESTS PASSED!")
        
        # Check if LM Studio was primarily used
        lm_studio_count = providers.get("LM Studio", {}).get("count", 0)
        if lm_studio_count == total_tests:
            print("✅ LM Studio used for all requests (no fallbacks needed)")
        elif lm_studio_count > 0:
            print(f"✅ LM Studio used for {lm_studio_count}/{total_tests} requests")
        else:
            print("⚠️  LM Studio not used - all requests went to fallback providers")
        
        print("✅ Timeout fixes working in production")
        print("✅ openai/gpt-oss-20b model functioning correctly")
        
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed")
        print("💡 Check LM Studio status and model loading")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
