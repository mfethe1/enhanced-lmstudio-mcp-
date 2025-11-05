#!/usr/bin/env python3
"""
Test chat_with_tools with very short timeouts to quickly trigger fallback
"""

import sys
import os
import time

# Add the current directory to Python path so we can import server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set very short timeouts for testing
os.environ["HTTP_CONNECT_TIMEOUT"] = "2"  # 2 seconds to connect
os.environ["HTTP_READ_TIMEOUT_SIMPLE"] = "5"  # 5 seconds to read response

from server import EnhancedLMStudioMCPServer, handle_chat_with_tools

def test_quick_fallback():
    """Test chat_with_tools with quick timeouts to trigger fallback"""
    print("🔧 Testing Chat_with_Tools Quick Fallback")
    print("=" * 60)
    print("Using short timeouts: connect=2s, read=5s")
    print("This should quickly fallback to OpenAI/Anthropic if LM Studio is slow")
    print("-" * 60)
    
    # Initialize server
    server = EnhancedLMStudioMCPServer()
    
    # Test with the exact PLIP integration request that was failing
    test_args = {
        "instruction": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges).",
        "allowed_tools": [],
        "tool_choice": "auto",
        "max_iters": 4,
        "temperature": 0.2
    }
    
    print(f"Testing with LM Studio at: {server.base_url}")
    print("Request: PLIP integration planning (the original failing request)")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
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
        
        # Check if we got actual content (not an error)
        is_success = (
            content and 
            not content.startswith("PREVENTIVE MEASURES FAILED") and
            not content.startswith("ALL PROVIDERS FAILED") and
            not content.startswith("Failed to get response") and
            len(content) > 50  # Substantial content
        )
        
        if is_success:
            print(f"  ✅ SUCCESS: Got substantial content!")
            print(f"  Content preview: '{content[:150]}{'...' if len(content) > 150 else ''}'")
            
            # Check which provider was used
            provider_used = "Unknown"
            if "openai" in model.lower():
                provider_used = "OpenAI"
            elif "anthropic" in model.lower() or "claude" in model.lower():
                provider_used = "Anthropic"
            elif "gpt-oss" in model.lower() or "lmstudio" in str(transcript).lower():
                provider_used = "LM Studio"
            
            print(f"  🎯 Provider used: {provider_used}")
            
            # Check transcript for retry/fallback behavior
            retry_count = 0
            fallback_used = False
            for entry in transcript:
                entry_str = str(entry).lower()
                if "retry" in entry_str or "attempt" in entry_str:
                    retry_count += 1
                if "fallback" in entry_str or "openai" in entry_str or "anthropic" in entry_str:
                    fallback_used = True
            
            if retry_count > 0:
                print(f"  🔄 Retry attempts: {retry_count}")
            if fallback_used:
                print(f"  🔀 Fallback providers were used")
                
        else:
            print(f"  ❌ ISSUE: Got error or insufficient content")
            print(f"  Content: '{content[:300]}{'...' if len(content) > 300 else ''}'")
        
        # Show key transcript entries
        if transcript:
            print(f"\n📋 Key Transcript Entries:")
            for i, entry in enumerate(transcript):
                entry_str = str(entry)
                # Show entries that indicate important events
                if any(keyword in entry_str.lower() for keyword in ['retry', 'fallback', 'failed', 'success', 'attempt', 'timeout']):
                    print(f"  {i+1}. {entry}")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_quick_fallback()
    
    if result:
        content = result.get('content', '')
        if (content and 
            not content.startswith('PREVENTIVE MEASURES FAILED') and 
            not content.startswith('ALL PROVIDERS FAILED') and
            len(content) > 50):
            print("\n🎉 SUCCESS: Chat_with_tools is working!")
            print("✅ The retry logic and fallback providers are functioning correctly.")
            print("✅ Users will get meaningful responses instead of empty content.")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Function works but may need tuning")
            print("Check the content and transcript above for details.")
    else:
        print("\n❌ FAILED: Could not test function")
