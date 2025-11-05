#!/usr/bin/env python3
"""
Test the fallback providers directly (bypass LM Studio)
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Temporarily disable LM Studio by setting a bad URL
os.environ["LMSTUDIO_API_BASE"] = "http://localhost:9999"  # Non-existent port

from server import EnhancedLMStudioMCPServer, handle_chat_with_tools

def test_fallback_only():
    """Test chat_with_tools fallback when LM Studio is unavailable"""
    print("🔧 Testing Chat_with_Tools Fallback Only")
    print("=" * 60)
    print("LM Studio disabled (port 9999) - should fallback immediately")
    print("-" * 60)
    
    # Initialize server with bad LM Studio URL
    server = EnhancedLMStudioMCPServer()
    print(f"LM Studio URL (should fail): {server.base_url}")
    
    # Test with the original PLIP integration request
    test_args = {
        "instruction": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges).",
        "allowed_tools": [],
        "tool_choice": "auto",
        "max_iters": 4,
        "temperature": 0.2
    }
    
    print("Request: PLIP integration planning (original failing request)")
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
            len(content) > 100  # Substantial content for PLIP question
        )
        
        if is_success:
            print(f"  ✅ SUCCESS: Got substantial content!")
            print(f"  Content preview: '{content[:200]}{'...' if len(content) > 200 else ''}'")
            
            # Determine which provider was used
            provider_used = "Unknown"
            if "gpt" in model.lower() or "openai" in model.lower():
                provider_used = "OpenAI"
            elif "claude" in model.lower() or "anthropic" in model.lower():
                provider_used = "Anthropic"
            
            print(f"  🎯 Provider used: {provider_used}")
            
            # Check if the content is relevant to PLIP
            plip_keywords = ["plip", "protein", "ligand", "pdb", "interaction", "bond"]
            relevant_keywords = sum(1 for keyword in plip_keywords if keyword.lower() in content.lower())
            print(f"  📊 PLIP relevance: {relevant_keywords}/{len(plip_keywords)} keywords found")
            
            if relevant_keywords >= 3:
                print(f"  ✅ Content is highly relevant to PLIP integration!")
            else:
                print(f"  ⚠️  Content may not be specific to PLIP")
                
        else:
            print(f"  ❌ ISSUE: Got error or insufficient content")
            print(f"  Content: '{content[:300]}{'...' if len(content) > 300 else ''}'")
        
        # Show key transcript entries
        if transcript:
            print(f"\n📋 Key Transcript Entries:")
            fallback_entries = []
            error_entries = []
            
            for i, entry in enumerate(transcript):
                entry_str = str(entry).lower()
                if any(keyword in entry_str for keyword in ['fallback', 'openai', 'anthropic', 'failed', 'timeout']):
                    if 'fallback' in entry_str or 'openai' in entry_str or 'anthropic' in entry_str:
                        fallback_entries.append(f"  {i+1}. {entry}")
                    elif 'failed' in entry_str or 'timeout' in entry_str:
                        error_entries.append(f"  {i+1}. {entry}")
            
            if error_entries:
                print("  Errors/Timeouts:")
                for entry in error_entries[:3]:  # Show first 3
                    print(entry)
                    
            if fallback_entries:
                print("  Fallback Usage:")
                for entry in fallback_entries[:3]:  # Show first 3
                    print(entry)
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_fallback_only()
    
    if result:
        content = result.get('content', '')
        if (content and 
            not content.startswith('PREVENTIVE MEASURES FAILED') and 
            not content.startswith('ALL PROVIDERS FAILED') and
            len(content) > 100):
            print("\n🎉 SUCCESS: Chat_with_tools fallback is working!")
            print("✅ When LM Studio fails, fallback providers deliver quality responses.")
            print("✅ Users will get meaningful PLIP integration guidance instead of empty responses.")
            print("✅ The retry logic and fallback system are functioning correctly.")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Function works but content may need improvement")
            print("Check the content quality and relevance above.")
    else:
        print("\n❌ FAILED: Fallback system is not working")
