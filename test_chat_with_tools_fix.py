#!/usr/bin/env python3
"""
Test script to verify the fixed chat_with_tools function works correctly.
This will test the new robust retry logic with LM Studio.
"""

import json
import sys
import os

# Add the current directory to Python path so we can import server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import EnhancedLMStudioMCPServer

def test_chat_with_tools_fix():
    """Test the fixed chat_with_tools function"""
    print("🔧 Testing Fixed Chat_with_Tools Function")
    print("=" * 60)
    
    # Initialize server
    server = EnhancedLMStudioMCPServer()
    
    # Test arguments that previously failed
    test_args = {
        "instruction": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges).",
        "allowed_tools": [],
        "tool_choice": "auto",
        "max_iters": 4,
        "temperature": 0.5,
        "max_tokens": 280
    }
    
    print(f"Testing with LM Studio at: {server.base_url}")
    print(f"Using model: {getattr(server, 'model_name', 'openai/gpt-oss-20b')}")
    print("-" * 60)
    
    try:
        # Call the fixed function
        from server import handle_chat_with_tools
        result = handle_chat_with_tools(test_args, server)
        
        print("✅ SUCCESS: Got response from chat_with_tools")
        print(f"Content length: {len(result.get('content', ''))}")
        print(f"Model used: {result.get('model', 'unknown')}")
        print(f"Tool choice: {result.get('tool_choice', 'unknown')}")
        
        # Print transcript to see retry behavior
        transcript = result.get('transcript', [])
        if transcript:
            print(f"\n📋 Transcript ({len(transcript)} entries):")
            for i, entry in enumerate(transcript[:10]):  # Show first 10 entries
                print(f"  {i+1}. {entry}")
            if len(transcript) > 10:
                print(f"  ... and {len(transcript) - 10} more entries")
        
        # Print first 200 chars of content
        content = result.get('content', '')
        if content:
            print(f"\n📝 Content preview:")
            print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
        else:
            print("\n❌ WARNING: Content is empty!")
            
        # Check for hints (error indicators)
        hints = result.get('hints', {})
        if hints:
            print(f"\n⚠️  Hints provided (indicates issues):")
            for key, value in hints.items():
                print(f"  {key}: {value}")
                
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_chat_with_tools_fix()
    
    if result:
        # Check if we got actual content (not an error message)
        content = result.get('content', '')
        if content and not content.startswith('PREVENTIVE MEASURES FAILED'):
            print("\n🎉 SUCCESS: Fixed chat_with_tools is working!")
        else:
            print("\n❌ STILL FAILING: Need more fixes")
    else:
        print("\n❌ FAILED: Could not test function")
