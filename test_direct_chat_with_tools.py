#!/usr/bin/env python3
"""
Direct test of chat_with_tools functionality using handle_tool_call.
"""

import json
import os
import sys
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def test_direct_chat_with_tools():
    """Test chat_with_tools directly via handle_tool_call"""
    print("=" * 80)
    print("DIRECT TEST: chat_with_tools via handle_tool_call")
    print("=" * 80)
    
    # Set environment variables
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "LMSTUDIO_MODEL": "openai/gpt-oss-20b",
        "LMSTUDIO_FUNCTION_MODEL": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0",
        "HTTP_CONNECT_TIMEOUT": "2",
        "HTTP_READ_TIMEOUT_SIMPLE": "8",
        "HTTP_READ_TIMEOUT_COMPLEX": "45"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create server instance
    test_server = server.EnhancedLMStudioMCPServer()
    
    # Test simple case first
    print("Testing simple math question...")
    simple_args = {
        "instruction": "What is 2 + 2? Please provide a brief answer.",
        "allowed_tools": [],
        "tool_choice": "none",
        "max_iters": 1,
        "temperature": 0.1
    }
    
    try:
        result = server.handle_chat_with_tools(simple_args, test_server)
        
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            
            # Check for error patterns
            if 'content' in result and 'PREVENTIVE MEASURES FAILED' in str(result.get('content', '')):
                print("✗ Got old error pattern!")
                print(f"Content: {result['content']}")
                return False
            
            # Check for success indicators
            if 'final_response' in result:
                final_resp = str(result['final_response'])
                print(f"✓ Final response ({len(final_resp)} chars): {final_resp[:200]}...")
                
                if '4' in final_resp or 'four' in final_resp.lower():
                    print("✓ Contains expected answer")
            
            if 'transcript' in result:
                transcript = result['transcript']
                print(f"✓ Transcript: {len(transcript)} entries")
                
                # Show first few transcript entries
                for i, entry in enumerate(transcript[:3]):
                    if isinstance(entry, dict):
                        note = entry.get('note', str(entry)[:50])
                        print(f"  {i+1}. {note}")
            
            if 'retries_used' in result:
                print(f"✓ Retries used: {result['retries_used']}")
            
            if 'model_validation' in result:
                print(f"✓ Model validation: {result['model_validation']}")
            
            return True
        
        elif isinstance(result, str):
            print(f"✓ Got string result: {result[:200]}...")
            return True
        
        else:
            print(f"✓ Got result: {str(result)[:200]}...")
            return True
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_original_failing_case():
    """Test the original PLIP case"""
    print("\n" + "=" * 80)
    print("TESTING ORIGINAL FAILING CASE")
    print("=" * 80)
    
    # Set environment variables
    env_vars = {
        "LM_STUDIO_URL": "http://localhost:1234",
        "LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "MODEL_NAME": "openai/gpt-oss-20b",
        "LMSTUDIO_MODEL": "openai/gpt-oss-20b",
        "LMSTUDIO_FUNCTION_MODEL": "openai/gpt-oss-20b",
        "EXPOSE_PUBLIC_ONLY": "0",
        "HTTP_CONNECT_TIMEOUT": "2",
        "HTTP_READ_TIMEOUT_SIMPLE": "8",
        "HTTP_READ_TIMEOUT_COMPLEX": "45"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create server instance
    test_server = server.EnhancedLMStudioMCPServer()
    
    print("Testing original PLIP integration question...")
    original_args = {
        "instruction": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges).",
        "allowed_tools": [],
        "tool_choice": "auto",
        "max_iters": 4,
        "temperature": 0.2
    }
    
    try:
        result = server.handle_chat_with_tools(original_args, test_server)
        
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            
            # Check for the old error pattern
            if 'content' in result and 'PREVENTIVE MEASURES FAILED' in str(result.get('content', '')):
                print("✗ STILL GETTING OLD ERROR PATTERN!")
                print(f"Content: {result['content'][:300]}...")
                return False
            
            # Check for success indicators
            if 'final_response' in result:
                final_resp = str(result['final_response'])
                print(f"✓ Final response ({len(final_resp)} chars): {final_resp[:300]}...")
            
            if 'transcript' in result:
                transcript = result['transcript']
                print(f"✓ Transcript: {len(transcript)} entries")
                
                # Show key transcript entries
                for i, entry in enumerate(transcript[:5]):
                    if isinstance(entry, dict):
                        note = entry.get('note', str(entry)[:80])
                        print(f"  {i+1}. {note}")
            
            if 'retries_used' in result:
                print(f"✓ Retries used: {result['retries_used']}")
            
            return True
        
        else:
            print(f"✓ Got non-dict result: {str(result)[:300]}...")
            return True
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct chat_with_tools tests"""
    print("Direct chat_with_tools Test Suite")
    print("=" * 80)
    
    tests = [
        test_direct_chat_with_tools,
        test_original_failing_case
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("DIRECT TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_func.__name__}")
    
    if passed == total:
        print("\n🎉 Direct chat_with_tools tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
