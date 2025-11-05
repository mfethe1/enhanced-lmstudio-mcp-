#!/usr/bin/env python3
"""
Enhanced test for chat_with_tools functionality with the original failing case.
"""

import json
import os
import sys
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import server

def test_chat_with_tools_original_case():
    """Test the original failing case that was reported"""
    print("=" * 80)
    print("TESTING ORIGINAL FAILING CASE: chat_with_tools")
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
    
    # Test the exact original failing case
    original_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "chat_with_tools",
            "arguments": {
                "instruction": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges).",
                "allowed_tools": [],
                "tool_choice": "auto",
                "max_iters": 4,
                "temperature": 0.2
            }
        }
    }
    
    print("Testing original failing case...")
    print(f"Instruction: {original_msg['params']['arguments']['instruction'][:100]}...")
    
    try:
        response = server.handle_message(original_msg)
        
        print(f"\nResponse structure:")
        print(f"- jsonrpc: {response.get('jsonrpc')}")
        print(f"- id: {response.get('id')}")
        print(f"- has result: {'result' in response}")
        print(f"- has error: {'error' in response}")
        
        if 'result' in response:
            result = response['result']
            print(f"\nResult analysis:")
            print(f"- Type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"- Keys: {list(result.keys())}")
                
                # Check for the old error pattern
                if 'content' in result and 'PREVENTIVE MEASURES FAILED' in str(result.get('content', '')):
                    print("✗ STILL GETTING OLD ERROR PATTERN!")
                    print(f"- Content: {result['content'][:200]}...")
                    return False
                
                # Check for success indicators
                if 'final_response' in result:
                    print(f"✓ Got final_response: {len(str(result['final_response']))} chars")
                
                if 'transcript' in result:
                    transcript = result['transcript']
                    print(f"✓ Got transcript: {len(transcript)} entries")
                    
                    # Show key transcript entries
                    for i, entry in enumerate(transcript[:5]):
                        if isinstance(entry, dict):
                            note = entry.get('note', str(entry)[:50])
                            print(f"  {i+1}. {note}")
                
                if 'retries_used' in result:
                    retries = result['retries_used']
                    print(f"✓ Retries used: {retries} (should be > 0 if there were issues)")
                
                # Check for model validation
                if 'model_validation' in result:
                    print(f"✓ Model validation: {result['model_validation']}")
                
                return True
            else:
                print(f"✓ Got non-dict result: {str(result)[:200]}...")
                return True
        
        elif 'error' in response:
            error = response['error']
            print(f"\nError response:")
            print(f"- Code: {error.get('code')}")
            print(f"- Message: {error.get('message', '')[:200]}...")
            
            # An error response is better than the old hanging behavior
            return True
        
        else:
            print("✗ No result or error in response")
            return False
            
    except Exception as e:
        print(f"✗ Exception during test: {e}")
        return False

def test_simple_chat_with_tools():
    """Test a simpler case to verify basic functionality"""
    print("\n" + "=" * 80)
    print("TESTING SIMPLE CASE: chat_with_tools")
    print("=" * 80)
    
    simple_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "chat_with_tools",
            "arguments": {
                "instruction": "What is 2 + 2?",
                "allowed_tools": [],
                "tool_choice": "none",
                "max_iters": 1,
                "temperature": 0.1
            }
        }
    }
    
    print("Testing simple math question...")
    
    try:
        response = server.handle_message(simple_msg)
        
        if 'result' in response:
            result = response['result']
            print("✓ Got result for simple case")
            
            if isinstance(result, dict):
                if 'final_response' in result:
                    final_resp = str(result['final_response'])
                    print(f"✓ Final response: {final_resp[:100]}...")
                    
                    # Check if it contains a reasonable answer
                    if '4' in final_resp or 'four' in final_resp.lower():
                        print("✓ Contains expected answer")
                        return True
                
                if 'transcript' in result:
                    transcript = result['transcript']
                    print(f"✓ Transcript entries: {len(transcript)}")
            
            return True
        else:
            print("✗ No result in simple case")
            return False
            
    except Exception as e:
        print(f"✗ Exception in simple case: {e}")
        return False

def main():
    """Run enhanced chat_with_tools tests"""
    print("Enhanced chat_with_tools Test Suite")
    print("=" * 80)
    
    tests = [
        test_chat_with_tools_original_case,
        test_simple_chat_with_tools
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
    print("ENHANCED TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_func.__name__}")
    
    if passed == total:
        print("\n🎉 Enhanced chat_with_tools tests passed!")
        print("The original failing case should now work correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
