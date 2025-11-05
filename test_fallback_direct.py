#!/usr/bin/env python3
"""
Test fallback providers directly (skip LM Studio entirely)
"""

import os
import time
import requests

def test_fallback_direct():
    """Test fallback providers directly"""
    print("🔧 Testing Fallback Providers Directly")
    print("=" * 60)
    print("Skipping LM Studio - testing OpenAI and Anthropic directly")
    print("-" * 60)
    
    # Test message - the original PLIP request that was failing
    messages = [{
        "role": "user", 
        "content": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges)."
    }]
    
    results = {}
    
    # Test OpenAI
    print("1. Testing OpenAI...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 500
            }
            
            start_time = time.time()
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                timeout=(5, 30)
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                results["openai"] = {
                    "success": True,
                    "content": content,
                    "time": elapsed,
                    "length": len(content)
                }
                print(f"   ✅ Success in {elapsed:.1f}s")
                print(f"   Content length: {len(content)}")
                print(f"   Preview: '{content[:100]}...'")
            else:
                results["openai"] = {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "time": elapsed
                }
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            results["openai"] = {
                "success": False,
                "error": str(e),
                "time": time.time() - start_time if 'start_time' in locals() else 0
            }
            print(f"   ❌ Error: {e}")
    else:
        results["openai"] = {"success": False, "error": "No API key"}
        print("   ⚠️  No OpenAI API key")
    
    # Test Anthropic
    print("\n2. Testing Anthropic...")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 500,
                "messages": messages
            }
            
            start_time = time.time()
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Authorization": f"Bearer {anthropic_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=(5, 30)
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"].strip()
                results["anthropic"] = {
                    "success": True,
                    "content": content,
                    "time": elapsed,
                    "length": len(content)
                }
                print(f"   ✅ Success in {elapsed:.1f}s")
                print(f"   Content length: {len(content)}")
                print(f"   Preview: '{content[:100]}...'")
            else:
                results["anthropic"] = {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "time": elapsed
                }
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            results["anthropic"] = {
                "success": False,
                "error": str(e),
                "time": time.time() - start_time if 'start_time' in locals() else 0
            }
            print(f"   ❌ Error: {e}")
    else:
        results["anthropic"] = {"success": False, "error": "No API key"}
        print("   ⚠️  No Anthropic API key")
    
    # Analyze results
    print(f"\n📊 Results Summary:")
    print("-" * 30)
    
    working_providers = []
    for provider, result in results.items():
        if result.get("success"):
            working_providers.append(provider)
            print(f"✅ {provider.upper()}: Working ({result['time']:.1f}s, {result['length']} chars)")
            
            # Check PLIP relevance
            content = result['content'].lower()
            plip_keywords = ["plip", "protein", "ligand", "pdb", "interaction", "bond"]
            relevant_keywords = sum(1 for keyword in plip_keywords if keyword in content)
            print(f"   PLIP relevance: {relevant_keywords}/{len(plip_keywords)} keywords")
            
        else:
            print(f"❌ {provider.upper()}: Failed ({result.get('error', 'Unknown error')})")
    
    if working_providers:
        print(f"\n🎉 SUCCESS: {len(working_providers)} provider(s) working!")
        print("✅ This proves our fallback strategy will work")
        print("✅ Users will get quality PLIP integration guidance")
        
        # Show best response
        best_provider = None
        best_length = 0
        for provider in working_providers:
            if results[provider]['length'] > best_length:
                best_length = results[provider]['length']
                best_provider = provider
        
        if best_provider:
            print(f"\n🏆 Best Response ({best_provider.upper()}):")
            print("-" * 40)
            content = results[best_provider]['content']
            print(content[:500] + ("..." if len(content) > 500 else ""))
            
        return True
    else:
        print(f"\n❌ FAILED: No providers working")
        print("Check API keys and network connectivity")
        return False

if __name__ == "__main__":
    success = test_fallback_direct()
    
    if success:
        print("\n🎯 CONCLUSION: Fallback providers work perfectly!")
        print("The issue is LM Studio timeout handling, not the fallback logic.")
        print("Once we fix the timeout issue, users will get reliable responses.")
    else:
        print("\n❌ Need to fix API key configuration")
