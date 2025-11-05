#!/usr/bin/env python3
"""
Test a minimal robust request function with proper timeout handling
"""

import sys
import os
import time
import requests

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def minimal_robust_request(base_url, model, messages, temperature, max_tokens):
    """
    Minimal robust request with proper timeout and fallback
    """
    transcript = []
    
    # Step 1: Try LM Studio with short timeout
    print("Step 1: Trying LM Studio...")
    try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(2, 8)  # 2s connect, 8s read
        )
        
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    print(f"✅ LM Studio success: '{content[:50]}...'")
                    transcript.append({"note": "LM Studio success"})
                    return data, transcript
                else:
                    print("❌ LM Studio returned empty content")
                    transcript.append({"note": "LM Studio empty response"})
            else:
                print("❌ LM Studio no choices")
                transcript.append({"note": "LM Studio no choices"})
        else:
            print(f"❌ LM Studio HTTP {response.status_code}")
            transcript.append({"note": f"LM Studio HTTP {response.status_code}"})
            
    except requests.exceptions.Timeout:
        print("⏰ LM Studio timeout")
        transcript.append({"note": "LM Studio timeout"})
    except requests.exceptions.ConnectionError:
        print("🔌 LM Studio connection error")
        transcript.append({"note": "LM Studio connection error"})
    except Exception as e:
        print(f"❌ LM Studio error: {e}")
        transcript.append({"note": f"LM Studio error: {str(e)[:50]}"})
    
    # Step 2: Try OpenAI fallback
    print("Step 2: Trying OpenAI fallback...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                timeout=(5, 15)
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                print(f"✅ OpenAI success: '{content[:50]}...'")
                transcript.append({"note": "OpenAI fallback success"})
                return data, transcript
            else:
                print(f"❌ OpenAI HTTP {response.status_code}")
                transcript.append({"note": f"OpenAI HTTP {response.status_code}"})
                
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
            transcript.append({"note": f"OpenAI error: {str(e)[:50]}"})
    else:
        print("⚠️  No OpenAI API key")
        transcript.append({"note": "No OpenAI API key"})
    
    # Step 3: Try Anthropic fallback
    print("Step 3: Trying Anthropic fallback...")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": max_tokens,
                "messages": messages
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Authorization": f"Bearer {anthropic_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=(5, 15)
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"].strip()
                # Convert to OpenAI format
                openai_format = {
                    "choices": [{
                        "message": {"content": content}
                    }]
                }
                print(f"✅ Anthropic success: '{content[:50]}...'")
                transcript.append({"note": "Anthropic fallback success"})
                return openai_format, transcript
            else:
                print(f"❌ Anthropic HTTP {response.status_code}")
                transcript.append({"note": f"Anthropic HTTP {response.status_code}"})
                
        except Exception as e:
            print(f"❌ Anthropic error: {e}")
            transcript.append({"note": f"Anthropic error: {str(e)[:50]}"})
    else:
        print("⚠️  No Anthropic API key")
        transcript.append({"note": "No Anthropic API key"})
    
    # All providers failed
    print("❌ All providers failed")
    transcript.append({"note": "All providers failed"})
    return None, transcript

def test_minimal_robust():
    """Test the minimal robust request function"""
    print("🔧 Testing Minimal Robust Request Function")
    print("=" * 60)
    
    # Test parameters
    base_url = "http://localhost:1234"
    model = "openai/gpt-oss-20b"
    messages = [{"role": "user", "content": "PLIP integration planning: What is the minimal reliable pipeline to analyze protein–ligand interactions from a PDB in Python using PLIP? Outline API usage (PDBComplex, load_pdb, analyze), ligand selection strategies, and JSON output fields suitable for our artifacts (H-bonds, hydrophobics, salt bridges)."}]
    temperature = 0.2
    max_tokens = 500
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Request: PLIP integration planning")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        data, transcript = minimal_robust_request(
            base_url, model, messages, temperature, max_tokens
        )
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
        
        if data:
            content = data["choices"][0]["message"]["content"].strip()
            print(f"\n✅ SUCCESS!")
            print(f"Content length: {len(content)}")
            print(f"Content preview: '{content[:200]}{'...' if len(content) > 200 else ''}'")
            
            # Check PLIP relevance
            plip_keywords = ["plip", "protein", "ligand", "pdb", "interaction", "bond"]
            relevant_keywords = sum(1 for keyword in plip_keywords if keyword.lower() in content.lower())
            print(f"PLIP relevance: {relevant_keywords}/{len(plip_keywords)} keywords found")
            
        else:
            print(f"\n❌ FAILED: No response from any provider")
        
        print(f"\n📋 Transcript ({len(transcript)} entries):")
        for i, entry in enumerate(transcript):
            print(f"  {i+1}. {entry}")
            
        return data is not None
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_robust()
    
    if success:
        print("\n🎉 SUCCESS: Minimal robust request is working!")
        print("✅ This proves the concept - we can implement proper fallback logic")
        print("✅ Users will get meaningful responses instead of hanging/empty content")
    else:
        print("\n❌ FAILED: Need to check API keys and connectivity")
