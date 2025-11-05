#!/usr/bin/env python3
"""
Fix the MCP server to properly support reasoning models like openai/gpt-oss-20b
"""
import json

def create_reasoning_model_patch():
    """Create a patch for server.py to handle reasoning models"""
    
    patch_code = '''
def _extract_content_from_response(data, model_name=""):
    """
    Extract content from LM Studio response, handling reasoning models.
    
    Reasoning models (like openai/gpt-oss-20b) put their thinking in 'reasoning' 
    and final answer in 'content'. If content is empty, we use reasoning.
    """
    choices = data.get('choices', [])
    if not choices:
        return ""
    
    choice = choices[0]
    message = choice.get('message', {})
    
    # Get content and reasoning
    content = message.get('content', '').strip()
    reasoning = message.get('reasoning', '').strip()
    
    # For reasoning models, prefer content but fallback to reasoning
    if content:
        return content
    elif reasoning:
        # For reasoning models, we can return the reasoning as the response
        # or format it nicely
        if 'gpt-oss' in model_name.lower() or 'reasoning' in model_name.lower():
            return f"[Reasoning]: {reasoning}"
        return reasoning
    
    # Fallback to text field for completions endpoint
    text = choice.get('text', '').strip()
    if text:
        # Clean up special tokens for reasoning models
        if '<|channel|>' in text:
            # Extract the actual content after special tokens
            parts = text.split('<|message|>')
            if len(parts) > 1:
                return parts[-1].strip()
        return text
    
    return ""

def _is_reasoning_model(model_name):
    """Check if a model is a reasoning model that needs special handling"""
    reasoning_indicators = [
        'gpt-oss', 'reasoning', 'thinking', 'o1-', 'chain-of-thought'
    ]
    model_lower = model_name.lower()
    return any(indicator in model_lower for indicator in reasoning_indicators)

def _make_reasoning_model_request(server, model, messages, temperature=0.1, max_tokens=None):
    """
    Make a request optimized for reasoning models.
    
    For reasoning models:
    1. Use higher max_tokens to allow for reasoning
    2. Use lower temperature for more consistent reasoning
    3. Handle both chat/completions and completions endpoints
    """
    import requests
    import time
    
    # Optimize parameters for reasoning models
    if _is_reasoning_model(model):
        max_tokens = max_tokens or 500  # More tokens for reasoning
        temperature = min(temperature, 0.3)  # Lower temp for reasoning
    else:
        max_tokens = max_tokens or 150
    
    # Try chat/completions first
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            f"{server.base_url}/v1/chat/completions",
            json=payload,
            timeout=120  # Longer timeout for reasoning
        )
        
        if response.status_code == 200:
            data = response.json()
            content = _extract_content_from_response(data, model)
            if content:
                return content
    
    except Exception as e:
        print(f"Chat completions failed: {e}")
    
    # Fallback to completions endpoint for reasoning models
    if _is_reasoning_model(model) and len(messages) == 1 and messages[0].get('role') == 'user':
        try:
            completions_payload = {
                "model": model,
                "prompt": messages[0]['content'],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{server.base_url}/v1/completions",
                json=completions_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                content = _extract_content_from_response(data, model)
                if content:
                    return content
                    
        except Exception as e:
            print(f"Completions fallback failed: {e}")
    
    return "No response generated"
'''
    
    return patch_code

def create_test_reasoning_model():
    """Create a test script for the reasoning model support"""
    
    test_code = '''#!/usr/bin/env python3
"""
Test reasoning model support
"""
import sys
import os
sys.path.append('.')

from server import EnhancedLMStudioMCPServer

def test_reasoning_model():
    print("🧪 Testing Reasoning Model Support")
    print("=" * 50)
    
    server = EnhancedLMStudioMCPServer()
    
    # Test with openai/gpt-oss-20b
    test_prompts = [
        "What is 2+2?",
        "Explain photosynthesis in one sentence",
        "Say hello",
        "Complete this: The capital of France is"
    ]
    
    for prompt in test_prompts:
        print(f"\\n🔧 Testing: '{prompt}'")
        
        try:
            # Use the new reasoning model request function
            messages = [{"role": "user", "content": prompt}]
            response = _make_reasoning_model_request(
                server, 
                "openai/gpt-oss-20b", 
                messages,
                temperature=0.1,
                max_tokens=100
            )
            
            if response and response.strip():
                print(f"✅ Response: {response[:100]}...")
            else:
                print("❌ Empty response")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_reasoning_model()
'''
    
    return test_code

def main():
    print("🔧 Creating Reasoning Model Support Patch")
    print("=" * 50)
    
    # Create the patch code
    patch = create_reasoning_model_patch()
    
    # Save patch to file
    with open('reasoning_model_patch.py', 'w') as f:
        f.write(patch)
    
    print("✅ Created reasoning_model_patch.py")
    
    # Create test script
    test = create_test_reasoning_model()
    with open('test_reasoning_support.py', 'w') as f:
        f.write(test)
    
    print("✅ Created test_reasoning_support.py")
    
    print("\n🎯 Next Steps:")
    print("1. Review reasoning_model_patch.py")
    print("2. Integrate the functions into server.py")
    print("3. Update the LM Studio request functions to use the new logic")
    print("4. Test with: python test_reasoning_support.py")
    
    print("\n💡 Key Improvements:")
    print("• Handles reasoning models that use 'reasoning' field")
    print("• Falls back to completions endpoint when needed")
    print("• Optimizes parameters for reasoning models")
    print("• Cleans up special tokens in responses")
    print("• Increases timeouts for reasoning models")

if __name__ == "__main__":
    main()
