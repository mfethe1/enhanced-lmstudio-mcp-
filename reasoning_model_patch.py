
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
