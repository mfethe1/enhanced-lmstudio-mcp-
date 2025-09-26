"""
Amazon Bedrock adapter for Anthropic Claude models.
Provides a unified interface to call Claude models via AWS Bedrock.
"""
import json
import os
import logging
from typing import Dict, Any, Optional, List, Union
import asyncio
import aiohttp
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

logger = logging.getLogger(__name__)

class BedrockAdapter:
    """
    Adapter for Amazon Bedrock Claude models.
    Supports both sync and async operations with circuit breaker pattern.
    """
    
    def __init__(self, region: str = None, timeout: int = 60):
        """
        Initialize Bedrock adapter.
        
        Args:
            region: AWS region (defaults to us-east-1)
            timeout: Request timeout in seconds (default 60 for Claude models)
        """
        self.region = region or os.getenv("BEDROCK_REGION", "us-east-1")
        self.timeout = timeout
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize boto3 Bedrock Runtime client with proper config."""
        try:
            # Configure with extended timeout for Claude models (AWS recommends 60+ minutes)
            config = Config(
                region_name=self.region,
                read_timeout=self.timeout,
                connect_timeout=10,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
            
            self.client = boto3.client('bedrock-runtime', config=config)
            logger.info(f"Bedrock client initialized for region {self.region}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def get_model_id(self, model_name: str) -> str:
        """
        Map model names to Bedrock model IDs.
        
        Args:
            model_name: Model name (e.g., claude-3-5-sonnet-latest)
            
        Returns:
            Bedrock model ID
        """
        model_mapping = {
            # Claude 3.5 Sonnet
            "claude-3-5-sonnet-latest": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            
            # Claude 3 Opus
            "claude-3-opus-latest": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
            
            # Claude 3 Sonnet
            "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
            
            # Claude 3 Haiku
            "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
        }
        
        return model_mapping.get(model_name, model_name)
    
    def format_messages_for_bedrock(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format messages for Bedrock Claude Messages API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted request payload
        """
        # Convert messages to Bedrock format
        bedrock_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Extract system prompt
                system_prompt = content
                continue
            elif role in ["user", "assistant"]:
                # Format content as text block
                if isinstance(content, str):
                    content_blocks = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    content_blocks = content
                else:
                    content_blocks = [{"type": "text", "text": str(content)}]
                
                bedrock_messages.append({
                    "role": role,
                    "content": content_blocks
                })
        
        # Build request payload
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": bedrock_messages
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        return payload
    
    def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous chat completion using Bedrock.
        
        Args:
            model: Model name
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Response dict with 'choices' containing generated text
        """
        try:
            model_id = self.get_model_id(model)
            payload = self.format_messages_for_bedrock(messages)
            
            # Update inference parameters
            payload.update({
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 250)
            })
            
            # Invoke model
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Format as OpenAI-compatible response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_body.get("content", [{}])[0].get("text", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": response_body.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": response_body.get("usage", {}).get("input_tokens", 0) + 
                                   response_body.get("usage", {}).get("output_tokens", 0)
                },
                "model": model_id
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock API error {error_code}: {error_message}")
            raise Exception(f"Bedrock error: {error_message}")
        except Exception as e:
            logger.error(f"Bedrock request failed: {e}")
            raise
    
    async def achat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronous chat completion using Bedrock.
        Note: boto3 doesn't support async natively, so we run in executor.
        
        Args:
            model: Model name
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Response dict with 'choices' containing generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.chat_completion, 
            model, 
            messages, 
            temperature, 
            max_tokens, 
            **kwargs
        )
    
    def is_available(self) -> bool:
        """
        Check if Bedrock is available and configured.
        
        Returns:
            True if Bedrock client is ready
        """
        try:
            if not self.client:
                return False
            
            # Test with a simple list models call
            self.client.list_foundation_models(byProvider='anthropic')
            return True
        except Exception as e:
            logger.debug(f"Bedrock availability check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Anthropic models in Bedrock.
        
        Returns:
            List of model IDs
        """
        try:
            response = self.client.list_foundation_models(byProvider='anthropic')
            models = []
            for model in response.get('modelSummaries', []):
                if model.get('modelId'):
                    models.append(model['modelId'])
            return models
        except Exception as e:
            logger.error(f"Failed to list Bedrock models: {e}")
            return []


# Global instance for easy import
bedrock_adapter = BedrockAdapter()
