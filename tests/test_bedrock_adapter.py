"""
Tests for Bedrock adapter functionality.
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Import the adapter
try:
    from bedrock_adapter import BedrockAdapter, bedrock_adapter
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    BedrockAdapter = None
    bedrock_adapter = None


@pytest.mark.skipif(not BEDROCK_AVAILABLE, reason="Bedrock adapter not available")
class TestBedrockAdapter:
    """Test suite for Bedrock adapter."""
    
    def test_model_id_mapping(self):
        """Test model name to Bedrock ID mapping."""
        adapter = BedrockAdapter()
        
        # Test known mappings
        assert adapter.get_model_id("claude-3-5-sonnet-latest") == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert adapter.get_model_id("claude-3-opus-latest") == "anthropic.claude-3-opus-20240229-v1:0"
        
        # Test passthrough for unknown models
        assert adapter.get_model_id("unknown-model") == "unknown-model"
    
    def test_message_formatting(self):
        """Test message formatting for Bedrock API."""
        adapter = BedrockAdapter()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = adapter.format_messages_for_bedrock(messages)
        
        # Check structure
        assert "anthropic_version" in formatted
        assert "max_tokens" in formatted
        assert "messages" in formatted
        assert "system" in formatted
        
        # Check system prompt extraction
        assert formatted["system"] == "You are a helpful assistant."
        
        # Check message conversion
        bedrock_messages = formatted["messages"]
        assert len(bedrock_messages) == 3  # System message excluded
        assert bedrock_messages[0]["role"] == "user"
        assert bedrock_messages[0]["content"][0]["text"] == "Hello"
    
    @patch('boto3.client')
    def test_sync_chat_completion(self, mock_boto_client):
        """Test synchronous chat completion."""
        # Mock Bedrock response
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps({
            "content": [{"text": "Hello! How can I help you?"}],
            "usage": {"input_tokens": 10, "output_tokens": 8}
        }).encode()
        
        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        adapter = BedrockAdapter()
        adapter.client = mock_client
        
        messages = [{"role": "user", "content": "Hello"}]
        response = adapter.chat_completion("claude-3-5-sonnet-latest", messages)
        
        # Verify response format
        assert "choices" in response
        assert "usage" in response
        assert "model" in response
        assert response["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 8
    
    @patch('boto3.client')
    @pytest.mark.asyncio
    async def test_async_chat_completion(self, mock_boto_client):
        """Test asynchronous chat completion."""
        # Mock Bedrock response
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps({
            "content": [{"text": "Async response"}],
            "usage": {"input_tokens": 5, "output_tokens": 3}
        }).encode()
        
        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client
        
        adapter = BedrockAdapter()
        adapter.client = mock_client
        
        messages = [{"role": "user", "content": "Test async"}]
        response = await adapter.achat_completion("claude-3-opus-latest", messages)
        
        # Verify response format
        assert "choices" in response
        assert response["choices"][0]["message"]["content"] == "Async response"
    
    @patch('boto3.client')
    def test_availability_check(self, mock_boto_client):
        """Test availability checking."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_boto_client.return_value = mock_client
        
        adapter = BedrockAdapter()
        adapter.client = mock_client
        
        assert adapter.is_available() == True
        
        # Test failure case
        mock_client.list_foundation_models.side_effect = Exception("AWS error")
        assert adapter.is_available() == False
    
    @patch('boto3.client')
    def test_get_available_models(self, mock_boto_client):
        """Test getting available models."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
                {"modelId": "anthropic.claude-3-opus-20240229-v1:0"}
            ]
        }
        mock_boto_client.return_value = mock_client
        
        adapter = BedrockAdapter()
        adapter.client = mock_client
        
        models = adapter.get_available_models()
        assert len(models) == 2
        assert "anthropic.claude-3-5-sonnet-20241022-v2:0" in models
        assert "anthropic.claude-3-opus-20240229-v1:0" in models
    
    def test_initialization_without_credentials(self):
        """Test initialization without AWS credentials."""
        with patch('boto3.client') as mock_boto:
            from botocore.exceptions import NoCredentialsError
            mock_boto.side_effect = NoCredentialsError()
            
            with pytest.raises(NoCredentialsError):
                BedrockAdapter()


@pytest.mark.skipif(not BEDROCK_AVAILABLE, reason="Bedrock adapter not available")
class TestBedrockIntegration:
    """Integration tests for Bedrock with server."""
    
    @patch.dict(os.environ, {"USE_BEDROCK": "1", "BEDROCK_REGION": "us-east-1"})
    def test_bedrock_environment_config(self):
        """Test Bedrock environment configuration."""
        assert os.getenv("USE_BEDROCK") == "1"
        assert os.getenv("BEDROCK_REGION") == "us-east-1"
    
    def test_global_adapter_instance(self):
        """Test global bedrock_adapter instance."""
        if BEDROCK_AVAILABLE:
            assert bedrock_adapter is not None
            assert isinstance(bedrock_adapter, BedrockAdapter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
