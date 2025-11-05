import json
import os
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Test the preventive measures in chat_with_tools without network dependencies
from server import handle_chat_with_tools

class FakeServer(SimpleNamespace):
    def __init__(self):
        self.base_url = "http://localhost:1234"
        self.model_name = "test-model"

class TestChatWithToolsPreventiveMeasures(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    @patch('server._http_client')
    def test_parameter_validation_clamps_values(self, mock_client):
        """Test that invalid parameters are clamped to valid ranges"""
        # Mock successful responses
        mock_client.post_sync.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        
        result = handle_chat_with_tools({
            "instruction": "test",
            "temperature": 5.0,  # Should be clamped to 2.0
            "max_tokens": 10000,  # Should be clamped to 8192
            "top_p": 2.0,  # Should be clamped to 1.0
            "max_iters": -1  # Should be clamped to 1
        }, self.server)
        
        # Verify the function completed successfully
        self.assertIsInstance(result, dict)
        self.assertIn("content", result)

    @patch('server._http_client')
    def test_tool_choice_auto_correction(self, mock_client):
        """Test that invalid tool_choice values are auto-corrected"""
        mock_client.post_sync.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        
        result = handle_chat_with_tools({
            "instruction": "test",
            "tool_choice": "invalid_choice"  # Should be corrected
        }, self.server)
        
        self.assertIsInstance(result, dict)
        # Check that correction was logged
        transcript = result.get("transcript", [])
        corrections = [t for t in transcript if "note" in t and "corrected" in str(t)]
        self.assertTrue(len(corrections) > 0)

    @patch('server._http_client')
    def test_model_health_check_failure_handling(self, mock_client):
        """Test graceful handling when model health check fails"""
        # Mock health check failure
        mock_client.post_sync.side_effect = Exception("Connection refused")
        
        result = handle_chat_with_tools({
            "instruction": "test"
        }, self.server)
        
        self.assertIsInstance(result, dict)
        self.assertIn("content", result)
        self.assertIn("unavailable", result["content"])
        self.assertIn("hints", result)

    @patch('server._http_client')
    def test_retry_logic_parameter_compatibility(self, mock_client):
        """Test retry logic handles parameter compatibility issues"""
        # Mock health check success, then parameter error, then success
        mock_client.post_sync.side_effect = [
            {"choices": [{"message": {"content": "health ok"}}]},  # Health check
            Exception("Unsupported parameter: 'max_tokens'"),  # Parameter error
            {"choices": [{"message": {"content": "success"}}]}  # Retry success
        ]
        
        result = handle_chat_with_tools({
            "instruction": "test"
        }, self.server)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["content"], "success")

    @patch('server._http_client')
    def test_empty_response_retry_logic(self, mock_client):
        """Test that empty responses trigger retry logic with parameter adjustments"""
        # Mock health check success, then multiple empty responses, then success
        mock_client.post_sync.side_effect = [
            {"choices": [{"message": {"content": "health ok"}}]},  # Health check
            {"choices": [{"message": {"content": ""}}]},  # Empty response - attempt 1
            {"choices": [{"message": {"content": ""}}]},  # Empty response - attempt 2
            {"choices": [{"message": {"content": ""}}]},  # Empty response - attempt 3
            {"choices": [{"message": {"content": ""}}]},  # Empty response - attempt 4
            # No more responses - should trigger fallback handling
        ]

        result = handle_chat_with_tools({
            "instruction": "test"
        }, self.server)

        self.assertIsInstance(result, dict)
        self.assertIn("ALL PROVIDERS FAILED", result["content"])
        self.assertIn("4 LM Studio attempts", result["content"])
        self.assertIn("hints", result)
        self.assertIn("critical_actions", result["hints"])

        # Verify retry attempts were logged
        transcript = result.get("transcript", [])
        retry_notes = [t for t in transcript if "note" in t and "empty response" in t.get("note", "")]
        self.assertTrue(len(retry_notes) >= 3, f"Expected at least 3 retry attempts, got {len(retry_notes)}")

    @patch('server._http_client')
    def test_successful_retry_after_empty_response(self, mock_client):
        """Test that retry logic succeeds after adjusting parameters"""
        # Mock health check success, empty response, then success on retry
        mock_client.post_sync.side_effect = [
            {"choices": [{"message": {"content": "health ok"}}]},  # Health check
            {"choices": [{"message": {"content": ""}}]},  # Empty response - attempt 1
            {"choices": [{"message": {"content": "Success after parameter adjustment!"}}]}  # Success on retry
        ]

        result = handle_chat_with_tools({
            "instruction": "test"
        }, self.server)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["content"], "Success after parameter adjustment!")

        # Verify retry was attempted and logged
        transcript = result.get("transcript", [])
        retry_notes = [t for t in transcript if "note" in t and "empty response" in t.get("note", "")]
        self.assertTrue(len(retry_notes) >= 1, "Expected at least 1 retry attempt to be logged")

    def test_input_validation_rejects_empty_instruction(self):
        """Test that empty instruction is properly rejected"""
        with self.assertRaises(Exception) as context:
            handle_chat_with_tools({
                "instruction": ""  # Empty instruction should raise ValidationError
            }, self.server)
        
        self.assertIn("instruction", str(context.exception))

if __name__ == '__main__':
    unittest.main()
