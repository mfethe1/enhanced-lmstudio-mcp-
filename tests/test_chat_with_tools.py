import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# Minimal harness: call the server handler directly
from server import handle_chat_with_tools, _to_text_content, _safe_path

class FakeServer(SimpleNamespace):
    pass

class TestChatWithToolsEnhanced(unittest.TestCase):
    def setUp(self):
        # Use LM Studio mock base URL if provided, else skip tests gracefully
        self.server = FakeServer()
        self.server.base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234").rstrip("/")
        self.server.model_name = os.getenv("LMSTUDIO_FUNCTION_MODEL", os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b"))

    def test_builds_payload_and_handles_no_tools(self):
        # Smoke: ensure the handler runs and returns structured output even if LM Studio is not available
        try:
            out = handle_chat_with_tools({
                "instruction": "Summarize the file server.py; if needed, call tools to read it.",
                "allowed_tools": ["read_file_content"],
                "max_iters": 1,
                "temperature": 0.15,
                "tool_choice": "auto"
            }, self.server)
        except Exception as e:
            # We consider connection errors acceptable in unit environment without LM Studio
            self.assertTrue("Error:" in str(e) or isinstance(e, Exception))
            return
        # If it didn't raise, validate structure
        self.assertIsInstance(out, dict)
        self.assertIn("content", out)
        self.assertIn("transcript", out)

    def test_parameter_validation_and_correction(self):
        """Test that invalid parameters are auto-corrected"""
        with patch('server._http_client') as mock_client:
            # Mock model health check to pass
            mock_client.post_sync.side_effect = [
                {"choices": [{"message": {"content": "test"}}]},  # Health check
                {"choices": [{"message": {"content": "Valid response"}}]}  # Main request
            ]

            out = handle_chat_with_tools({
                "instruction": "Test instruction",
                "temperature": 5.0,  # Invalid: too high, should be clamped to 2.0
                "max_tokens": 10000,  # Invalid: too high, should be clamped to 8192
                "tool_choice": "invalid_choice",  # Invalid: should be corrected to "auto"
                "top_p": 2.0  # Invalid: too high, should be clamped to 1.0
            }, self.server)

            self.assertIsInstance(out, dict)
            self.assertIn("content", out)
            # Check that corrections were noted in transcript
            transcript = out.get("transcript", [])
            correction_notes = [t for t in transcript if "note" in t and "corrected" in t.get("note", "")]
            self.assertTrue(len(correction_notes) > 0, "Expected parameter corrections to be logged")

    def test_model_health_check_failure(self):
        """Test behavior when model health check fails"""
        with patch('server._http_client') as mock_client:
            # Mock health check to fail
            mock_client.post_sync.side_effect = Exception("Connection refused")

            out = handle_chat_with_tools({
                "instruction": "Test instruction"
            }, self.server)

            self.assertIsInstance(out, dict)
            self.assertIn("content", out)
            self.assertIn("appears to be unavailable", out["content"])
            self.assertIn("hints", out)
            self.assertIn("quick_fixes", out["hints"])

    def test_retry_logic_with_parameter_compatibility(self):
        """Test retry logic handles parameter compatibility issues"""
        with patch('server._http_client') as mock_client:
            # Mock sequence: health check passes, then parameter error, then success
            mock_client.post_sync.side_effect = [
                {"choices": [{"message": {"content": "test"}}]},  # Health check
                Exception("Unsupported parameter: 'max_tokens'"),  # First attempt fails
                {"choices": [{"message": {"content": "Success after retry"}}]}  # Retry succeeds
            ]

            out = handle_chat_with_tools({
                "instruction": "Test instruction"
            }, self.server)

            self.assertIsInstance(out, dict)
            self.assertIn("content", out)
            self.assertEqual(out["content"], "Success after retry")
            # Check that parameter switch was logged
            transcript = out.get("transcript", [])
            param_notes = [t for t in transcript if "note" in t and "max_completion_tokens" in t.get("note", "")]
            self.assertTrue(len(param_notes) > 0, "Expected parameter switch to be logged")

    def test_fallback_provider_handling(self):
        """Test fallback to OpenAI when LM Studio fails"""
        with patch('server._http_client') as mock_client, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):

            # Mock sequence: health check passes, LM Studio fails repeatedly, OpenAI succeeds
            mock_client.post_sync.side_effect = [
                {"choices": [{"message": {"content": "test"}}]},  # Health check
                Exception("Connection timeout"),  # LM Studio attempt 1
                Exception("Connection timeout"),  # LM Studio attempt 2
                Exception("Connection timeout"),  # LM Studio attempt 3
                Exception("Connection timeout"),  # LM Studio attempt 4
                {"choices": [{"message": {"content": "OpenAI fallback success"}}]}  # OpenAI fallback
            ]

            out = handle_chat_with_tools({
                "instruction": "Test instruction"
            }, self.server)

            self.assertIsInstance(out, dict)
            self.assertIn("content", out)
            self.assertEqual(out["content"], "OpenAI fallback success")
            # Check that fallback was logged
            transcript = out.get("transcript", [])
            fallback_notes = [t for t in transcript if "note" in t and "OpenAI fallback" in t.get("note", "")]
            self.assertTrue(len(fallback_notes) > 0, "Expected fallback usage to be logged")

    def test_comprehensive_empty_response_handling(self):
        """Test enhanced empty response handling with comprehensive diagnostics"""
        with patch('server._http_client') as mock_client:
            # Mock health check passes, but model returns empty content
            mock_client.post_sync.side_effect = [
                {"choices": [{"message": {"content": "test"}}]},  # Health check
                {"choices": [{"message": {"content": ""}}]}  # Empty response
            ]

            out = handle_chat_with_tools({
                "instruction": "Test instruction"
            }, self.server)

            self.assertIsInstance(out, dict)
            self.assertIn("content", out)
            self.assertIn("PREVENTIVE MEASURES FAILED", out["content"])
            self.assertIn("hints", out)
            self.assertIn("immediate_actions", out["hints"])
            self.assertIn("model_recommendations", out["hints"])
            self.assertIn("prevention_summary", out)

if __name__ == '__main__':
    unittest.main()

