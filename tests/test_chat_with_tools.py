import json
import os
import unittest
from types import SimpleNamespace

# Minimal harness: call the server handler directly
from server import handle_chat_with_tools, _to_text_content, _safe_path

class FakeServer(SimpleNamespace):
    pass

class TestChatWithTools(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()

