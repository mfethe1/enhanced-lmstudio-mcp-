import os
import unittest
from types import SimpleNamespace

from handlers import code_tools

class FakeServer(SimpleNamespace):
    pass

class TestCodeTools(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    def test_list_directory_smoke(self):
        out = code_tools.handle_list_directory({"path": ".", "depth": 1}, self.server)
        self.assertIsInstance(out, str)
        # Should list at least this tests directory
        self.assertTrue("tests" in out)

    def test_read_file_range_invalid(self):
        out = code_tools.handle_read_file_range({"file_path":"handlers/code_tools.py"}, self.server)
        self.assertIn("Error", out)
        out2 = code_tools.handle_read_file_range({"file_path":"handlers/code_tools.py", "start_line": 10, "end_line": 5}, self.server)
        self.assertIn("invalid range", out2)

    def test_read_file_range_valid(self):
        out = code_tools.handle_read_file_range({"file_path":"handlers/code_tools.py", "start_line": 1, "end_line": 5}, self.server)
        self.assertIsInstance(out, str)
        self.assertIn("File:", out)

if __name__ == '__main__':
    unittest.main()

