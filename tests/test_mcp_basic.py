import json
import re
import types

import pytest

import server as mcp


def test_text_only_response_envelope():
    # Simulate a simple tool call result
    result = {"a": 1, "b": 2}
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}
    # Validate envelope
    assert "result" in payload
    content = payload["result"]["content"]
    assert isinstance(content, list) and content
    item = content[0]
    assert item.get("type") == "text"
    assert isinstance(item.get("text"), str)


def test_sanitizer_extracts_message_content_json():
    # Chat completion-like JSON leaked as text
    fake = json.dumps({
        "id": "x",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
    })
    out = mcp._sanitize_llm_output(fake)
    assert out == "Hello"


def test_sanitizer_strips_verbose_prefix():
    txt = "[openai/gpt-oss-20b] Generated prediction:\n{\n  \"choices\": [{\"message\": {\"content\": \"OK\"}}]\n}"
    out = mcp._sanitize_llm_output(txt)
    assert out == "OK"


def test_extract_code_from_text():
    text = """
Here are tests:
```python
import pytest

def test_add():
    assert 1+1 == 2
```
    """
    code = mcp._extract_code_from_text(text)
    assert "def test_add" in code


@pytest.mark.parametrize("input_text, expected_contains", [
    ("""no fences\nimport pytest\ndef test_x():\n    assert True\n""", "def test_x"),
    ("""```\ncode\n```""", "code"),
])
def test_extract_code_from_text_fallbacks(input_text, expected_contains):
    code = mcp._extract_code_from_text(input_text)
    assert expected_contains in code

