import pytest

from utils import normalize_whitespace


def test_normalize_basic_newlines():
    s = "a\r\nb\rc\nd"
    assert normalize_whitespace(s) == "a\nb\nc\nd"


def test_collapse_blank_lines():
    s = "line1\n\n\nline2\n\n\n\nline3\n\n"
    assert normalize_whitespace(s) == "line1\n\nline2\n\nline3\n"


def test_remove_consecutive_duplicates():
    s = "same\nsame\nother\nother\nother\nend"
    assert normalize_whitespace(s) == "same\nother\nend"


def test_idempotent_behavior():
    s = "A\n\n\nA\nA\n\nB\n\n\n\nB\n"
    once = normalize_whitespace(s)
    twice = normalize_whitespace(once)
    assert once == twice


def test_truncation():
    s = "x" * 50
    out = normalize_whitespace(s, max_chars=10)
    assert out == "x" * 10


def test_type_error():
    with pytest.raises(TypeError):
        normalize_whitespace(123)  # type: ignore[arg-type]

