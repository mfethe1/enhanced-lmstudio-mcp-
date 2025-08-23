from typing import Optional


def normalize_whitespace(text: str, max_chars: Optional[int] = None) -> str:
    """
    Normalize whitespace in a text string.

    Steps:
    - Normalize all newline variations (\r\n, \r) to "\n".
    - Remove duplicate consecutive lines (keep the first occurrence in a run).
    - Collapse runs of blank lines to a single blank line (at most one empty line between non-empty lines).
    - Optionally truncate the result to at most `max_chars` characters.

    This function is deterministic and idempotent (running it multiple times yields the same result).

    Args:
        text: Input text to normalize.
        max_chars: If provided, truncate the normalized output to at most this many characters.

    Returns:
        A normalized string.
    """
    if not isinstance(text, str):
        # Be explicit to avoid surprising behavior
        raise TypeError("normalize_whitespace expects a string input")

    # Normalize newlines
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into logical lines (without retaining separators)
    lines = s.split("\n")

    # Remove duplicate consecutive lines and collapse blank-line runs
    normalized_lines = []
    last_line: Optional[str] = None
    blank_run = 0

    for line in lines:
        is_blank = len(line.strip()) == 0

        # Collapse blank line runs to at most one
        if is_blank:
            blank_run += 1
            if blank_run > 1:
                # Skip extra blanks
                continue
        else:
            blank_run = 0

        # Remove exact duplicate consecutive lines
        if last_line is not None and line == last_line:
            # Skip duplicates
            continue

        normalized_lines.append(line)
        last_line = line

    result = "\n".join(normalized_lines)

    # Optional truncation
    if isinstance(max_chars, int) and max_chars >= 0:
        if len(result) > max_chars:
            result = result[:max_chars]

    return result

