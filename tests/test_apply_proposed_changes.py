import os
from pathlib import Path

import server as s


def test_apply_proposed_changes_supports_file_and_path_headers(tmp_path, monkeypatch):
    # Ensure operations happen in tmp cwd by chdir
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        text = (
            "Some plan text\n\n"
            "```python\n# File: a.txt\nHello A\n```\n\n"
            "```txt\n# path: b.txt\nHello B\n```\n"
        )
        applied = s._apply_proposed_changes(text, dry_run=False)
        assert any("a.txt" in line for line in applied)
        assert any("b.txt" in line for line in applied)
        assert Path("a.txt").exists()
        assert Path("b.txt").exists()
    finally:
        # cleanup and restore cwd
        for p in ("a.txt", "b.txt"):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        os.chdir(cwd)

