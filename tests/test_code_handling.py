"""Tests for code-aware extraction in verra.ingest.extractors."""

from __future__ import annotations

from verra.ingest.extractors import extract_code_blocks, summarize_code_block


# ---------------------------------------------------------------------------
# extract_code_blocks
# ---------------------------------------------------------------------------


def test_extract_fenced_code_block():
    text = """\
Here is a simple example:

```python
def hello():
    print("hello world")
```

And some more prose after.
"""
    prose, blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    block = blocks[0]
    assert block["language"] == "python"
    assert "def hello" in block["code"]
    # Prose should not contain the raw code
    assert "def hello" not in prose
    assert "And some more prose" in prose


def test_extract_fenced_code_block_no_language():
    text = "Before.\n\n```\nsome code here\n```\n\nAfter."
    prose, blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["language"] == ""
    assert "some code here" in blocks[0]["code"]


def test_extract_indented_code_block():
    text = "Some explanation:\n\n    x = 1\n    y = 2\n    print(x + y)\n\nMore text."
    prose, blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "x = 1" in blocks[0]["code"]
    assert "More text" in prose


def test_no_code_in_plain_text():
    text = (
        "This is a completely normal paragraph with no code blocks at all. "
        "It mentions technical terms like deployment and pipeline but has no actual code."
    )
    prose, blocks = extract_code_blocks(text)
    assert blocks == []
    assert prose  # prose preserved


def test_multiple_code_blocks():
    text = """\
First block:

```python
def foo(): pass
```

Some text in between.

```javascript
const bar = () => {};
```

End.
"""
    prose, blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    languages = {b["language"] for b in blocks}
    assert "python" in languages
    assert "javascript" in languages
    assert "First block" in prose
    assert "End" in prose


def test_code_block_context_captured():
    text = "Refer to this migration script:\n\n```sql\nSELECT * FROM users;\n```\n"
    _, blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "migration script" in blocks[0]["context"]


# ---------------------------------------------------------------------------
# summarize_code_block
# ---------------------------------------------------------------------------


def test_code_block_summarization_python():
    code = """\
import os
import sys

def read_file(path):
    # Read the file contents
    with open(path) as f:
        return f.read()

class FileReader:
    pass
"""
    summary = summarize_code_block(code, language="python")
    assert "python" in summary.lower()
    assert "read_file" in summary
    assert "FileReader" in summary
    assert "os" in summary or "sys" in summary


def test_code_block_summarization_no_language():
    code = "x = 42\n# answer to everything\nprint(x)"
    summary = summarize_code_block(code)
    assert len(summary) > 0
    assert "answer to everything" in summary or "x" in summary


def test_summarization_empty_code():
    # Should not raise; returns a non-empty fallback
    summary = summarize_code_block("", language="")
    assert isinstance(summary, str)
