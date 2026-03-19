"""Tests for content extractors."""

from __future__ import annotations

from pathlib import Path

import pytest

from verra.ingest.extractors import (
    ExtractedDocument,
    detect_and_extract,
    extract_csv,
    extract_text,
)


FIXTURES = Path(__file__).parent / "fixtures"


class TestExtractText:
    def test_txt_extraction(self) -> None:
        doc = extract_text(FIXTURES / "sample.txt")
        assert isinstance(doc, ExtractedDocument)
        assert "Q1 Planning" in doc.content
        assert doc.format == "txt"
        assert doc.page_count == 1

    def test_md_extraction(self) -> None:
        doc = extract_text(FIXTURES / "sample.md")
        assert "Refund Policy" in doc.content
        assert doc.format == "md"
        assert doc.metadata["file_name"] == "sample.md"

    def test_encoding_errors_dont_crash(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_bytes(b"Hello \xff world")
        doc = extract_text(bad)
        assert "Hello" in doc.content  # replacement char used, no exception


class TestExtractCsv:
    def test_csv_extraction(self) -> None:
        doc = extract_csv(FIXTURES / "sample.csv")
        assert "Jake Mitchell" in doc.content
        assert doc.format == "csv"
        assert doc.metadata["row_count"] == 5  # header + 4 rows

    def test_csv_metadata(self) -> None:
        doc = extract_csv(FIXTURES / "sample.csv")
        assert doc.metadata["file_name"] == "sample.csv"


class TestDetectAndExtract:
    def test_dispatches_txt(self) -> None:
        doc = detect_and_extract(FIXTURES / "sample.txt")
        assert doc.format == "txt"

    def test_dispatches_md(self) -> None:
        doc = detect_and_extract(FIXTURES / "sample.md")
        assert doc.format == "md"

    def test_dispatches_csv(self) -> None:
        doc = detect_and_extract(FIXTURES / "sample.csv")
        assert doc.format == "csv"

    def test_raises_on_unsupported(self, tmp_path: Path) -> None:
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            detect_and_extract(unsupported)

    def test_pdf_import_error_on_missing_dep(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If fitz is not importable, extract_pdf should raise ImportError."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "fitz":
                raise ImportError("No module named 'fitz'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from verra.ingest.extractors import extract_pdf
        dummy = tmp_path / "dummy.pdf"
        dummy.write_bytes(b"%PDF-1.4")
        with pytest.raises(ImportError, match="pymupdf"):
            extract_pdf(dummy)
