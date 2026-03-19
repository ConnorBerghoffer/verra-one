"""Tests for graceful error handling in LLM client and extractors."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_client():
    """Import and construct an LLMClient with a dummy model."""
    from verra.agent.llm import LLMClient
    return LLMClient(model="openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# DOCX extraction warnings
# ---------------------------------------------------------------------------


class TestEmptyDocxWarns:
    """extract_docx() should log a warning when very little text is extracted."""

    def test_empty_docx_warns(self, tmp_path: Path, caplog) -> None:
        """A DOCX that yields fewer than 10 chars triggers a warning and sets extraction_warning."""
        from verra.ingest.extractors import extract_docx

        # Mock python-docx so we can control paragraph output without needing a real .docx
        fake_para = MagicMock()
        fake_para.text = ""  # empty paragraph

        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para]

        dummy_path = tmp_path / "empty.docx"
        dummy_path.touch()

        with patch("docx.Document", return_value=fake_doc, create=True):
            with caplog.at_level(logging.WARNING, logger="verra.ingest.extractors"):
                result = extract_docx(dummy_path)

        assert "extraction_warning" in result.metadata
        assert "empty" in result.metadata["extraction_warning"].lower() or \
               "image" in result.metadata["extraction_warning"].lower()

        # Warning should appear in logs
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("empty" in m.lower() or "image" in m.lower() or "very little" in m.lower()
                   for m in warning_messages), (
            f"Expected a warning about empty/image document, got: {warning_messages}"
        )

    def test_non_empty_docx_no_warning(self, tmp_path: Path, caplog) -> None:
        """A DOCX with meaningful content must NOT set extraction_warning."""
        from verra.ingest.extractors import extract_docx

        fake_para = MagicMock()
        fake_para.text = "This is a real paragraph with more than ten characters."

        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para]

        dummy_path = tmp_path / "real.docx"
        dummy_path.touch()

        with patch("docx.Document", return_value=fake_doc, create=True):
            result = extract_docx(dummy_path)

        assert "extraction_warning" not in result.metadata


# ---------------------------------------------------------------------------
# LLM client error messages
# ---------------------------------------------------------------------------


class TestLLMAuthError:
    """complete() raises RuntimeError with a helpful message on auth failure."""

    def test_llm_auth_error_message(self) -> None:
        client = _make_llm_client()

        # Simulate litellm.AuthenticationError
        import litellm

        fake_auth_error = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        with patch("litellm.completion", side_effect=fake_auth_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.complete([{"role": "user", "content": "hello"}])

        error_msg = str(exc_info.value).lower()
        assert "api key" in error_msg, (
            f"Expected 'api key' in error message, got: {exc_info.value}"
        )


class TestLLMConnectionError:
    """complete() raises RuntimeError with a helpful message on connection failure."""

    def test_llm_connection_error_message(self) -> None:
        client = _make_llm_client()

        import litellm

        fake_conn_error = litellm.APIConnectionError(
            message="Connection refused",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        with patch("litellm.completion", side_effect=fake_conn_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.complete([{"role": "user", "content": "hello"}])

        error_msg = str(exc_info.value).lower()
        assert "connection" in error_msg or "internet" in error_msg, (
            f"Expected connection-related message, got: {exc_info.value}"
        )
