"""Local folder crawler.

Recursively walks a directory, skipping ignored paths and unsupported formats,
and yields (path, ExtractedDocument) tuples ready for chunking.
"""


from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Generator

from verra.config import VERRA_HOME
from verra.ingest.extractors import SUPPORTED_EXTENSIONS, ExtractedDocument, detect_and_extract

logger = logging.getLogger(__name__)


# Default ignore patterns (always applied, even without .verraignore)

_DEFAULT_IGNORES: list[str] = [
    # Version control / build artefacts
    ".git",
    ".svn",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".venv",
    "venv",
    "node_modules",
    # macOS / Windows system files
    ".DS_Store",
    "Thumbs.db",
    # Large binary / archive formats
    "*.zip",
    "*.tar.gz",
    "*.rar",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.dmg",
    "*.iso",
    # Media files (not extractable as text)
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.bmp",
    "*.svg",
    "*.ico",
    # Credential / key files
    "*.key",
    "*.pem",
    "*.crt",
    "*.pfx",
    "*secret*",
    "*credential*",
    "*password*",
    ".env",
    ".env.*",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
]


# Credential heuristic detection

_CREDENTIAL_PATTERNS: list[str] = [
    "server-keys",
    "api-key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "private-key",
    "privatekey",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".pem",
    ".key",
    ".env",
    "client_secret",
    "service_account",
    "credentials.json",
    "keyfile",
    ".pfx",
    ".p12",
]


def is_credential_file(file_path: Path) -> bool:
    """Heuristic check: return True if the file name suggests it contains credentials.

    Checks the lower-cased file name (including extension) against a list of
    known credential-related terms.  This is a best-effort guard — it will not
    catch every case, but it prevents the most common mistakes.
    """
    name_lower = file_path.name.lower()
    return any(p in name_lower for p in _CREDENTIAL_PATTERNS)


# .verraignore support


def load_ignore_patterns(folder_path: Path) -> list[str]:
    """Load ignore patterns from .verraignore files.

    Checks two locations (in order, later entries take precedence by appending):

    1. ``~/.verra/.verraignore`` — global patterns applied to every ingest
    2. ``{folder_path}/.verraignore`` — local patterns for this folder only

    Format: one gitignore-style glob pattern per line.
    Lines starting with ``#`` are treated as comments and ignored.
    Blank lines are also skipped.

    The built-in ``_DEFAULT_IGNORES`` list is always prepended so that system
    files, credential files, and media are excluded even when no .verraignore
    file exists.
    """
    patterns: list[str] = list(_DEFAULT_IGNORES)

    for source in [VERRA_HOME / ".verraignore", folder_path / ".verraignore"]:
        if source.exists():
            for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)

    return patterns


def should_ignore(file_path: Path, patterns: list[str]) -> bool:
    """Return True if the file name or path matches any of the given patterns.

    Checks both the bare file name and the full path string so that patterns
    like ``*.pyc`` (name-only) and ``node_modules/`` (directory prefix) both work.
    """
    candidates = [file_path.name, str(file_path)]
    for pattern in patterns:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


def _is_ignored(path: Path, root: Path, patterns: list[str]) -> bool:
    """Return True if any segment of the path relative to root matches an ignore pattern."""
    relative = path.relative_to(root)
    candidates = [path.name, str(relative)]
    for pattern in patterns:
        for candidate in candidates:
            if fnmatch.fnmatch(candidate, pattern):
                return True
    return False


# Public API


def crawl_folder(
    root: Path,
) -> Generator[tuple[Path, ExtractedDocument], None, None]:
    """Yield (path, ExtractedDocument) for every supported file under root.

    Skips:
    - Files matching .verraignore patterns (global + local) or default patterns.
    - Files detected as likely credential files by name heuristic.
    - Files with unsupported extensions.
    - Files that cannot be read/extracted (logs error, continues).
    """
    root = root.resolve()
    patterns = load_ignore_patterns(root)

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)

        # Prune ignored and symlinked directories in-place so os.walk
        # doesn't descend into them (prevents symlink traversal attacks).
        dirnames[:] = [
            d for d in dirnames
            if not (current_dir / d).is_symlink()
            and not _is_ignored(current_dir / d, root, patterns)
        ]

        for filename in filenames:
            file_path = current_dir / filename

            # Security: skip symlinks to prevent symlink traversal attacks
            if file_path.is_symlink():
                logger.debug("Skipped symlink: %s", file_path)
                continue

            # Security: verify resolved path is still under root
            try:
                resolved = file_path.resolve()
                if not str(resolved).startswith(str(root)):
                    logger.warning("Skipped path escaping root: %s", file_path)
                    continue
            except OSError:
                continue

            if _is_ignored(file_path, root, patterns):
                continue

            # Credential heuristic: skip files that look like they store secrets.
            if is_credential_file(file_path):
                logger.warning("Skipped potential credential file: %s", file_path)
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Security: skip files larger than 500MB
            try:
                if file_path.stat().st_size > 500 * 1024 * 1024:
                    logger.warning("Skipped oversized file (>500MB): %s", file_path)
                    continue
            except OSError:
                continue

            try:
                doc = detect_and_extract(file_path)
                yield file_path, doc
            except Exception as exc:
                # Non-fatal: log and continue
                logger.warning("Could not extract %s: %s", file_path, exc)
