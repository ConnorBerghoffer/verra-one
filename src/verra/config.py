"""Configuration model for Verra One.

Config is stored at ~/.verra/config.yaml.
On first run, ensure_data_dir() creates the directory with all subdirectories.
"""


from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VERRA_HOME = Path(os.environ.get("VERRA_HOME", Path.home() / ".verra"))


def ensure_data_dir() -> Path:
    """Create ~/.verra/ and all required subdirectories if they don't exist.

    Database layout (consolidated — 2 SQLite files + ChromaDB):
      core.db     — metadata, entities, state, memory
      analysis.db — analysis, provenance, assertions, coverage
      chroma/     — vector embeddings (ChromaDB, managed separately)
      oauth/      — OAuth token cache
      logs/       — rotating log files

    The sqlite/ subdirectory is NOT created; databases live directly in
    VERRA_HOME so that ``ls ~/.verra/`` is clean and predictable.
    """
    VERRA_HOME.mkdir(parents=True, exist_ok=True, mode=0o700)
    for subdir in ["chroma", "oauth", "logs"]:
        (VERRA_HOME / subdir).mkdir(exist_ok=True, mode=0o700)
    # Ensure restrictive permissions on data directory
    try:
        VERRA_HOME.chmod(0o700)
        (VERRA_HOME / "oauth").chmod(0o700)
    except OSError:
        pass
    return VERRA_HOME


# ---------------------------------------------------------------------------
# Config models (Pydantic v2)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    name: str = "verra"
    model: str = "ollama/llama3.2"
    api_key: str | None = None


class SourceConfig(BaseModel):
    type: str  # "folder" | "gmail" | "drive" | "outlook"
    path: str | None = None
    account: str | None = None
    labels: list[str] = Field(default_factory=list)
    since: str | None = None  # ISO date string, e.g. "2023-01-01"


class SyncConfig(BaseModel):
    interval: int = 300  # seconds
    enabled: bool = True


class BriefingConfig(BaseModel):
    enabled: bool = True
    max_items: int = 5
    stale_lead_days: int = 14
    contract_warning_days: int = 30


class MemoryConfig(BaseModel):
    persist: bool = True
    path: str = str(VERRA_HOME / "core.db")


class RemoteConfig(BaseModel):
    host: str | None = None
    port: int = 8484


class VerraConfig(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sources: list[SourceConfig] = Field(default_factory=list)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> VerraConfig:
    """Load config from YAML, returning defaults if the file doesn't exist."""
    config_path = path or VERRA_HOME / "config.yaml"
    if not config_path.exists():
        return VerraConfig()
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return VerraConfig.model_validate(raw)


def save_config(config: VerraConfig, path: Path | None = None) -> None:
    """Persist config to YAML."""
    config_path = path or VERRA_HOME / "config.yaml"
    ensure_data_dir()
    import stat

    # Atomic write: write to temp file then rename (prevents corrupt config on crash)
    tmp_path = config_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
    try:
        tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass
    tmp_path.replace(config_path)  # atomic on POSIX
