"""Base connector interface for Verra One data sources.


All connectors (Gmail, Drive, Outlook, etc.) should inherit from
BaseConnector. This ensures a consistent interface for:
  - Authentication
  - Data fetching
  - CLI integration
  - Sync daemon integration

To add a new connector:
  1. Create a class inheriting from BaseConnector.
  2. Implement authenticate(), fetch(), and ingest().
  3. Decorate the class with @register_connector.

Example
-------
    from verra.ingest.base import BaseConnector, register_connector
    from verra.ingest.pipeline import IngestStats

    @register_connector
    class NotionConnector(BaseConnector):
        connector_type = "notion"
        display_name = "Notion"

        def authenticate(self) -> bool:
            ...

        def ingest(self, metadata_store, vector_store, **kwargs) -> IngestStats:
            ...
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from verra.ingest.pipeline import IngestStats


class BaseConnector(ABC):
    """Abstract base class for all Verra data source connectors.

    Subclasses must set ``connector_type`` and ``display_name`` as class
    attributes, then implement ``authenticate()`` and ``ingest()``.

    Class attributes
    ----------------
    connector_type:
        Short identifier used as the registry key and in CLI commands,
        e.g. ``"gmail"``, ``"drive"``, ``"notion"``.
    display_name:
        Human-readable label shown in the UI and log output,
        e.g. ``"Gmail"``, ``"Google Drive"``, ``"Notion"``.
    """

    # Override in every subclass.
    connector_type: str = ""
    display_name: str = ""

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the data source.

        Implementations should:
        - Load cached tokens when available and refresh if expired.
        - Trigger an interactive auth flow (OAuth, device code, etc.)
          only when no valid cached token exists.
        - Persist new tokens to disk with 0o600 permissions.

        Returns
        -------
        bool
            True when authentication succeeded and the connector is ready
            to make API calls; False when a required credential file is
            missing and setup instructions have been printed instead.

        Raises
        ------
        RuntimeError
            On unexpected auth failure (token exchange error, network
            issue, etc.) that prevents the connector from being usable.
        ImportError
            When a required third-party library (e.g. ``msal``,
            ``google-auth-oauthlib``) is not installed.
        """
        ...

    @abstractmethod
    def ingest(
        self,
        metadata_store: Any,
        vector_store: Any,
        **kwargs: Any,
    ) -> IngestStats:
        """Ingest data from the source into the Verra knowledge base.

        Implementations should follow the standard pipeline pattern:
        1. Check sync state for a delta cursor (page token, history ID,
           etc.) and decide between a full fetch and a delta fetch.
        2. Fetch items from the remote API.
        3. Compute a content hash per item; skip unchanged items.
        4. Register each new document in ``metadata_store``.
        5. Chunk the extracted text and add chunks to both stores.
        6. Persist the updated sync cursor via
           ``metadata_store.upsert_sync_state()``.
        7. Return a populated ``IngestStats`` instance.

        Parameters
        ----------
        metadata_store:
            ``MetadataStore`` instance (SQLite-backed).
        vector_store:
            ``VectorStore`` instance (ChromaDB-backed).
        **kwargs:
            Connector-specific options (e.g. ``since``, ``folder_id``,
            ``max_results``, ``force_reindex``).

        Returns
        -------
        IngestStats
            Populated counters and any per-item error strings.
        """
        ...


# ---------------------------------------------------------------------------
# Connector registry
# ---------------------------------------------------------------------------

# Maps connector_type strings to their concrete connector classes.
# Populated at import time by @register_connector decorators.
CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {}


def register_connector(cls: type[BaseConnector]) -> type[BaseConnector]:
    """Class decorator — register *cls* in CONNECTOR_REGISTRY.

    The class must have a non-empty ``connector_type`` attribute.

    Usage
    -----
        @register_connector
        class NotionConnector(BaseConnector):
            connector_type = "notion"
            ...

    Raises
    ------
    ValueError
        If ``connector_type`` is empty or a connector with the same type
        has already been registered.
    """
    if not cls.connector_type:
        raise ValueError(
            f"Cannot register {cls.__name__}: connector_type must be set."
        )
    if cls.connector_type in CONNECTOR_REGISTRY:
        existing = CONNECTOR_REGISTRY[cls.connector_type]
        raise ValueError(
            f"connector_type {cls.connector_type!r} is already registered "
            f"by {existing.__name__}. Each connector_type must be unique."
        )
    CONNECTOR_REGISTRY[cls.connector_type] = cls
    return cls


def get_connector(connector_type: str) -> type[BaseConnector] | None:
    """Return the connector class registered under *connector_type*, or None."""
    return CONNECTOR_REGISTRY.get(connector_type)


def list_connectors() -> list[str]:
    """Return all registered connector type strings in insertion order."""
    return list(CONNECTOR_REGISTRY.keys())
