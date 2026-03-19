"""ChromaDB vector store wrapper.

Uses ChromaDB's PersistentClient (embedded mode, no separate server).
Embedding model: nomic-ai/nomic-embed-text-v1.5 (768-dim, matryoshka).
Requires `einops` in the environment — installed automatically with `pip install einops`.

Collection name: "verra_chunks"

**Important — model migration note**:
If you switch embedding models on an existing Verra installation the stored
vectors are incompatible with the new model and searches will silently return
wrong results.  Re-ingest from scratch after any model change:

    rm -rf ~/.verra/chroma
    verra ingest <data-dir>

The collection metadata key ``embedding_model`` records which model was used
so that future tooling can detect mismatches automatically.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verra.ingest.chunking import Chunk


_COLLECTION_NAME = "verra_chunks"
_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

logger = logging.getLogger(__name__)


def _make_embedding_function() -> Any:
    """Return a SentenceTransformerEmbeddingFunction for *_EMBEDDING_MODEL*.

    Falls back to ``all-MiniLM-L12-v2`` if the nomic model cannot be loaded
    (e.g. missing ``einops`` or network issues), then to ``BAAI/bge-small-en-v1.5``
    as a last resort.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    candidates = [
        dict(model_name=_EMBEDDING_MODEL, trust_remote_code=True),
        dict(model_name="all-MiniLM-L12-v2"),
        dict(model_name="BAAI/bge-small-en-v1.5"),
    ]

    for kwargs in candidates:
        try:
            ef = SentenceTransformerEmbeddingFunction(**kwargs)
            # Warm-up: confirm the model actually loads before returning
            ef(["warmup"])
            model_name = kwargs["model_name"]
            if model_name != _EMBEDDING_MODEL:
                logger.warning(
                    "nomic-embed-text-v1.5 unavailable; using fallback embedding "
                    "model %r.  Retrieval quality may differ.",
                    model_name,
                )
            return ef, model_name
        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding model %r failed to load: %s", kwargs["model_name"], exc)

    raise RuntimeError(
        "No embedding model could be loaded.  Install `einops` and ensure "
        "sentence-transformers is available: pip install einops sentence-transformers"
    )


class VectorStore:
    """Wrapper around a ChromaDB persistent collection."""

    def __init__(self, persist_dir: Path | str) -> None:
        import chromadb

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        # Use nomic-embed-text-v1.5 for better retrieval quality (768-dim,
        # matryoshka representation learning).  Falls back gracefully if the
        # model is unavailable — see _make_embedding_function() above.
        self._embedding_fn, self._active_model = _make_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            # Record the model in collection metadata so future tooling can
            # detect incompatible re-opens (e.g. after a model change).
            metadata={"embedding_model": self._active_model},
        )

        self._warn_model_mismatch()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warn_model_mismatch(self) -> None:
        """Log a warning if the collection was built with a different model.

        ChromaDB does not enforce embedding-function consistency across opens,
        so we store the model name in collection metadata ourselves and check
        it at startup.
        """
        stored = (self._collection.metadata or {}).get("embedding_model")
        if stored and stored != self._active_model:
            logger.warning(
                "Vector store mismatch: collection was indexed with %r but the "
                "active embedding model is %r.  Search results will be unreliable. "
                "Re-ingest from scratch: rm -rf ~/.verra/chroma && verra ingest <dir>",
                stored,
                self._active_model,
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunk_ids: list[int], chunks: list[Chunk]) -> None:
        """Embed and store a list of chunks.

        chunk_ids must be the SQLite IDs from MetadataStore so we can
        cross-reference back to full metadata.
        """
        if not chunks:
            return

        ids = [str(cid) for cid in chunk_ids]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            # ChromaDB metadata values must be str, int, float, or bool
            safe_meta: dict[str, Any] = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            metadatas.append(safe_meta)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-n most similar chunks for a query string.

        Parameters
        ----------
        query:
            Natural language question or search string.
        n_results:
            How many results to return.
        where:
            Optional ChromaDB metadata filter dict.

        Returns
        -------
        List of dicts with keys: id, document, metadata, distance.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        hits: list[dict[str, Any]] = []
        for i, doc_id in enumerate(result["ids"][0]):
            hits.append(
                {
                    "id": doc_id,
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                    "distance": result["distances"][0][i] if result["distances"] else None,
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_document_id(self, document_id: int) -> None:
        """Remove all chunks belonging to a specific document."""
        try:
            # Query for all chunk IDs belonging to this document
            # Try both int and string since ChromaDB where-clause requires exact type match
            results = self._collection.get(
                where={"document_id": document_id}
            )
            if not results.get("ids"):
                results = self._collection.get(
                    where={"document_id": str(document_id)}
                )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
        except Exception as exc:
            logger.warning(
                "Failed to delete vectors for document_id=%s: %s", document_id, exc
            )

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (for testing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"embedding_model": self._active_model},
        )
