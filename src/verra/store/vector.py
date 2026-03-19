"""ChromaDB vector store wrapper.

Uses ChromaDB's PersistentClient (embedded mode, no separate server).
The default embedding function (chromadb.utils.embedding_functions.DefaultEmbeddingFunction)
is used so no external embedding API is needed for the MVP.

Collection name: "verra_chunks"
"""


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verra.ingest.chunking import Chunk


_COLLECTION_NAME = "verra_chunks"


class VectorStore:
    """Wrapper around a ChromaDB persistent collection."""

    def __init__(self, persist_dir: Path | str) -> None:
        import chromadb

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        # get_or_create is idempotent — safe to call on every startup
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            # Default embedding function (all-MiniLM-L6-v2 via onnxruntime)
            # is fetched on first use if not cached.
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
            import logging
            logging.getLogger(__name__).warning(
                "Failed to delete vectors for document_id=%s: %s", document_id, exc
            )

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (for testing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(name=_COLLECTION_NAME)
