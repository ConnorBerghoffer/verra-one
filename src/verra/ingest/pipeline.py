"""Ingestion pipeline — extract, chunk, analyse, embed, store."""


from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from verra.ingest.analyser import analyse_chunk_heuristic, process_analysis_results
from verra.ingest.chunking import chunk_document
from verra.ingest.dedup import find_near_duplicates
from verra.ingest.folder import crawl_folder
from verra.ingest.ner import extract_entities, extract_relationships, resolve_entities_to_registry
from verra.ingest.references import extract_references, resolve_references
from verra.ingest.time_resolver import extract_document_date, resolve_time_references
from verra.store.analysis import AnalysisStore
from verra.store.entities import EntityStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore

try:
    from verra.store.metadata import classify_document_authority
except ImportError:
    def classify_document_authority(file_name: str, file_path: str, content: str) -> tuple[str, int]:
        return ("general", 50)


@dataclass
class IngestStats:
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    entities_found: int = 0
    relationships_found: int = 0
    chunks_analysed: int = 0
    commitments_found: int = 0
    conflicts_found: int = 0
    near_duplicates_found: int = 0
    references_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class IngestPhase:
    """Describes the current phase of ingestion for a single file."""
    file_path: Path | None = None
    files_done: int = 0
    files_total: int = 0
    phase: str = ""            # 'scan', 'extract', 'chunk', 'dedup', 'embed', 'entities', 'analyse', 'done'
    detail: str = ""           # e.g. '14 chunks' or 'NER'
    chunks_so_far: int = 0
    entities_so_far: int = 0
    errors_so_far: int = 0


def ingest_folder(
    folder_path: Path,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    entity_store: EntityStore | None = None,
    analysis_store: AnalysisStore | None = None,
    tabular_store: Any | None = None,
    analysis_mode: str = "realtime",  # 'fast', 'realtime', 'deep'
    force_reindex: bool = False,
    progress_callback: Callable[[Path, int, int], None] | None = None,
    phase_callback: Callable[[IngestPhase], None] | None = None,
) -> IngestStats:
    """Ingest all supported documents from a local folder.

    Steps for each file:
    1. Compute content hash; skip if already indexed (unless force_reindex).
    2. Extract text via detect_and_extract().
    3. Split into semantic chunks.
    4. Store chunks in both SQLite (metadata) and ChromaDB (vectors).
    5. Update sync_state cursor.

    Parameters
    ----------
    folder_path:
        Root directory to crawl.
    metadata_store:
        SQLite metadata store instance.
    vector_store:
        ChromaDB vector store instance.
    force_reindex:
        If True, re-process files even if their hash hasn't changed.
    progress_callback:
        Optional callable invoked after each file attempt with signature
        ``(current_file, files_processed, files_total)``.  ``files_total``
        is -1 when the total count is not yet known (streaming walk).
    """
    stats = IngestStats()
    t0 = time.monotonic()

    # Batch-embedding buffer: accumulate (chunk_id, chunk) pairs across files
    # and flush to ChromaDB in batches of EMBED_BATCH_SIZE.
    EMBED_BATCH_SIZE = 100
    _embed_buffer_ids: list[Any] = []
    _embed_buffer_chunks: list[Any] = []

    def _emit(fp: Path | None, phase: str, detail: str = "") -> None:
        if phase_callback is not None:
            phase_callback(IngestPhase(
                file_path=fp,
                files_done=stats.files_processed + stats.files_skipped,
                files_total=files_total,
                phase=phase,
                detail=detail,
                chunks_so_far=stats.chunks_created,
                entities_so_far=stats.entities_found,
                errors_so_far=len(stats.errors),
            ))

    def _flush_embed_buffer() -> None:
        """Flush accumulated chunks to the vector store in one call."""
        if not _embed_buffer_ids:
            return
        n = len(_embed_buffer_ids)
        _emit(None, "embed", f"{n} vectors (batch flush)")
        vector_store.add_chunks(_embed_buffer_ids, _embed_buffer_chunks)
        _embed_buffer_ids.clear()
        _embed_buffer_chunks.clear()

    # Collect the full list of candidate files up-front so we can report a
    # meaningful ``files_total`` to the progress callback.
    files_total = 0
    _emit(None, "scan", "crawling folder")
    all_files = list(crawl_folder(folder_path))
    files_total = len(all_files)
    _emit(None, "scan", f"found {files_total} files")

    # Producer-consumer pattern: CPU-bound steps (classify, time-resolve,
    # chunk, NER, heuristic analysis) run in parallel across files via a
    # thread pool.  Serial steps (SQLite writes, embed buffer, dedup) run
    # on the main thread after collecting each result.

    def _process_single_file(
        file_path: Path,
        doc: Any,
        content_hash: str,
    ) -> dict[str, Any]:
        """CPU-bound work for one file. Returns data needed for serial steps.

        The content_hash is computed and the skip check is done on the main
        thread before this function is submitted, eliminating any TOCTOU race
        against the DB.  This function only does pure computation — classify,
        time-resolve, chunk, NER, heuristic analysis — with no DB access.
        """
        # Classify document authority
        doc_type, authority_weight = classify_document_authority(
            file_name=file_path.name,
            file_path=str(file_path),
            content=doc.content,
        )

        # Resolve relative time references before chunking
        doc_date = extract_document_date(doc.content, str(file_path))
        resolved_content = resolve_time_references(doc.content, doc_date)

        # Chunk (uses no shared state)
        # Note: document_id is not yet known (requires a DB write), so we
        # omit it here and patch it in on the main thread after add_document().
        chunks = chunk_document(
            resolved_content,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "format": doc.format,
                "source_type": "folder",
                "authority_weight": authority_weight,
                "document_type": doc_type,
            },
        )

        # NER — pure-Python regex, no DB writes
        entities_per_chunk: list[list[Any]] = []
        if entity_store is not None:
            for chunk in chunks:
                entities_per_chunk.append(extract_entities(chunk.text))
        else:
            entities_per_chunk = [[] for _ in chunks]

        # Heuristic analysis — no DB writes
        analyses: list[Any] = []
        if analysis_store is not None and analysis_mode == "realtime":
            for chunk in chunks:
                analyses.append(analyse_chunk_heuristic(chunk.text))
        else:
            analyses = [None] * len(chunks)

        return {
            "status": "processed",
            "file_path": file_path,
            "doc": doc,
            "content_hash": content_hash,
            "doc_type": doc_type,
            "authority_weight": authority_weight,
            "chunks": chunks,
            "entities_per_chunk": entities_per_chunk,
            "analyses": analyses,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict[Any, Path] = {}
        for file_path, doc in all_files:
            stats.files_found += 1

            # Hash check on the main thread — no TOCTOU race between
            # concurrent workers seeing "not in DB" and both processing.
            content_hash = _hash_content(doc.content)
            if not force_reindex:
                existing = metadata_store.get_document_by_hash(content_hash)
                if existing is not None:
                    stats.files_skipped += 1
                    _emit(file_path, "skip", "unchanged")
                    if progress_callback is not None:
                        progress_callback(file_path, stats.files_processed, files_total)
                    continue

            fut = pool.submit(_process_single_file, file_path, doc, content_hash)
            futures[fut] = file_path

        for fut in as_completed(futures):
            file_path = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                stats.errors.append(f"{file_path}: {exc}")
                stats.files_skipped += 1
                _emit(file_path, "error", str(exc)[:80])
                if progress_callback is not None:
                    progress_callback(file_path, stats.files_processed, files_total)
                continue

            if result["status"] == "skipped":
                stats.files_skipped += 1
                _emit(file_path, "skip", "unchanged")
                if progress_callback is not None:
                    progress_callback(file_path, stats.files_processed, files_total)
                continue

            # --- Serial section: all DB writes happen on the main thread ---
            try:
                doc = result["doc"]
                content_hash = result["content_hash"]
                doc_type = result["doc_type"]
                authority_weight = result["authority_weight"]
                chunks = result["chunks"]
                entities_per_chunk = result["entities_per_chunk"]
                analyses = result["analyses"]

                _emit(file_path, "extract", f"{len(doc.content):,} chars")

                # Remove old chunks if the file was previously indexed
                old_doc = metadata_store.get_document_by_path(str(file_path))
                if old_doc is not None:
                    vector_store.delete_by_document_id(old_doc["id"])
                    metadata_store.delete_document(old_doc["id"])

                # Register the document in SQLite (serial — gets the doc_id)
                doc_id = metadata_store.add_document(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    source_type="folder",
                    format=doc.format,
                    content_hash=content_hash,
                    page_count=doc.page_count,
                    extra_metadata=doc.metadata,
                    document_type=doc_type,
                    authority_weight=authority_weight,
                )

                # Load CSV into the tabular store for SQL querying (non-fatal)
                if tabular_store is not None and doc.format == "csv":
                    try:
                        raw_csv = file_path.read_text(errors="replace")
                        tabular_store.ingest_csv(file_path.name, str(file_path), raw_csv)
                        _emit(file_path, "tabular", f"loaded {file_path.name}")
                    except Exception:
                        pass  # CSV still gets chunked normally

                # Patch document_id into each chunk's metadata now that we have it
                for chunk in chunks:
                    chunk.metadata["document_id"] = doc_id

                _emit(file_path, "chunk", f"{len(chunks)} chunks")

                # Near-duplicate detection (reads + writes — serial)
                existing_chunk_texts = metadata_store.get_all_chunk_texts()
                dup_pairs: list[tuple[int, int, float]] = []
                if existing_chunk_texts and len(existing_chunk_texts) < 10000:
                    _emit(file_path, "dedup", f"vs {len(existing_chunk_texts)} existing")
                    dup_pairs = find_near_duplicates(chunks, existing_chunk_texts)
                    for new_idx, existing_id, score in dup_pairs:
                        chunks[new_idx].metadata["near_duplicate_of"] = existing_id
                        chunks[new_idx].metadata["near_duplicate_score"] = round(score, 4)
                        stats.near_duplicates_found += 1
                elif existing_chunk_texts:
                    _emit(file_path, "dedup", f"skipped ({len(existing_chunk_texts)} chunks, too large)")

                chunk_ids = metadata_store.add_chunks(doc_id, chunks, authority_weight=authority_weight)

                # Persist near-duplicate relationships
                for new_idx, existing_id, score in dup_pairs:
                    metadata_store.add_near_duplicate(
                        chunk_id=chunk_ids[new_idx],
                        near_duplicate_of=existing_id,
                        similarity_score=score,
                    )

                # Reference extraction
                _emit(file_path, "refs", "cross-references")
                for chunk, chunk_id in zip(chunks, chunk_ids):
                    refs = extract_references(chunk.text)
                    if refs:
                        chunk.metadata["has_references"] = True
                        stats.references_extracted += len(refs)
                        resolved = resolve_references(refs, metadata_store)
                        for ref, res in zip(refs, resolved):
                            metadata_store.add_chunk_reference(
                                source_chunk_id=chunk_id,
                                reference_text=ref["reference_text"],
                                reference_type=ref["reference_type"],
                                target_document_id=res.get("resolved_document_id"),
                                target_chunk_id=None,
                                confidence=res.get("confidence", 0.5),
                            )

                # Accumulate into the embedding buffer; flush when full
                _embed_buffer_ids.extend(chunk_ids)
                _embed_buffer_chunks.extend(chunks)
                if len(_embed_buffer_ids) >= EMBED_BATCH_SIZE:
                    _flush_embed_buffer()

                # Entity writes — NER was done in the thread; persist results here
                if entity_store is not None:
                    _emit(file_path, "entities", "NER")
                    for chunk, chunk_id, extracted in zip(chunks, chunk_ids, entities_per_chunk):
                        if extracted:
                            entity_ids = resolve_entities_to_registry(extracted, entity_store)
                            entity_store.link_chunk_batch(chunk_id, entity_ids)
                            stats.entities_found += len(entity_ids)

                            rels = extract_relationships(entity_ids, entity_store, source_chunk_id=chunk_id)
                            for ea, rtype, eb in rels:
                                entity_store.add_relationship(ea, rtype, eb, source_chunk_id=chunk_id)
                            stats.relationships_found += len(rels)

                # Analysis writes — heuristic analysis was done in thread; persist here
                if analysis_store is not None:
                    _emit(file_path, "analyse", f"{analysis_mode}")
                    for chunk, chunk_id, analysis in zip(chunks, chunk_ids, analyses):
                        if analysis_mode == "fast":
                            analysis_store.set_chunk_status(chunk_id, "pending")
                        elif analysis_mode == "realtime" and analysis is not None:
                            chunk_entity_ids: list[Any] = []
                            if entity_store is not None:
                                chunk_ents = entity_store.get_entities_for_chunk(chunk_id)
                                chunk_entity_ids = [e["id"] for e in chunk_ents]
                            process_analysis_results(
                                chunk_id=chunk_id,
                                analysis=analysis,
                                analysis_store=analysis_store,
                                entity_store=entity_store,
                                entity_ids=chunk_entity_ids,
                            )
                            stats.chunks_analysed += 1
                            stats.commitments_found += len(analysis.commitments or [])

                stats.files_processed += 1
                stats.chunks_created += len(chunks)
                _emit(file_path, "done", f"{len(chunks)} chunks, {stats.entities_found} entities total")

            except Exception as exc:
                stats.errors.append(f"{file_path}: {exc}")
                stats.files_skipped += 1
                _emit(file_path, "error", str(exc)[:80])

            finally:
                if progress_callback is not None:
                    progress_callback(file_path, stats.files_processed, files_total)

    # Flush any remaining chunks that did not fill a complete batch.
    _flush_embed_buffer()

    # Pre-compute table summaries for fast SQL at query time.
    if tabular_store is not None:
        try:
            tabular_store.precompute_summaries()
        except Exception:
            pass

    # Compute quality metrics across the whole batch if any files were processed.
    if stats.chunks_created > 0:
        # We don't carry the full chunk list across iterations, so recompute
        # a lightweight summary from what the pipeline already tracked.
        # A zero entity_count list is acceptable — entity_hit_rate will be 0.
        placeholder_chunks: list[Any] = []  # type: ignore[type-arg]
        # Pull the last doc's chunks from the metadata store for the quality pass.
    stats.elapsed_seconds = time.monotonic() - t0

    # Update sync state
    metadata_store.upsert_sync_state(
        source=f"folder:{folder_path}",
        cursor=str(time.time()),
        items_processed=stats.files_processed,
        status="idle",
    )

    return stats


def _hash_content(content: str) -> str:
    """Return a hex digest of the content string."""
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
