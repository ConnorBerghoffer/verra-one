"""HTTP API server for Verra One.

Exposes the core functionality over HTTP so third-party UIs,
dashboards, and integrations can interact with the system.

Usage:
    verra serve --port 8484
    verra serve --port 8484 --host 0.0.0.0  # listen on all interfaces

Endpoints:
    GET  /api/health              — liveness probe
    GET  /api/status              — system stats
    POST /api/chat                — ask a question, get a complete response
    GET  /api/chat/stream         — SSE streaming response
    POST /api/search              — vector search without LLM
    GET  /api/documents           — list ingested documents (paginated)
    DELETE /api/documents/{doc_id} — delete a document and its vectors
    GET  /api/briefing            — current briefing items
    GET  /api/entities            — list extracted entities
"""


from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from verra.config import VERRA_HOME, ensure_data_dir, load_config

# ---------------------------------------------------------------------------
# Module-level state — populated by create_app()
# ---------------------------------------------------------------------------

_required_key: str | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    conversation_id: int


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DocumentsResponse(BaseModel):
    documents: list[dict[str, Any]]
    total: int


class DeleteResponse(BaseModel):
    deleted: bool


class StatusResponse(BaseModel):
    documents: int
    chunks: int
    entities: int
    disk_bytes: int
    model: str


class BriefingItemOut(BaseModel):
    category: str
    title: str
    detail: str
    urgency: int


class BriefingResponse(BaseModel):
    items: list[BriefingItemOut]


class EntitiesResponse(BaseModel):
    entities: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Disk usage helper (replicated from cli.py to avoid import of Rich)
# ---------------------------------------------------------------------------


def _disk_usage(path: Any) -> int:
    """Return total bytes used recursively. Returns 0 if missing."""
    from pathlib import Path

    p = Path(path)
    total = 0
    if not p.exists():
        return 0
    for child in p.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Store initialisation helper
# This is called once at startup and the stores are held on app.state.
# ---------------------------------------------------------------------------


def _init_stores(app: FastAPI) -> None:
    """Create DatabaseManager, stores, and cache them on app.state."""
    from verra.store.db import DatabaseManager
    from verra.store.entities import EntityStore
    from verra.store.memory import MemoryStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    entity_store = EntityStore.from_connection(db.core)
    memory_store = MemoryStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    app.state.db = db
    app.state.config = config
    app.state.metadata_store = metadata_store
    app.state.entity_store = entity_store
    app.state.memory_store = memory_store
    app.state.vector_store = vector_store


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(api_key: str | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    api_key:
        When set, every request (except GET /api/health) must include this
        value in the ``X-API-Key`` header. Pass None to disable auth.
    """
    global _required_key
    _required_key = api_key

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialise stores on startup and clean up on shutdown."""
        _init_stores(app)
        yield
        if hasattr(app.state, "db"):
            app.state.db.close()

    app = FastAPI(
        title="Verra One API",
        version="0.1.0",
        description="HTTP API for the Verra One business knowledge assistant.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS — permit localhost frontends and custom origins by default.
    # Tighten this for production by passing allow_origins explicitly.
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                       "http://localhost:5173", "http://127.0.0.1:5173",
                       "http://localhost:8080", "http://127.0.0.1:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Auth middleware — applied to every route except /api/health
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Any) -> Any:
        if request.url.path == "/api/health":
            return await call_next(request)
        if _required_key:
            incoming_key = request.headers.get("X-API-Key")
            if incoming_key != _required_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )
        return await call_next(request)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    # ---- Health -------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Simple liveness probe — always returns 200 when the server is up."""
        return HealthResponse(status="ok", version="0.1.0")

    # ---- Status -------------------------------------------------------

    @app.get("/api/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        """Return system-level statistics: document count, chunk count, entity count, disk use."""
        config = app.state.config
        core_db = VERRA_HOME / "core.db"

        doc_count = 0
        chunk_count = 0
        entity_count = 0

        if core_db.exists():
            conn = sqlite3.connect(str(core_db))
            conn.row_factory = sqlite3.Row
            try:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            except Exception:
                pass
            finally:
                conn.close()

        return StatusResponse(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            disk_bytes=_disk_usage(VERRA_HOME),
            model=config.agent.model,
        )

    # ---- Chat (blocking) ---------------------------------------------

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        """Send a message and receive a complete answer with source citations.

        A new conversation is created when ``conversation_id`` is omitted.
        Subsequent turns should pass the returned ``conversation_id`` to
        maintain context.
        """
        from verra.agent.chat import ChatEngine
        from verra.agent.llm import LLMClient

        config = app.state.config
        try:
            llm = LLMClient(model=config.agent.model)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc

        try:
            engine = ChatEngine(
                llm=llm,
                metadata_store=app.state.metadata_store,
                vector_store=app.state.vector_store,
                memory_store=app.state.memory_store,
                entity_store=app.state.entity_store,
                conversation_id=body.conversation_id,
            )
            response = engine.ask(body.message)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chat error: {exc}") from exc

        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            conversation_id=engine.conversation_id,
        )

    # ---- Chat (SSE streaming) ----------------------------------------

    async def _stream_generator(
        message: str,
        conversation_id: int | None,
        config: Any,
        app_state: Any,
    ):
        """Async generator that yields SSE-formatted chunks from stream_ask()."""
        from verra.agent.chat import ChatEngine
        from verra.agent.llm import LLMClient

        try:
            llm = LLMClient(model=config.agent.model)
            engine = ChatEngine(
                llm=llm,
                metadata_store=app_state.metadata_store,
                vector_store=app_state.vector_store,
                memory_store=app_state.memory_store,
                entity_store=app_state.entity_store,
                conversation_id=conversation_id,
            )
            for chunk in engine.stream_ask(message):
                # Escape newlines inside the data payload so SSE framing works
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    @app.get("/api/chat/stream")
    async def chat_stream(
        message: str = Query(..., description="The user's message"),
        conversation_id: int | None = Query(None, description="Existing conversation ID"),
    ) -> StreamingResponse:
        """Stream the assistant response as Server-Sent Events.

        Each ``data:`` line is a text chunk.  The final event is ``data: [DONE]``.
        """
        return StreamingResponse(
            _stream_generator(
                message=message,
                conversation_id=conversation_id,
                config=app.state.config,
                app_state=app.state,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering when proxied
            },
        )

    # ---- Search -------------------------------------------------------

    @app.post("/api/search", response_model=SearchResponse)
    async def search(body: SearchRequest) -> SearchResponse:
        """Run a vector search without invoking the LLM.

        Returns raw chunk results ranked by semantic similarity and authority.
        """
        from verra.retrieval.router import parse_query
        from verra.retrieval.search import search as _search

        try:
            classified = parse_query(body.query)
            results = _search(
                classified,
                app.state.metadata_store,
                app.state.vector_store,
                n_results=body.n_results,
                entity_store=app.state.entity_store,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search error: {exc}") from exc

        return SearchResponse(
            results=[
                SearchResult(
                    text=r.text,
                    score=round(r.score, 4),
                    metadata=r.metadata,
                )
                for r in results
            ]
        )

    # ---- Documents ----------------------------------------------------

    @app.get("/api/documents", response_model=DocumentsResponse)
    async def list_documents(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        source_type: str | None = Query(None, description="Filter by source type, e.g. 'folder'"),
    ) -> DocumentsResponse:
        """Return a paginated list of all ingested documents."""
        all_docs = app.state.metadata_store.list_documents(source_type=source_type)
        total = len(all_docs)
        page = all_docs[offset: offset + limit]
        return DocumentsResponse(documents=page, total=total)

    @app.delete("/api/documents/{doc_id}", response_model=DeleteResponse)
    async def delete_document(doc_id: int) -> DeleteResponse:
        """Delete a document and all of its associated vector embeddings.

        The document row is removed from SQLite (cascading to chunks) and the
        corresponding vectors are removed from ChromaDB.
        """
        # Verify the document exists before attempting deletion
        docs = app.state.metadata_store.list_documents()
        doc_ids = {d["id"] for d in docs}
        if doc_id not in doc_ids:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        try:
            app.state.vector_store.delete_by_document_id(doc_id)
            app.state.metadata_store.delete_document(doc_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Deletion failed: {exc}") from exc

        return DeleteResponse(deleted=True)

    # ---- Briefing -----------------------------------------------------

    @app.get("/api/briefing", response_model=BriefingResponse)
    async def briefing() -> BriefingResponse:
        """Return current actionable briefing items (stale leads, expiring contracts, etc.)."""
        from verra.briefing.detector import BriefingDetector

        core_db = VERRA_HOME / "core.db"
        if not core_db.exists():
            return BriefingResponse(items=[])

        config = app.state.config

        try:
            detector = BriefingDetector(
                core_conn=app.state.db.core,
                analysis_conn=app.state.db.analysis,
                config=config.briefing,
            )
            items = detector.detect_all()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Briefing error: {exc}") from exc

        return BriefingResponse(
            items=[
                BriefingItemOut(
                    category=item.category,
                    title=item.title,
                    detail=item.detail,
                    urgency=item.urgency,
                )
                for item in items
            ]
        )

    # ---- Entities ----------------------------------------------------

    @app.get("/api/entities", response_model=EntitiesResponse)
    async def list_entities(
        type: str | None = Query(None, description="Filter by entity type, e.g. 'person'"),
        limit: int = Query(50, ge=1, le=500),
    ) -> EntitiesResponse:
        """Return extracted entities, optionally filtered by type."""
        try:
            entities = app.state.entity_store.list_entities(entity_type=type)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Entity query error: {exc}") from exc

        return EntitiesResponse(entities=entities[:limit])

    return app
