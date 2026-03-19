"""MCP server for Verra One — exposes search and chat as tools over stdio."""

from __future__ import annotations

import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Tool definitions (static — returned on tools/list)
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "verra_search",
        "description": (
            "Search the user's business data (emails, documents, contracts, "
            "meeting notes, etc.) using hybrid retrieval. Returns relevant "
            "text passages with source attribution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "verra_ask",
        "description": (
            "Ask a question about the user's business data. Uses RAG to find "
            "relevant context and generate a comprehensive answer with source "
            "citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to answer",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "verra_status",
        "description": (
            "Get the status of the Verra knowledge base — number of documents, "
            "chunks, and entities indexed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# Engine initialisation (deferred until first tool call)
# ---------------------------------------------------------------------------


def _build_handler() -> Any:
    """Initialise stores and return a callable tool handler.

    Deferred so that the server process can start (and respond to
    ``initialize``) before the potentially-slow store setup runs.
    """
    import sqlite3

    from verra.agent.chat import ChatEngine
    from verra.agent.llm import LLMClient
    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.retrieval.router import parse_query
    from verra.retrieval.search import search as do_search
    from verra.store.db import DatabaseManager
    from verra.store.memory import MemoryStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()
    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")
    memory_store = MemoryStore.from_connection(db.core)
    llm = LLMClient(model=config.agent.model)

    entity_store = None
    try:
        from verra.store.entities import EntityStore

        entity_store = EntityStore.from_connection(db.core)
    except Exception:
        pass

    engine = ChatEngine(
        llm=llm,
        metadata_store=metadata_store,
        vector_store=vector_store,
        memory_store=memory_store,
        entity_store=entity_store,
    )

    def handle(name: str, args: dict[str, Any]) -> str:
        if name == "verra_search":
            query_text = args.get("query", "")
            n = int(args.get("n_results", 5))
            classified = parse_query(query_text)
            results = do_search(
                classified,
                metadata_store,
                vector_store,
                n_results=n,
                entity_store=entity_store,
            )
            if not results:
                return "No results found."
            parts: list[str] = []
            for i, r in enumerate(results, 1):
                source = (
                    r.metadata.get("file_name")
                    or r.metadata.get("subject")
                    or r.metadata.get("source_type")
                    or "unknown"
                )
                parts.append(f"[{i}] {source} (score: {r.score:.2f})\n{r.text[:500]}")
            return "\n\n---\n\n".join(parts)

        elif name == "verra_ask":
            question = args.get("question", "")
            response = engine.ask(question, use_multi_hop=False)
            answer = response.answer
            sources = [s.get("label", "") for s in response.sources[:5] if s.get("label")]
            if sources:
                return f"{answer}\n\nSources: {', '.join(sources)}"
            return answer

        elif name == "verra_status":
            core_db = VERRA_HOME / "core.db"
            if not core_db.exists():
                return "No data indexed yet."
            conn = sqlite3.connect(str(core_db))
            try:
                docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                try:
                    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                except Exception:
                    entities = 0
            finally:
                conn.close()
            return (
                f"Documents: {docs}\n"
                f"Chunks: {chunks}\n"
                f"Entities: {entities}\n"
                f"Model: {config.agent.model}"
            )

        return f"Unknown tool: {name}"

    return handle


# ---------------------------------------------------------------------------
# Stdio JSON-RPC 2.0 transport
# ---------------------------------------------------------------------------


def _send(msg: dict[str, Any]) -> None:
    """Write a single JSON-RPC message to stdout using the LSP framing."""
    text = json.dumps(msg)
    # Content-Length header uses byte length so multibyte chars are handled.
    body = text.encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n"
    sys.stdout.buffer.write(header.encode("ascii") + body)
    sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    """Read one LSP-framed JSON-RPC message from stdin.

    Returns None when stdin reaches EOF.
    """
    headers: dict[str, str] = {}
    while True:
        line_bytes = sys.stdin.buffer.readline()
        if not line_bytes:
            return None  # EOF
        line = line_bytes.decode("utf-8").rstrip("\r\n")
        if line == "":
            break  # blank line separates headers from body
        if ":" in line:
            key, _, val = line.partition(":")
            headers[key.strip()] = val.strip()

    content_length = int(headers.get("Content-Length", 0))
    if content_length == 0:
        return None

    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8"))


def run_stdio_server() -> None:
    """Run the MCP server over stdio (JSON-RPC 2.0).

    This is the entry point invoked by ``verra mcp``.  It loops forever,
    reading requests and writing responses, until stdin closes.
    """
    import logging

    logging.basicConfig(
        filename="/tmp/verra-mcp.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("verra-mcp")
    log.info("Verra MCP server starting")

    # Lazily initialised on the first tools/call so the server responds to
    # initialise/tools-list immediately.
    _handle: Any = None

    while True:
        try:
            msg = _read_message()
            if msg is None:
                log.info("stdin closed — shutting down")
                break

            method: str = msg.get("method", "")
            msg_id: Any = msg.get("id")
            params: dict[str, Any] = msg.get("params") or {}

            log.debug("received method=%s id=%s", method, msg_id)

            if method == "initialize":
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "verra-one", "version": "0.3.2"},
                        },
                    }
                )

            elif method == "notifications/initialized":
                pass  # notification — no response required

            elif method == "tools/list":
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": _TOOLS},
                    }
                )

            elif method == "tools/call":
                tool_name: str = params.get("name", "")
                tool_args: dict[str, Any] = params.get("arguments") or {}
                log.debug("tool_call name=%s args=%s", tool_name, tool_args)

                # Lazy init — build handler on first real call
                if _handle is None:
                    try:
                        _handle = _build_handler()
                    except Exception as exc:
                        log.exception("failed to initialise Verra engine")
                        _send(
                            {
                                "jsonrpc": "2.0",
                                "id": msg_id,
                                "result": {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"Verra failed to initialise: {exc}\n"
                                                "Run 'verra setup' to configure Verra."
                                            ),
                                        }
                                    ],
                                    "isError": True,
                                },
                            }
                        )
                        continue

                try:
                    result_text = _handle(tool_name, tool_args)
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [{"type": "text", "text": result_text}]
                            },
                        }
                    )
                except Exception as exc:
                    log.exception("tool call failed: %s", tool_name)
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [{"type": "text", "text": f"Error: {exc}"}],
                                "isError": True,
                            },
                        }
                    )

            elif method == "ping":
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})

            else:
                log.warning("unknown method: %s", method)
                if msg_id is not None:
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}",
                            },
                        }
                    )

        except Exception as exc:  # noqa: BLE001
            log.error("unhandled error in server loop: %s", exc, exc_info=True)
            break

    log.info("Verra MCP server stopped")
