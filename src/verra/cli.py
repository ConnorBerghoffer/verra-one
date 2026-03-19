"""CLI for Verra One."""


from __future__ import annotations

# Suppress LiteLLM stderr noise before any downstream imports trigger it.
import os

os.environ.setdefault("LITELLM_LOG", "ERROR")

import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Suppress LiteLLM debug output.
try:
    import litellm

    litellm.suppress_debug_info = True
    litellm.set_verbose = False  # type: ignore[assignment]
except ImportError:
    pass

console = Console()

__version__ = "1.0.1"


def _check_for_updates() -> None:
    """Check PyPI for a newer version. Non-blocking, silent on failure."""
    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://pypi.org/pypi/verra-one/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        latest = data.get("info", {}).get("version", "")
        if latest and latest != __version__:
            console.print(
                f"  [yellow]Update available:[/yellow] {__version__} → {latest}"
            )
            console.print(
                f"  [dim]Run: pipx upgrade verra-one[/dim]\n"
            )
    except Exception:
        pass  # offline, PyPI down, not published yet — all fine


def _apply_api_key_from_config() -> None:
    """If config.agent.api_key is set, configure LiteLLM directly.

    Avoids leaking the key via os.environ to child processes.
    Falls back to env var only if LiteLLM direct config fails.
    """
    try:
        from verra.config import load_config

        config = load_config()
        if config.agent.api_key:
            model = config.agent.model.lower()
            # Set env var for LiteLLM (required by its internal routing).
            # Use setdefault so explicit env vars take precedence.
            if "anthropic" in model or "claude" in model:
                os.environ.setdefault("ANTHROPIC_API_KEY", config.agent.api_key)
            elif "gpt" in model or "openai" in model:
                os.environ.setdefault("OPENAI_API_KEY", config.agent.api_key)
    except Exception:
        console.print("[yellow]Warning: could not load config for API key.[/yellow]")


def _disk_usage(path: Path) -> int:
    """Return total bytes used by *path* recursively. Returns 0 if missing."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _has_data() -> bool:
    """Return True if any data has been ingested."""
    from verra.config import VERRA_HOME

    chroma_dir = VERRA_HOME / "chroma"
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def _print_model_recommendations() -> None:
    """Print recommended Ollama models with RAM requirements."""
    console.print()
    console.print("  [bold]Recommended models:[/bold]")
    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Notes")
    table.add_row("1", "llama3.1:8b", "4.7 GB", "8 GB", "[green]Recommended[/green] — best balance of quality and speed")
    table.add_row("2", "llama3.2:3b", "2.0 GB", "4 GB", "Fast, works on low-end hardware")
    table.add_row("3", "qwen2.5:7b", "4.4 GB", "8 GB", "Strong with structured data and tables")
    table.add_row("4", "gemma2:9b", "5.4 GB", "10 GB", "Google, strong reasoning")
    table.add_row("5", "phi4:14b", "9.1 GB", "16 GB", "Microsoft, great instruction following")
    table.add_row("6", "llama3.1:70b", "40 GB", "48 GB", "Near-cloud quality, needs powerful machine")
    console.print(table)
    console.print()


def _get_doc_count() -> int:
    """Quick doc count from core.db, 0 if not available."""
    import sqlite3
    from verra.config import VERRA_HOME

    core_db = VERRA_HOME / "core.db"
    if not core_db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(core_db))
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _print_banner(model: str) -> None:
    """Print the Verra One MOTD."""
    from rich.rule import Rule

    doc_count = _get_doc_count()
    docs_str = f"{doc_count} docs indexed" if doc_count > 0 else "no data yet"

    console.print()
    console.print(Rule(
        Text.assemble((" verra one ", "bold cyan")),
        style="dim",
    ))
    console.print()
    console.print(f"  [dim]model[/dim]  {model}")
    console.print(f"  [dim]data [/dim]  {docs_str}")
    if doc_count == 0:
        console.print(f"  [dim]      [/dim]  [yellow]run: verra ingest <folder>[/yellow]")
    console.print()


def _print_status_bar(model: str) -> None:
    """Print the bottom status strip."""
    doc_count = _get_doc_count()
    docs_str = f"{doc_count} docs" if doc_count > 0 else "no data"
    model_short = model.split("/")[-1] if "/" in model else model
    console.print(
        f"[dim]  {model_short} | {docs_str} | /help for commands | Ctrl+C to quit[/dim]"
    )
    console.print()


# Interactive chat REPL (Claude Code-style)


def _run_chat_repl(model_override: str | None = None) -> None:
    """Launch the interactive chat REPL with streaming responses."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from rich.live import Live

    from verra.agent.chat import ChatEngine
    from verra.agent.llm import LLMClient
    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.store.db import DatabaseManager
    from verra.store.memory import MemoryStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    _apply_api_key_from_config()
    ensure_data_dir()
    config = load_config()
    effective_model = model_override or config.agent.model

    if not _has_data():
        console.print(
            "[yellow]No data indexed yet.[/yellow] "
            "Run [bold]verra ingest <folder>[/bold] to add your documents first.\n"
        )

    # Initialize stores
    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")
    memory_store = MemoryStore.from_connection(db.core)
    llm = LLMClient(model=effective_model)

    # Optional stores (may not exist yet)
    entity_store = None
    try:
        from verra.store.entities import EntityStore

        entity_store = EntityStore.from_connection(db.core)
    except Exception:
        pass

    tabular_store = None
    try:
        from verra.store.tabular import TabularStore

        tabular_store = TabularStore(VERRA_HOME / "tabular.db")
    except Exception:
        pass

    engine = ChatEngine(
        llm=llm,
        metadata_store=metadata_store,
        vector_store=vector_store,
        memory_store=memory_store,
        entity_store=entity_store,
        tabular_store=tabular_store,
    )

    _print_banner(effective_model)
    _print_status_bar(effective_model)

    # prompt_toolkit setup — Enter sends, Escape+Enter for newline
    bindings = KeyBindings()

    @bindings.add(Keys.Enter)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.Escape, Keys.Enter)
    def _newline(event):
        event.current_buffer.insert_text("\n")

    session: PromptSession = PromptSession(
        message=HTML("<ansibrightcyan><b>&gt; </b></ansibrightcyan>"),
        key_bindings=bindings,
        multiline=True,
    )

    # Track last retrieval results so /sources can display them
    _last_results: list[Any] = []
    # Track last conversation_id for feedback recording
    _last_conv_id: list[int | None] = [None]

    try:
        while True:
            try:
                user_input = session.prompt().strip()
            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye.[/dim]")
                break
            except EOFError:
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue

            # Feedback shortcut — y/n/yes/no after a response records rating
            if user_input.strip().lower() in ("y", "n", "yes", "no"):
                if _last_conv_id[0] is not None:
                    rating = (
                        "positive"
                        if user_input.strip().lower() in ("y", "yes")
                        else "negative"
                    )
                    engine.memory_store.record_feedback(_last_conv_id[0], rating)
                    console.print("  [dim]noted.[/dim]\n")
                    _last_conv_id[0] = None
                continue

            # Built-in slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/exit", "/quit", "/q"):
                    console.print("[dim]Goodbye.[/dim]")
                    break
                elif cmd == "/help":
                    _print_help()
                    continue
                elif cmd == "/status":
                    _show_status()
                    continue
                elif cmd == "/clear":
                    console.clear()
                    _print_banner(effective_model)
                    _print_status_bar(effective_model)
                    continue
                elif cmd == "/new":
                    engine.conversation_id = engine.memory_store.new_conversation()
                    engine._history.clear()
                    _last_results.clear()
                    console.print("[dim]New conversation started.[/dim]\n")
                    continue
                elif cmd == "/model":
                    console.print(f"[dim]Current model: {effective_model}[/dim]\n")
                    continue
                elif cmd == "/briefing":
                    _show_briefing(db)
                    continue
                elif cmd == "/sources":
                    _repl_show_sources(_last_results)
                    continue
                elif cmd == "/docs":
                    _repl_show_docs()
                    continue
                elif cmd.startswith("/search"):
                    search_query = user_input[len("/search"):].strip()
                    if not search_query:
                        console.print("[yellow]Usage:[/yellow] /search <query>\n")
                    else:
                        _repl_search(search_query, metadata_store, vector_store)
                    continue
                else:
                    console.print(f"[yellow]Unknown command: {cmd}[/yellow] — type /help\n")
                    continue

            # --- Two-phase streaming response ---
            console.print()
            t_start = time.time()
            try:
                # Phase 1: retrieval with animated spinner status line
                _spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                _spin_idx = [0]

                def _search_render() -> Text:
                    dot = _spinner_chars[_spin_idx[0] % len(_spinner_chars)]
                    _spin_idx[0] += 1
                    return Text.assemble(("  " + dot + "  searching...", "dim"))

                with Live(
                    _search_render(),
                    console=console,
                    refresh_per_second=10,
                    transient=True,
                ) as live_search:
                    live_search.update(_search_render())
                    t_retrieval_start = time.time()
                    results, _classified = engine.retrieve(user_input)
                    t_retrieval = time.time() - t_retrieval_start

                # Store results for /sources command
                _last_results.clear()
                _last_results.extend(results)

                # Build deduped source name list
                _source_seen: set[str] = set()
                unique_sources: list[str] = []
                for r in results:
                    name = r.metadata.get("file_name", "")
                    if name and name not in _source_seen:
                        unique_sources.append(name)
                        _source_seen.add(name)

                console.print(
                    f"  [dim]searched  "
                    f"{len(results)} results \u00b7 {len(unique_sources)} files \u00b7 "
                    f"{t_retrieval:.1f}s[/dim]"
                )

                # Compute confidence from retrieval results
                from verra.agent.chat import ConfidenceLevel, compute_confidence
                confidence = compute_confidence(results)

                # Phase 2: streaming LLM response with live panel
                full_response: list[str] = []

                with Live(
                    Panel(
                        Text("...", style="dim"),
                        border_style="dim",
                        padding=(0, 1),
                    ),
                    console=console,
                    refresh_per_second=12,
                    vertical_overflow="visible",
                ) as live_gen:
                    for chunk in engine.stream_with_context(user_input, results):
                        full_response.append(chunk)
                        accumulated = "".join(full_response)
                        try:
                            content = Markdown(accumulated)
                        except Exception:
                            content = Text(accumulated)
                        live_gen.update(Panel(
                            content,
                            border_style="cyan",
                            padding=(0, 1),
                        ))

                elapsed = time.time() - t_start
                answer_text = "".join(full_response)

                # Build inline citation legend: map [N] references in the response
                # to actual filenames from retrieval results.
                import re as _re_cite
                cited_nums = {int(m) for m in _re_cite.findall(r'\[(\d+)\]', answer_text)}
                legend_parts: list[str] = []
                for _ci, _cr in enumerate(results, 1):
                    if _ci in cited_nums:
                        _fname = (_cr.metadata.get("file_name", "")
                                  or _cr.metadata.get("subject", "")
                                  or _cr.metadata.get("source_type", f"source {_ci}"))
                        legend_parts.append(f"[{_ci}] {_fname}")

                # Confidence badge + source footer below the response panel
                _confidence_badges = {
                    ConfidenceLevel.HIGH:   "[green]\u25a0[/green]",
                    ConfidenceLevel.MEDIUM: "[yellow]\u25a0[/yellow]",
                    ConfidenceLevel.LOW:    "[red]\u25a0[/red]",
                }
                badge = _confidence_badges.get(confidence, "")

                if legend_parts:
                    # Numbered citation legend — only sources actually cited in text
                    legend_str = " \u00b7 ".join(legend_parts)
                    if badge:
                        console.print(
                            f"  {badge} [dim]{confidence.value} confidence \u00b7 {legend_str}[/dim]"
                        )
                    else:
                        console.print(f"  [dim]{legend_str}[/dim]")
                elif unique_sources:
                    # Fallback: plain source list when no [N] markers were used
                    sources_str = " \u00b7 ".join(unique_sources[:5])
                    if len(unique_sources) > 5:
                        sources_str += f" \u00b7 +{len(unique_sources) - 5} more"
                    if badge:
                        console.print(
                            f"  {badge} [dim]{confidence.value} confidence \u00b7 {sources_str}[/dim]"
                        )
                    else:
                        console.print(f"  [dim]{sources_str}[/dim]")
                elif badge:
                    console.print(f"  {badge} [dim]{confidence.value} confidence[/dim]")

                console.print(
                    f"  [dim]{len(results)} chunks \u00b7 "
                    f"{len(unique_sources)} sources \u00b7 "
                    f"{elapsed:.1f}s[/dim]"
                )

                # Feedback prompt — user can type y/n as their next input
                _last_conv_id[0] = engine.conversation_id
                console.print("  [dim]helpful? [y/n][/dim]")
                console.print()

            except RuntimeError as exc:
                console.print(Panel(
                    f"[red]{exc}[/red]",
                    border_style="red",
                    padding=(0, 1),
                ))
            except Exception as exc:
                console.print(Panel(
                    f"[red]{exc}[/red]",
                    border_style="red",
                    padding=(0, 1),
                ))
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
    finally:
        db.close()


def _print_help() -> None:
    """Print in-REPL help."""
    from rich.columns import Columns

    console.print()
    left = (
        "[cyan]/search[/cyan] [dim]<query>[/dim]  search without LLM\n"
        "[cyan]/sources[/cyan]         sources from last answer\n"
        "[cyan]/docs[/cyan]            browse ingested documents\n"
        "[cyan]/briefing[/cyan]        actionable insights\n"
        "[cyan]/status[/cyan]          ingestion stats\n"
        "[cyan]/model[/cyan]           current model"
    )
    right = (
        "[cyan]/new[/cyan]             new conversation\n"
        "[cyan]/clear[/cyan]           clear screen\n"
        "[cyan]/help[/cyan]            this help\n"
        "[cyan]/exit[/cyan]            quit"
    )
    console.print(Panel(
        Columns([left, right], padding=(0, 4)),
        title="[dim]commands[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print("  [dim]Enter sends your message. Escape+Enter for newline.[/dim]")
    console.print()


def _show_briefing(db: Any) -> None:
    """Show briefing items in the REPL."""
    from verra.briefing.detector import BriefingDetector
    from verra.config import load_config

    config = load_config()
    detector = BriefingDetector(
        core_conn=db.core,
        analysis_conn=db.analysis,
        config=config.briefing,
    )
    items = detector.detect_all()

    if not items:
        console.print("  [dim]No actionable insights right now.[/dim]\n")
        return

    console.print()
    urgency_icons = {5: "[red]!![/red]", 4: "[yellow]![/yellow]", 3: "[cyan]·[/cyan]", 2: "[dim]·[/dim]", 1: "[dim]·[/dim]"}
    for item in items:
        icon = urgency_icons.get(item.urgency, "·")
        console.print(f"  {icon} [bold]{item.title}[/bold]")
        console.print(f"    [dim]{item.detail}[/dim]")
    console.print()


def _show_status() -> None:
    """Print a quick status summary (used by /status in REPL and verra status)."""
    import sqlite3

    from verra.config import VERRA_HOME, ensure_data_dir, load_config

    ensure_data_dir()
    config = load_config()
    core_db = VERRA_HOME / "core.db"

    doc_count = 0
    chunk_count = 0
    entity_count = 0

    if core_db.exists():
        conn = sqlite3.connect(str(core_db))
        try:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        except Exception:
            pass
        finally:
            conn.close()

    disk_bytes = _disk_usage(VERRA_HOME)

    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="dim")
    table.add_column("")
    table.add_row("Documents", str(doc_count))
    table.add_row("Chunks", str(chunk_count))
    table.add_row("Entities", str(entity_count))
    table.add_row("Disk", _fmt_bytes(disk_bytes))
    table.add_row("Model", config.agent.model)
    table.add_row("Sources", str(len(config.sources)))
    console.print(table)
    console.print()


def _repl_search(query: str, metadata_store: Any, vector_store: Any) -> None:
    """Run a search from within the REPL and display results inline."""
    from verra.retrieval.router import parse_query
    from verra.retrieval.search import search as do_search

    classified = parse_query(query)
    try:
        results = do_search(classified, metadata_store, vector_store, n_results=5)
    except Exception as exc:
        console.print(f"  [red]Search error:[/red] {exc}\n")
        return

    if not results:
        console.print("  [dim]No results found.[/dim]\n")
        return

    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Excerpt")

    for i, result in enumerate(results, 1):
        source = result.metadata.get("file_path") or result.metadata.get("source_type") or ""
        if source and len(source) > 30:
            source = "..." + source[-27:]
        excerpt = result.text.replace("\n", " ").strip()
        if len(excerpt) > 80:
            excerpt = excerpt[:77] + "..."
        table.add_row(str(i), f"{result.score:.2f}", source, excerpt)

    console.print(table)
    console.print()


def _repl_show_sources(results: list[Any]) -> None:
    """Display the source files from the last retrieval in the REPL."""
    if not results:
        console.print("  [dim]No sources yet — ask a question first.[/dim]\n")
        return

    seen: set[str] = set()
    sources: list[tuple[str, str, float]] = []
    for r in results:
        name = r.metadata.get("file_name", "")
        path = r.metadata.get("file_path", "")
        score = r.score
        label = name or path or r.metadata.get("source_type", "unknown")
        if label not in seen:
            seen.add(label)
            sources.append((label, path, score))

    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", justify="right")
    table.add_column("File")
    table.add_column("Score", justify="right", style="dim")

    for i, (label, _path, score) in enumerate(sources, 1):
        table.add_row(str(i), label, f"{score:.2f}")

    console.print(table)
    console.print()


def _show_docs(limit: int = 30, fmt: str | None = None) -> None:
    """Display ingested documents as a Rich table. Shared by REPL and CLI."""
    import sqlite3

    from verra.config import VERRA_HOME

    core_db = VERRA_HOME / "core.db"
    if not core_db.exists():
        console.print("  [dim]No data indexed yet.[/dim]\n")
        return

    try:
        conn = sqlite3.connect(str(core_db))
        conn.row_factory = sqlite3.Row

        where_clause = "WHERE d.format = ?" if fmt else ""
        params: tuple = (fmt,) if fmt else ()
        if fmt:
            params = (fmt, limit)
        else:
            params = (limit,)

        rows = conn.execute(
            f"""
            SELECT
                d.file_name,
                d.format,
                d.source_type,
                d.indexed_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            {where_clause}
            GROUP BY d.id
            ORDER BY d.indexed_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        total_docs = conn.execute(
            "SELECT COUNT(*) FROM documents" + (" WHERE format = ?" if fmt else ""),
            (fmt,) if fmt else (),
        ).fetchone()[0]

        total_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

        conn.close()
    except Exception as exc:
        console.print(f"  [red]Error reading documents:[/red] {exc}\n")
        return

    if not rows:
        console.print("  [dim]No documents found.[/dim]\n")
        return

    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Name")
    table.add_column("Format", style="dim", justify="center")
    table.add_column("Chunks", justify="right", style="dim")
    table.add_column("Ingested", style="dim")

    for row in rows:
        ingested = (row["indexed_at"] or "")[:10]  # YYYY-MM-DD
        table.add_row(
            row["file_name"],
            (row["format"] or "?").upper(),
            str(row["chunk_count"]),
            ingested,
        )

    console.print(table)

    disk_bytes = _disk_usage(VERRA_HOME)
    console.print(
        f"\n  [dim]{total_docs} documents \u00b7 {total_chunks} chunks \u00b7 {_fmt_bytes(disk_bytes)}[/dim]"
    )
    console.print()


def _repl_show_docs() -> None:
    """Show ingested documents from within the REPL."""
    _show_docs(limit=30)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="verra")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Verra One -- chat with your business data."""
    _check_for_updates()
    if ctx.invoked_subcommand is None:
        from verra.config import VERRA_HOME

        config_path = VERRA_HOME / "config.yaml"
        if not config_path.exists():
            console.print(
                "[yellow]Verra isn't set up yet.[/yellow] "
                "Running setup...\n"
            )
            ctx.invoke(setup)
            console.print()
        _run_chat_repl()




@main.command()
def setup() -> None:
    """Step-by-step setup: model, API key, and data sources."""
    import json
    import urllib.request

    from verra.config import SourceConfig, ensure_data_dir, load_config, save_config

    console.print()
    console.print("  [bold cyan]Verra One Setup[/bold cyan]")
    console.print("  [dim]Answer a few questions to get started.[/dim]")
    console.print()

    config = load_config()

    # ------------------------------------------------------------------
    # Step 1: Model provider
    # ------------------------------------------------------------------
    console.print("  [bold]Step 1:[/bold] Choose your LLM provider")
    console.print()
    console.print("    [dim]1.[/dim] Ollama         [dim](local, free, private — runs on your machine)[/dim]")
    console.print("    [dim]2.[/dim] Anthropic       [dim](Claude — requires API key, costs per token)[/dim]")
    console.print("    [dim]3.[/dim] OpenAI          [dim](GPT — requires API key, costs per token)[/dim]")
    console.print("    [dim]4.[/dim] Custom          [dim](any LiteLLM-compatible model string)[/dim]")
    console.print()

    provider = click.prompt("  Select provider", default="1")

    if provider == "1":
        # --- Ollama flow ---
        # Check if Ollama is running
        ollama_running = False
        ollama_models: list[str] = []
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
                ollama_models = [m["name"] for m in data.get("models", [])]
                ollama_running = True
        except Exception:
            pass

        if not ollama_running:
            console.print("  [yellow]Ollama is not running on localhost:11434.[/yellow]")
            console.print()
            console.print("  To install Ollama:")
            console.print("    [dim]curl -fsSL https://ollama.com/install.sh | sh[/dim]")
            console.print("    [dim]ollama serve[/dim]")
            console.print()
            if not click.confirm("  Continue with Ollama anyway? (you can start it later)", default=True):
                console.print("  [dim]Re-run 'verra setup' when ready.[/dim]")
                return
            console.print()
            _print_model_recommendations()
            rec_choice = click.prompt("  Select model", default="1")
            rec_models = {"1": "llama3.1:8b", "2": "llama3.2:3b", "3": "qwen2.5:7b",
                          "4": "gemma2:9b", "5": "phi4:14b", "6": "llama3.1:70b"}
            picked = rec_models.get(rec_choice, rec_choice)
            config.agent.model = f"ollama/{picked}"
            console.print(f"  [green]→[/green] {config.agent.model}")
            console.print(f"  [dim]Pull with: ollama pull {picked}[/dim]")
        else:
            console.print(f"  [green]Ollama is running.[/green]")
            if ollama_models:
                console.print(f"  Installed models:")
                for i, m in enumerate(ollama_models[:10], 1):
                    console.print(f"    [dim]{i}.[/dim] {m}")
                console.print(f"    [dim]{len(ollama_models) + 1}.[/dim] Pull a recommended model")
                console.print(f"    [dim]{len(ollama_models) + 2}.[/dim] Enter a custom model name")
                console.print()
                model_choice = click.prompt("  Select model", default="1")
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(ollama_models):
                        config.agent.model = f"ollama/{ollama_models[idx]}"
                    elif idx == len(ollama_models):
                        # Show recommendations
                        _print_model_recommendations()
                        rec_choice = click.prompt("  Select model", default="1")
                        rec_models = {"1": "llama3.1:8b", "2": "llama3.2:3b", "3": "qwen2.5:7b",
                                      "4": "gemma2:9b", "5": "phi4:14b", "6": "llama3.1:70b"}
                        picked = rec_models.get(rec_choice, rec_choice)
                        config.agent.model = f"ollama/{picked}"
                        console.print(f"  [dim]Pull with: ollama pull {picked}[/dim]")
                    else:
                        custom = click.prompt("  Model name")
                        config.agent.model = f"ollama/{custom}"
                except ValueError:
                    config.agent.model = f"ollama/{model_choice}"
            else:
                console.print("  [yellow]No models installed yet.[/yellow]")
                _print_model_recommendations()
                rec_choice = click.prompt("  Select model", default="1")
                rec_models = {"1": "llama3.1:8b", "2": "llama3.2:3b", "3": "qwen2.5:7b",
                              "4": "gemma2:9b", "5": "phi4:14b", "6": "llama3.1:70b"}
                picked = rec_models.get(rec_choice, rec_choice)
                config.agent.model = f"ollama/{picked}"
                console.print(f"  [dim]Pull with: ollama pull {picked}[/dim]")
            console.print(f"  [green]→[/green] {config.agent.model}")
        console.print()

    elif provider == "2":
        config.agent.model = "claude-3-5-haiku-20241022"
        console.print(f"  [green]→[/green] {config.agent.model}")
        console.print()

    elif provider == "3":
        config.agent.model = "gpt-4o-mini"
        console.print(f"  [green]→[/green] {config.agent.model}")
        console.print()

    elif provider == "4":
        config.agent.model = click.prompt("  Enter model string (LiteLLM format)")
        console.print(f"  [green]→[/green] {config.agent.model}")
        console.print()

    else:
        config.agent.model = provider
        console.print(f"  [green]→[/green] {config.agent.model}")
        console.print()

    # ------------------------------------------------------------------
    # Step 2: API key (if cloud provider)
    # ------------------------------------------------------------------
    model_lower = config.agent.model.lower()
    is_ollama = "ollama" in model_lower
    needs_key = any(kw in model_lower for kw in ("claude", "anthropic", "gpt", "openai"))

    if needs_key:
        console.print("  [bold]Step 2:[/bold] API Key")
        console.print()
        existing_key = config.agent.api_key or ""
        hint = " [dim](press Enter to keep existing)[/dim]" if existing_key else ""
        api_key_input = click.prompt(
            f"  API Key{hint}",
            default=existing_key or "",
            hide_input=True,
            prompt_suffix=": ",
        ).strip()

        if api_key_input:
            config.agent.api_key = api_key_input
            # Validate the key
            console.print("  [dim]Validating...[/dim]", end="")
            try:
                from verra.agent.llm import LLMClient

                test_llm = LLMClient(model=config.agent.model)
                # Set the env var so LiteLLM can find it
                if "anthropic" in model_lower or "claude" in model_lower:
                    os.environ["ANTHROPIC_API_KEY"] = api_key_input
                elif "gpt" in model_lower or "openai" in model_lower:
                    os.environ["OPENAI_API_KEY"] = api_key_input
                test_llm.complete([{"role": "user", "content": "hi"}])
                console.print("\r  [green]→[/green] API key valid and saved           ")
            except Exception as exc:
                console.print(f"\r  [yellow]→[/yellow] Key saved but validation failed: {str(exc)[:60]}  ")
                console.print("    [dim]You can fix this later and re-run 'verra setup'.[/dim]")
        elif existing_key:
            console.print("  [green]→[/green] Using existing key")
        else:
            console.print("  [yellow]→[/yellow] No key set — set via env var later")
        console.print()
    elif is_ollama:
        console.print("  [dim]Step 2: API Key — not needed for local Ollama models[/dim]")
        console.print("  [dim]  Your data stays 100% on your machine.[/dim]")
        console.print()
    else:
        console.print("  [dim]Step 2: API Key — not needed for this model[/dim]")
        console.print()

    # ------------------------------------------------------------------
    # Step 3: Data sources (loop — add multiple)
    # ------------------------------------------------------------------
    console.print("  [bold]Step 3:[/bold] Add data sources")
    console.print()
    console.print("    [dim]Your data is never sent to third parties. Local models keep everything on your machine.[/dim]")
    console.print("    [dim]Cloud models (Claude/OpenAI) send query context to the LLM API.[/dim]")
    console.print()

    folder_paths: list[Path] = []

    while True:
        console.print("    [dim]1.[/dim] Add a local folder")
        console.print("    [dim]2.[/dim] Connect Gmail       [dim](run 'verra gmail <email>' after setup)[/dim]")
        console.print("    [dim]3.[/dim] Connect Google Drive [dim](run 'verra drive <email>' after setup)[/dim]")
        console.print("    [dim]4.[/dim] Connect Outlook      [dim](run 'verra outlook <email>' after setup)[/dim]")
        console.print("    [dim]5.[/dim] Done — skip / finish adding sources")
        console.print()

        src_choice = click.prompt("  Select", default="5")

        if src_choice == "1":
            folder = click.prompt("  Folder path", type=click.Path(exists=True))
            folder_path = Path(folder).resolve()
            existing_paths = {s.path for s in config.sources if s.type == "folder"}
            if str(folder_path) not in existing_paths:
                config.sources.append(SourceConfig(type="folder", path=str(folder_path)))
                folder_paths.append(folder_path)
            console.print(f"  [green]→[/green] Added: {folder_path}")
            console.print()
            if not click.confirm("  Add another source?", default=False):
                break
            console.print()
        elif src_choice in ("2", "3", "4"):
            source_names = {"2": "Gmail", "3": "Google Drive", "4": "Outlook"}
            source_cmds = {"2": "verra gmail <email>", "3": "verra drive <email>", "4": "verra outlook <email>"}
            console.print(f"  [dim]{source_names[src_choice]} requires OAuth. Run after setup:[/dim]")
            console.print(f"    [bold]{source_cmds[src_choice]}[/bold]")
            console.print()
            if not click.confirm("  Add another source?", default=False):
                break
            console.print()
        else:
            break

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    ensure_data_dir()
    save_config(config)

    console.print()
    console.print("  [bold green]Setup complete.[/bold green]")
    console.print(f"  Config saved to [dim]~/.verra/config.yaml[/dim]")
    console.print()

    # Offer to ingest folder sources
    if folder_paths:
        should_ingest = click.confirm("  Ingest documents now?", default=True)
        if should_ingest:
            for fp in folder_paths:
                console.print()
                ctx = click.get_current_context()
                ctx.invoke(ingest, folder_path=fp, force=False, mode="realtime")
    else:
        console.print("  [dim]Next steps:[/dim]")
        console.print("    [dim]verra ingest <folder>    — add documents[/dim]")
        console.print("    [dim]verra gmail <email>      — connect Gmail[/dim]")
        console.print("    [dim]verra                    — start chatting[/dim]")




@main.command()
@click.option("--json-output", "as_json", is_flag=True, help="Output stats as JSON.")
def status(as_json: bool) -> None:
    """Show what's ingested, entity counts, sync state, and disk usage."""
    import json
    import sqlite3

    from verra.config import VERRA_HOME, ensure_data_dir, load_config

    ensure_data_dir()
    config = load_config()

    core_db = VERRA_HOME / "core.db"
    analysis_db = VERRA_HOME / "analysis.db"

    doc_count = 0
    chunk_count = 0
    format_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    relationship_count = 0

    if core_db.exists():
        conn = sqlite3.connect(str(core_db))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT format, COUNT(*) AS cnt FROM documents GROUP BY format"
            ).fetchall()
            for r in rows:
                fmt = r["format"] or "unknown"
                fmt_count = r["cnt"]
                format_counts[fmt] = fmt_count
                doc_count += fmt_count

            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

            type_rows = conn.execute(
                "SELECT entity_type, COUNT(*) AS cnt FROM entities GROUP BY entity_type"
            ).fetchall()
            for r in type_rows:
                entity_counts[r["entity_type"]] = r["cnt"]

            relationship_count = conn.execute(
                "SELECT COUNT(*) FROM relationships"
            ).fetchone()[0]
        except Exception:
            pass
        finally:
            conn.close()

    commitment_count = 0
    conflict_count = 0
    gap_count = 0

    if analysis_db.exists():
        conn = sqlite3.connect(str(analysis_db))
        conn.row_factory = sqlite3.Row
        try:
            commitment_count = conn.execute(
                "SELECT COUNT(*) FROM commitments WHERE status = 'open'"
            ).fetchone()[0]
            conflict_count = conn.execute(
                "SELECT COUNT(*) FROM conflicts WHERE resolved = 0"
            ).fetchone()[0]
            gap_count = conn.execute(
                "SELECT COUNT(*) FROM knowledge_gaps WHERE resolved = 0"
            ).fetchone()[0]
        except Exception:
            pass
        finally:
            conn.close()

    # Sync state
    sync_states: list[dict[str, str]] = []
    if core_db.exists():
        conn = sqlite3.connect(str(core_db))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT source, last_sync_at, status FROM sync_state"
            ).fetchall()
            for r in rows:
                sync_states.append(dict(r))
        except Exception:
            pass
        finally:
            conn.close()

    disk_bytes = _disk_usage(VERRA_HOME)

    # API key status — check provider-specific env vars separately from the
    # generic config key, so we don't misleadingly show both as "set".
    model_lower = config.agent.model.lower()
    is_anthropic_model = "anthropic" in model_lower or "claude" in model_lower
    is_openai_model = "gpt" in model_lower or "openai" in model_lower
    anthropic_key_set = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or (config.agent.api_key and is_anthropic_model)
    )
    openai_key_set = bool(
        os.environ.get("OPENAI_API_KEY")
        or (config.agent.api_key and is_openai_model)
    )

    # --- JSON output ---
    if as_json:
        data = {
            "documents": doc_count,
            "documents_by_format": format_counts,
            "chunks": chunk_count,
            "entities": entity_counts,
            "relationships": relationship_count,
            "open_commitments": commitment_count,
            "unresolved_conflicts": conflict_count,
            "knowledge_gaps": gap_count,
            "disk_bytes": disk_bytes,
            "disk_human": _fmt_bytes(disk_bytes),
            "model": config.agent.model,
            "anthropic_api_key_set": anthropic_key_set,
            "openai_api_key_set": openai_key_set,
            "sync_states": sync_states,
        }
        click.echo(json.dumps(data, indent=2))
        return

    # --- Render ---
    console.print()
    console.print("  [bold cyan]Verra One Status[/bold cyan]")
    console.print()

    stat_table = Table(show_header=False, box=None, padding=(0, 2))
    stat_table.add_column("", style="dim")
    stat_table.add_column("")

    if format_counts:
        for fmt, cnt in sorted(format_counts.items()):
            stat_table.add_row(f"Documents ({fmt.upper()})", str(cnt))
    else:
        stat_table.add_row("Documents", "0")

    stat_table.add_row("Total chunks", str(chunk_count))

    if entity_counts:
        for etype, cnt in sorted(entity_counts.items()):
            stat_table.add_row(f"Entities ({etype})", str(cnt))
    else:
        stat_table.add_row("Entities", "0")
    stat_table.add_row("Relationships", str(relationship_count))

    stat_table.add_row("Open commitments", str(commitment_count))
    stat_table.add_row("Unresolved conflicts", str(conflict_count))
    stat_table.add_row("Knowledge gaps", str(gap_count))

    stat_table.add_row("Disk usage", _fmt_bytes(disk_bytes))
    stat_table.add_row("Current model", config.agent.model)
    stat_table.add_row(
        "ANTHROPIC_API_KEY",
        "[green]set[/green]" if anthropic_key_set else "[dim]not set[/dim]",
    )
    stat_table.add_row(
        "OPENAI_API_KEY",
        "[green]set[/green]" if openai_key_set else "[dim]not set[/dim]",
    )

    console.print(stat_table)

    if sync_states:
        console.print()
        sync_table = Table(
            title="Sync State",
            show_header=True,
            header_style="bold magenta",
        )
        sync_table.add_column("Source", style="cyan")
        sync_table.add_column("Last Sync")
        sync_table.add_column("Status", style="green")
        for s in sync_states:
            status_color = "green" if s.get("status") == "idle" else "yellow"
            sync_table.add_row(
                s.get("source", ""),
                s.get("last_sync_at", "never"),
                f"[{status_color}]{s.get('status', '?')}[/{status_color}]",
            )
        console.print(sync_table)

    if doc_count == 0:
        console.print(
            "\n  [yellow]No data indexed yet.[/yellow] "
            "Run [bold]verra ingest <folder>[/bold] to get started."
        )
    console.print()


@main.command(name="info")
def info() -> None:
    """Alias for 'verra status'."""
    ctx = click.get_current_context()
    ctx.invoke(status)



@main.command(name="mcp")
def mcp_server() -> None:
    """Start the MCP server (stdio) for use with Claude Desktop and other AI clients."""
    from verra.mcp_server import run_stdio_server

    run_stdio_server()


@main.command(name="mcp-config")
def mcp_config() -> None:
    """Print the Claude Desktop MCP configuration for Verra One."""
    import json
    import shutil

    verra_path = shutil.which("verra") or "verra"
    config = {
        "mcpServers": {
            "verra-one": {
                "command": verra_path,
                "args": ["mcp"],
            }
        }
    }
    console.print()
    console.print("  [bold cyan]Claude Desktop MCP Configuration[/bold cyan]")
    console.print()
    console.print("  Add this to your Claude Desktop config:")
    console.print(
        "  [dim](~/Library/Application Support/Claude/claude_desktop_config.json)[/dim]"
    )
    console.print()
    console.print(json.dumps(config, indent=2))
    console.print()

@main.command(name="docs")
@click.option("--limit", default=30, show_default=True, help="Max documents to show.")
@click.option("--format-filter", "fmt", default=None, help="Filter by format: pdf, txt, csv")
def docs(limit: int, fmt: str | None) -> None:
    """Browse ingested documents."""
    from verra.config import ensure_data_dir

    ensure_data_dir()
    _show_docs(limit=limit, fmt=fmt)


@main.command()
def update() -> None:
    """Update Verra One to the latest version."""
    import subprocess

    console.print()
    console.print("  Checking for updates...")
    try:
        result = subprocess.run(
            ["pipx", "upgrade", "verra-one"], capture_output=True, text=True
        )
        if result.returncode == 0:
            msg = result.stdout.strip() or result.stderr.strip()
            console.print(f"  [green]{msg}[/green]")
        else:
            # pipx failed or not installed — fall back to pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", "verra-one"],
                capture_output=True,
                text=True,
            )
            msg = result.stdout.strip() or result.stderr.strip()
            console.print(f"  {msg}")
    except FileNotFoundError:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "verra-one"],
            capture_output=True,
            text=True,
        )
        msg = result.stdout.strip() or result.stderr.strip()
        console.print(f"  {msg}")
    console.print()


@main.command()
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Re-index files even if already processed.")
@click.option(
    "--mode",
    type=click.Choice(["fast", "realtime", "deep"]),
    default="realtime",
    help="Analysis mode.",
)
@click.option("--dry-run", is_flag=True, help="Preview what would be ingested without writing anything.")
def ingest(folder_path: Path, force: bool, mode: str, dry_run: bool) -> None:
    """Ingest all documents from FOLDER_PATH into the Verra One knowledge base."""
    import collections

    if dry_run:
        from verra.ingest.extractors import SUPPORTED_EXTENSIONS

        console.print()
        console.print(f"  [bold cyan]Dry run:[/bold cyan] {folder_path}")
        console.print("  [dim]No data will be written.[/dim]")
        console.print()

        ext_counts: dict[str, int] = collections.defaultdict(int)
        ext_bytes: dict[str, int] = collections.defaultdict(int)
        total_files = 0
        total_bytes = 0
        skipped = 0

        for dirpath, _dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                fp = Path(dirpath) / filename
                if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    skipped += 1
                    continue
                try:
                    size = fp.stat().st_size
                except OSError:
                    skipped += 1
                    continue
                ext = fp.suffix.lower()
                ext_counts[ext] += 1
                ext_bytes[ext] += size
                total_files += 1
                total_bytes += size

        # Rough estimate: ~200 tokens per chunk, ~4 chars per token -> ~800 chars per chunk
        est_chunks = max(1, total_bytes // 800)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        for ext, cnt in sorted(ext_counts.items()):
            table.add_row(f"Files ({ext})", f"{cnt}  ({_fmt_bytes(ext_bytes[ext])})")
        table.add_row("Total files", str(total_files))
        table.add_row("Total size", _fmt_bytes(total_bytes))
        table.add_row("Est. chunks", f"~{est_chunks}")
        table.add_row("Unsupported (skipped)", str(skipped))
        console.print(table)
        console.print()
        console.print("  [dim]Run without --dry-run to ingest.[/dim]")
        console.print()
        return

    from rich.live import Live
    from rich.text import Text as RichText

    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.ingest.pipeline import IngestPhase, ingest_folder
    from verra.store.analysis import AnalysisStore
    from verra.store.db import DatabaseManager
    from verra.store.entities import EntityStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    _apply_api_key_from_config()
    ensure_data_dir()
    config = load_config()

    console.print()
    console.print(f"  [bold cyan]Ingesting:[/bold cyan] {folder_path}")
    console.print(f"  [dim]Model: {config.agent.model} | Analysis: {mode}[/dim]")
    console.print()

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    entity_store = EntityStore.from_connection(db.core)
    analysis_store = AnalysisStore.from_connection(db.analysis)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    tabular_store = None
    try:
        from verra.store.tabular import TabularStore

        tabular_store = TabularStore(VERRA_HOME / "tabular.db")
    except Exception:
        pass

    # -- live state for the renderer --
    _live_state: dict[str, Any] = {
        "phase": "scan",
        "file": "",
        "detail": "",
        "files_done": 0,
        "files_total": 0,
        "chunks": 0,
        "entities": 0,
        "errors": 0,
        "t0": time.time(),
        "file_times": [],      # seconds-per-file ring buffer for ETA
    }
    _last_file_t = [time.time()]

    def _fmt_eta(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s:02d}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m = rem // 60
            return f"{h}h {m:02d}m"

    def _render() -> RichText:
        s = _live_state
        done = s["files_done"]
        total = s["files_total"]
        elapsed = time.time() - s["t0"]

        lines: list[str] = []

        # Line 1: progress bar
        if total > 0:
            pct = done / total
            bar_w = 30
            filled = int(pct * bar_w)
            bar = "━" * filled + "[dim]" + "━" * (bar_w - filled) + "[/dim]"
            pct_str = f"{pct * 100:5.1f}%"

            # ETA from rolling average
            times = s["file_times"]
            if len(times) >= 3 and done < total:
                avg = sum(times[-20:]) / len(times[-20:])
                eta = avg * (total - done)
                eta_str = f"~{_fmt_eta(eta)} left"
            elif done >= total:
                eta_str = "done"
            else:
                eta_str = "estimating..."

            lines.append(f"  {bar} {pct_str}  [dim]{done}[/dim]/{total} files  [dim]{eta_str}[/dim]")
        else:
            lines.append(f"  [dim]scanning...[/dim]")

        # Line 2: current file + what we're doing to it
        phase_labels = {
            "scan": "scanning",
            "hash": "hashing",
            "skip": "skipped",
            "extract": "reading",
            "chunk": "chunking",
            "dedup": "dedup",
            "refs": "references",
            "embed": "embedding",
            "entities": "extracting entities",
            "analyse": "analysing",
            "done": "done",
            "error": "error",
        }
        fname = s["file"]
        if len(fname) > 45:
            fname = "..." + fname[-42:]
        phase_label = phase_labels.get(s["phase"], s["phase"])
        detail = s["detail"]
        if s["phase"] == "error":
            lines.append(f"  [red]>[/red] {fname}  [red]{phase_label}[/red]  [dim]{detail}[/dim]")
        elif s["phase"] == "skip":
            lines.append(f"  [dim]> {fname}  {phase_label}[/dim]")
        elif s["phase"] == "done":
            lines.append(f"  [green]>[/green] {fname}  [dim]{detail}[/dim]")
        else:
            lines.append(f"  [cyan]>[/cyan] {fname}  [dim]{phase_label}[/dim]  [dim]{detail}[/dim]")

        # Line 3: running counters
        elapsed_str = _fmt_eta(elapsed)
        chunks = s["chunks"]
        ents = s["entities"]
        errs = s["errors"]

        parts = [f"[dim]{elapsed_str} elapsed[/dim]"]
        parts.append(f"[dim]{chunks:,} chunks[/dim]")
        parts.append(f"[dim]{ents:,} entities[/dim]")
        if errs > 0:
            parts.append(f"[yellow]{errs} errors[/yellow]")
        lines.append("  " + "  ".join(parts))

        return console.render_str("\n".join(lines))

    def on_phase(phase: IngestPhase) -> None:
        now = time.time()
        s = _live_state
        s["phase"] = phase.phase
        s["file"] = phase.file_path.name if phase.file_path else ""
        s["detail"] = phase.detail
        s["files_done"] = phase.files_done
        s["files_total"] = phase.files_total
        s["chunks"] = phase.chunks_so_far
        s["entities"] = phase.entities_so_far
        s["errors"] = phase.errors_so_far

        # Track per-file time for ETA (record on each "done" or "skip")
        if phase.phase in ("done", "skip"):
            dt = now - _last_file_t[0]
            if dt > 0.001:  # ignore sub-ms noise
                s["file_times"].append(dt)
                # Keep last 50 for rolling average
                if len(s["file_times"]) > 50:
                    s["file_times"] = s["file_times"][-50:]
            _last_file_t[0] = now

    def on_progress(file_path: Path, current: int, total: int) -> None:
        pass  # phase_callback handles everything

    with Live(_render(), console=console, refresh_per_second=8) as live:

        def on_phase_with_render(phase: IngestPhase) -> None:
            on_phase(phase)
            live.update(_render())

        stats = ingest_folder(
            folder_path=folder_path,
            metadata_store=metadata_store,
            vector_store=vector_store,
            entity_store=entity_store,
            analysis_store=analysis_store,
            tabular_store=tabular_store,
            analysis_mode=mode,
            force_reindex=force,
            progress_callback=on_progress,
            phase_callback=on_phase_with_render,
        )

    # Final summary
    console.print()
    elapsed_str = _fmt_eta(stats.elapsed_seconds)
    console.print(f"  [bold green]Ingestion complete.[/bold green]  {elapsed_str}")
    console.print()

    console.print(f"  [dim]files[/dim]      {stats.files_processed} processed, {stats.files_skipped} skipped")
    console.print(f"  [dim]chunks[/dim]     {stats.chunks_created:,}")
    console.print(f"  [dim]vectors[/dim]    {vector_store.count():,}")
    console.print(f"  [dim]entities[/dim]   {stats.entities_found:,}")
    console.print(f"  [dim]relations[/dim]  {stats.relationships_found:,}")

    if stats.errors:
        console.print()
        console.print(f"  [yellow]{len(stats.errors)} warnings:[/yellow]")
        for err in stats.errors[:5]:
            short = err if len(err) < 80 else err[:77] + "..."
            console.print(f"    [dim]{short}[/dim]")
        if len(stats.errors) > 5:
            console.print(f"    [dim]... and {len(stats.errors) - 5} more[/dim]")

    console.print()
    db.close()




@main.command()
@click.argument("ssh_target")
@click.option("--port", default=22, show_default=True, help="SSH port.")
@click.option("--model", default="llama3.2", show_default=True, help="Ollama model to install.")
@click.option("--verra-port", default=8484, show_default=True, help="Port to expose Verra One API on.")
def deploy(ssh_target: str, port: int, model: str, verra_port: int) -> None:
    """Deploy Verra One to a remote server via SSH.

    SSH_TARGET: user@hostname  or just hostname (defaults to ubuntu@host).

    Examples:

    \b
        verra deploy 192.168.1.10
        verra deploy ubuntu@my-server.example.com
        verra deploy ubuntu@my-server.example.com --model mistral --verra-port 9090
    """
    from verra.deploy.ssh import deploy_remote

    if "@" in ssh_target:
        user, host = ssh_target.split("@", 1)
    else:
        user, host = "ubuntu", ssh_target

    console.print()
    console.print(f"  [bold cyan]Deploying Verra One[/bold cyan] → {user}@{host}:{port}")
    console.print(f"  [dim]Model: {model}   API port: {verra_port}[/dim]")
    console.print()

    def log(msg: str) -> None:
        console.print(f"  [cyan]→[/cyan] {msg}")

    try:
        result = deploy_remote(
            host=host,
            user=user,
            port=port,
            model=model,
            verra_port=verra_port,
            log=log,
        )
        console.print("\n  [green]Verra One deployed successfully.[/green]")
        console.print(f"  Access:  http://{host}:{result.get('port', verra_port)}")
        console.print(f"  Status:  {result.get('status', 'ok')}")
    except RuntimeError as exc:
        console.print(f"\n  [red]Deploy failed:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n  [red]Unexpected error:[/red] {exc}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    console.print()




@main.command()
@click.argument("gmail_account")
@click.option("--since", default=None, help="Only ingest emails after this date (YYYY-MM-DD).")
@click.option("--labels", multiple=True, help="Gmail label IDs to filter (can be repeated).")
@click.option("--max-results", default=500, show_default=True, help="Max messages to fetch.")
def gmail(
    gmail_account: str,
    since: str | None,
    labels: tuple[str, ...],
    max_results: int,
) -> None:
    """Connect and ingest emails from a Gmail account.

    GMAIL_ACCOUNT: your Gmail address (e.g. you@gmail.com).

    On first run this opens a browser for OAuth consent.  Subsequent runs
    reuse the cached token at ~/.verra/oauth/<account>_token.json.
    """
    from verra.config import VERRA_HOME, SourceConfig, ensure_data_dir, load_config, save_config
    from verra.ingest.gmail import GmailIngestor, ingest_gmail
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()

    console.print()
    console.print(f"  [bold cyan]Gmail Ingest[/bold cyan]  {gmail_account}")
    console.print(f"  [dim]since={since or 'all time'}[/dim]")
    console.print()

    ingestor = GmailIngestor(account=gmail_account)

    with console.status("  [dim]Authenticating with Google...[/dim]"):
        try:
            authenticated = ingestor.authenticate()
        except ImportError as exc:
            console.print(f"  [red]Missing dependency:[/red] {exc}")
            return

    if not authenticated:
        return

    console.print("  [green]Authenticated.[/green] Fetching emails...")

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    try:
        with console.status("  [dim]Ingesting email threads...[/dim]"):
            stats = ingest_gmail(
                ingestor=ingestor,
                metadata_store=metadata_store,
                vector_store=vector_store,
                since=since,
                labels=list(labels) if labels else None,
                max_results=max_results,
            )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_row("Threads found", str(stats.files_found))
        table.add_row("Threads processed", str(stats.files_processed))
        table.add_row("Skipped (unchanged)", str(stats.files_skipped))
        table.add_row("Chunks created", str(stats.chunks_created))
        table.add_row("Total vectors", str(vector_store.count()))
        table.add_row("Time elapsed", f"{stats.elapsed_seconds:.1f}s")
        console.print(table)

        if stats.errors:
            console.print(f"\n  [yellow]Warnings ({len(stats.errors)}):[/yellow]")
            for err in stats.errors[:10]:
                console.print(f"    [dim]{err}[/dim]")
            if len(stats.errors) > 10:
                console.print(f"    [dim]... and {len(stats.errors) - 10} more[/dim]")

        existing_accounts = {s.account for s in config.sources if s.type == "gmail"}
        if gmail_account not in existing_accounts:
            config.sources.append(
                SourceConfig(
                    type="gmail",
                    account=gmail_account,
                    since=since,
                    labels=list(labels),
                )
            )
            save_config(config)
            console.print(
                f"\n  [green]Source saved to config.[/green] "
                "Run [bold]verra sync start[/bold] for continuous sync."
            )
    finally:
        db.close()
    console.print()




@main.command()
@click.option("--model", default=None, help="Override the model from config.")
def chat(model: str | None) -> None:
    """Start an interactive chat session with your data."""
    _run_chat_repl(model_override=model)




@main.command()
@click.option(
    "--type",
    "entity_type",
    default=None,
    help="Filter by type: person, company, project, product, location",
)
@click.option("--limit", default=50, show_default=True, help="Max entities to display.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def entities(entity_type: str | None, limit: int, as_json: bool) -> None:
    """Browse extracted entities and their relationships."""
    import json
    import sqlite3

    from verra.config import VERRA_HOME, ensure_data_dir

    ensure_data_dir()
    core_db = VERRA_HOME / "core.db"

    if not core_db.exists():
        console.print(
            "\n  [yellow]No data indexed yet.[/yellow] "
            "Run [bold]verra ingest <folder>[/bold] first.\n"
        )
        return

    conn = sqlite3.connect(str(core_db))
    conn.row_factory = sqlite3.Row

    try:
        if entity_type:
            rows = conn.execute(
                """
                SELECT e.id, e.canonical_name, e.entity_type,
                       COUNT(DISTINCT ea.alias) AS alias_count,
                       COUNT(DISTINCT ce.chunk_id) AS chunk_count
                FROM entities e
                LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
                LEFT JOIN chunk_entities ce ON ce.entity_id = e.id
                WHERE e.entity_type = ?
                GROUP BY e.id
                ORDER BY chunk_count DESC
                LIMIT ?
                """,
                (entity_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT e.id, e.canonical_name, e.entity_type,
                       COUNT(DISTINCT ea.alias) AS alias_count,
                       COUNT(DISTINCT ce.chunk_id) AS chunk_count
                FROM entities e
                LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
                LEFT JOIN chunk_entities ce ON ce.entity_id = e.id
                GROUP BY e.id
                ORDER BY chunk_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        if not rows:
            if as_json:
                click.echo(json.dumps([], indent=2))
            else:
                console.print("\n  [yellow]No entities found.[/yellow]\n")
            return

        if as_json:
            click.echo(
                json.dumps(
                    [
                        {
                            "id": row["id"],
                            "name": row["canonical_name"],
                            "type": row["entity_type"],
                            "aliases": row["alias_count"],
                            "chunks": row["chunk_count"],
                        }
                        for row in rows
                    ],
                    indent=2,
                )
            )
            return

        console.print()
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Aliases", justify="right")
        table.add_column("Chunks", justify="right")

        for row in rows:
            table.add_row(
                row["canonical_name"],
                row["entity_type"],
                str(row["alias_count"]),
                str(row["chunk_count"]),
            )

        console.print(table)
        console.print(f"\n  [dim]{len(rows)} entities shown.[/dim]\n")

    finally:
        conn.close()




@main.command()
@click.option("--limit", default=10, show_default=True, help="Number of conversations to show.")
def history(limit: int) -> None:
    """Show recent conversations."""
    import sqlite3

    from verra.config import VERRA_HOME, ensure_data_dir

    ensure_data_dir()
    core_db = VERRA_HOME / "core.db"

    if not core_db.exists():
        console.print("\n  [yellow]No conversation history yet.[/yellow]\n")
        return

    conn = sqlite3.connect(str(core_db))
    conn.row_factory = sqlite3.Row

    try:
        conversations = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count,
                   (SELECT content FROM messages
                    WHERE conversation_id = c.id AND role = 'user'
                    ORDER BY id LIMIT 1) AS first_question
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        if not conversations:
            console.print("\n  [yellow]No conversations yet.[/yellow]\n")
            return

        console.print()
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("When", style="dim")
        table.add_column("Messages", justify="right")
        table.add_column("First question")

        for row in conversations:
            first_q = row["first_question"] or ""
            if len(first_q) > 60:
                first_q = first_q[:57] + "..."
            table.add_row(
                row["updated_at"] or row["created_at"] or "",
                str(row["message_count"]),
                first_q,
            )

        console.print(table)
        console.print()

    finally:
        conn.close()




@main.command()
@click.confirmation_option(prompt="This will delete ALL ingested data. Are you sure?")
def clear() -> None:
    """Delete all ingested data and start fresh (config is preserved)."""
    import shutil

    from verra.config import VERRA_HOME, ensure_data_dir

    targets = [
        VERRA_HOME / "chroma",
        VERRA_HOME / "core.db",
        VERRA_HOME / "analysis.db",
        VERRA_HOME / "tabular.db",
        VERRA_HOME / "sqlite",
    ]

    removed: list[str] = []
    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(str(target))

    if removed:
        console.print("\n  [green]Removed:[/green]")
        for p in removed:
            console.print(f"    [dim]{p}[/dim]")
        console.print(
            "\n  [bold green]All data cleared.[/bold green] "
            "Config preserved at ~/.verra/config.yaml"
        )
    else:
        console.print("\n  [dim]Nothing to clear — no data files found.[/dim]")

    ensure_data_dir()
    console.print()




@main.group()
def sources() -> None:
    """Manage data sources."""


@sources.command(name="list")
def sources_list() -> None:
    """List all configured data sources."""
    import sqlite3

    from verra.config import VERRA_HOME, load_config

    config = load_config()

    if not config.sources:
        console.print(
            "\n  [yellow]No data sources configured.[/yellow]"
            "\n  Add one with [bold]verra sources add <path>[/bold] "
            "or [bold]verra ingest <path>[/bold].\n"
        )
        return

    # Build sync state lookup
    sync_lookup: dict[str, dict[str, str]] = {}
    core_db = VERRA_HOME / "core.db"
    if core_db.exists():
        conn = sqlite3.connect(str(core_db))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT source, last_sync_at, status FROM sync_state"
            ).fetchall()
            for r in rows:
                sync_lookup[r["source"]] = {
                    "last_sync_at": r["last_sync_at"],
                    "status": r["status"],
                }
        except Exception:
            pass
        finally:
            conn.close()

    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Type", style="cyan")
    table.add_column("Path / Account")
    table.add_column("Last Sync", style="dim")
    table.add_column("Status")

    for src in config.sources:
        identifier = src.path or src.account or "?"
        sync_info = sync_lookup.get(identifier, {})
        last_sync = sync_info.get("last_sync_at", "never")
        status_val = sync_info.get("status", "")
        if status_val == "idle":
            status_colored = f"[green]{status_val}[/green]"
        elif status_val:
            status_colored = f"[yellow]{status_val}[/yellow]"
        else:
            status_colored = "[dim]--[/dim]"
        table.add_row(src.type, identifier, last_sync, status_colored)

    console.print(table)
    console.print()


@sources.command(name="add")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def sources_add(path: Path) -> None:
    """Add a folder as a data source and ingest it immediately."""
    from verra.config import VERRA_HOME, SourceConfig, ensure_data_dir, load_config, save_config
    from verra.ingest.pipeline import ingest_folder
    from verra.store.analysis import AnalysisStore
    from verra.store.db import DatabaseManager
    from verra.store.entities import EntityStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    _apply_api_key_from_config()
    ensure_data_dir()
    config = load_config()

    abs_path = str(path.resolve())
    existing_paths = {s.path for s in config.sources if s.type == "folder"}
    if abs_path not in existing_paths:
        config.sources.append(SourceConfig(type="folder", path=abs_path))
        save_config(config)
        console.print(f"\n  [green]Source added:[/green] {abs_path}")
    else:
        console.print(f"\n  [dim]Source already configured:[/dim] {abs_path}")

    console.print(f"  [bold cyan]Ingesting:[/bold cyan] {abs_path}")
    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    entity_store = EntityStore.from_connection(db.core)
    analysis_store = AnalysisStore.from_connection(db.analysis)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    sa_tabular_store = None
    try:
        from verra.store.tabular import TabularStore as _TabularStore

        sa_tabular_store = _TabularStore(VERRA_HOME / "tabular.db")
    except Exception:
        pass

    with console.status("  [dim]Processing documents...[/dim]"):
        stats = ingest_folder(
            folder_path=path,
            metadata_store=metadata_store,
            vector_store=vector_store,
            entity_store=entity_store,
            analysis_store=analysis_store,
            tabular_store=sa_tabular_store,
        )

    console.print(
        f"  Done. [bold]{stats.files_processed}[/bold] files, "
        f"[bold]{stats.chunks_created}[/bold] chunks, "
        f"[bold]{stats.entities_found}[/bold] entities.\n"
    )
    db.close()




@main.group()
def sync() -> None:
    """Sync commands."""


@sync.command(name="start")
@click.option(
    "--background",
    "--daemon",
    "background",
    is_flag=True,
    default=False,
    help="Fork into the background and write a PID file to ~/.verra/sync.pid.",
)
def sync_start(background: bool) -> None:
    """Start the background sync daemon (runs until Ctrl+C).

    Pass --background (or --daemon) to detach from the terminal immediately.
    The child process writes its PID to ~/.verra/sync.pid so that
    'verra sync stop' can send SIGTERM to it later.
    """
    import os
    import sys

    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore
    from verra.sync.daemon import SyncDaemon

    ensure_data_dir()
    config = load_config()

    if not config.sync.enabled:
        console.print("\n  [yellow]Sync is disabled in config.[/yellow]\n")
        return

    if not config.sources:
        console.print(
            "\n  [yellow]No sources configured.[/yellow] "
            "Run [bold]verra ingest <folder>[/bold] first.\n"
        )
        return

    pid_file = VERRA_HOME / "sync.pid"

    if background:
        # -----------------------------------------------------------------
        # Fork-based daemonisation — POSIX only.
        # On Windows, fall back to subprocess.Popen with a detached process.
        # -----------------------------------------------------------------
        if hasattr(os, "fork"):
            pid = os.fork()
            if pid > 0:
                # Parent: record the child PID and exit.
                pid_file.write_text(str(pid))
                console.print(f"\n  [bold cyan]Sync daemon started (PID {pid})[/bold cyan]")
                console.print(f"  [dim]PID file: {pid_file}[/dim]")
                console.print(f"  [dim]Stop with: verra sync stop[/dim]\n")
                sys.exit(0)
            # Child: detach from the controlling terminal.
            os.setsid()
        else:
            # Windows fallback — spawn a new detached process.
            import subprocess

            cmd = [sys.executable, "-m", "verra.cli", "sync", "start"]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0),
            )
            pid_file.write_text(str(proc.pid))
            console.print(f"\n  [bold cyan]Sync daemon started (PID {proc.pid})[/bold cyan]")
            console.print(f"  [dim]PID file: {pid_file}[/dim]")
            console.print(f"  [dim]Stop with: verra sync stop[/dim]\n")
            return

    # ------------------------------------------------------------------
    # Foreground (or child process after fork) — run until interrupted.
    # ------------------------------------------------------------------
    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    daemon = SyncDaemon(
        config=config,
        metadata_store=metadata_store,
        vector_store=vector_store,
    )

    source_summary = ", ".join(s.path or s.account or "?" for s in config.sources)
    if not background:
        # Only print the banner in foreground mode; the background child is silent.
        console.print()
        console.print("  [bold cyan]Verra Sync Daemon[/bold cyan]")
        console.print(f"  [dim]Sources: {source_summary}[/dim]")
        console.print(f"  [dim]Interval: {config.sync.interval}s — Ctrl+C to stop[/dim]")
        console.print()

    daemon.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if not background:
            console.print("\n  [dim]Stopping daemon...[/dim]")
        daemon.stop()
        db.close()
        # Clean up PID file written by a background child
        if background and pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass
        if not background:
            console.print("  [green]Daemon stopped.[/green]\n")


@sync.command(name="stop")
def sync_stop() -> None:
    """Stop a background sync daemon that was started with --background.

    Reads ~/.verra/sync.pid and sends SIGTERM to the recorded process.
    """
    import os
    import signal

    from verra.config import VERRA_HOME

    pid_file = VERRA_HOME / "sync.pid"

    if not pid_file.exists():
        console.print("\n  [yellow]No PID file found — is the daemon running?[/yellow]\n")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError) as exc:
        console.print(f"\n  [red]Could not read PID file: {exc}[/red]\n")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        console.print(f"\n  [green]Sent SIGTERM to PID {pid}.[/green]\n")
    except ProcessLookupError:
        console.print(
            f"\n  [yellow]Process {pid} not found — removing stale PID file.[/yellow]\n"
        )
        pid_file.unlink(missing_ok=True)
    except PermissionError:
        console.print(f"\n  [red]Permission denied sending signal to PID {pid}.[/red]\n")


@sync.command(name="status")
def sync_status() -> None:
    """Show the sync state for all data sources."""
    import sqlite3

    from verra.config import VERRA_HOME, ensure_data_dir

    ensure_data_dir()
    core_db = VERRA_HOME / "core.db"

    if not core_db.exists():
        console.print("\n  [yellow]No sync data yet.[/yellow] Run [bold]verra ingest[/bold] first.\n")
        return

    conn = sqlite3.connect(str(core_db))
    conn.row_factory = sqlite3.Row
    try:
        states = [
            dict(r)
            for r in conn.execute(
                "SELECT source, last_sync_at, items_processed, status FROM sync_state"
            ).fetchall()
        ]
    except Exception:
        states = []
    finally:
        conn.close()

    if not states:
        console.print("\n  [yellow]No sources synced yet.[/yellow]\n")
        return

    console.print()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Source", style="cyan")
    table.add_column("Last Sync")
    table.add_column("Items", justify="right")
    table.add_column("Status")

    for state in states:
        status_color = "green" if state["status"] == "idle" else "yellow"
        table.add_row(
            state["source"],
            state["last_sync_at"],
            str(state["items_processed"]),
            f"[{status_color}]{state['status']}[/{status_color}]",
        )

    console.print(table)
    console.print()




@main.group()
def analytics() -> None:
    """Analytics commands."""


@analytics.command(name="run")
def analytics_run() -> None:
    """Run batch analysis across the full knowledge base and cache results."""
    from verra.config import VERRA_HOME, ensure_data_dir
    from verra.analytics.batch import BatchAnalytics
    from verra.store.db import DatabaseManager
    from verra.store.entities import EntityStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    core_db = VERRA_HOME / "core.db"

    if not core_db.exists():
        console.print("\n  [yellow]No data indexed yet.[/yellow] Run [bold]verra ingest[/bold] first.\n")
        return

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    entity_store = EntityStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    console.print("\n  [bold cyan]Running batch analytics...[/bold cyan]")

    batch = BatchAnalytics(
        metadata_store=metadata_store,
        entity_store=entity_store,
        vector_store=vector_store,
        db_path=VERRA_HOME / "analysis.db",
    )

    batch.compute_entity_mention_counts()
    entity_count = len(batch.get_analytics("entity_mention_counts"))
    console.print(f"  [green]→[/green] Entity mentions ({entity_count} entities)")

    batch.compute_communication_frequency()
    freq_count = len(batch.get_analytics("communication_frequency"))
    console.print(f"  [green]→[/green] Communication frequency ({freq_count} periods)")

    batch.compute_source_distribution()
    src_count = len(batch.get_analytics("source_distribution"))
    console.print(f"  [green]→[/green] Source distribution ({src_count} types)")

    influence_count = batch.run_influence_analysis()
    console.print(f"  [green]→[/green] Influence graph ({influence_count} edges)")

    batch.close()
    console.print("\n  [green]Analytics complete.[/green]\n")
    db.close()


@analytics.command(name="show")
def analytics_show() -> None:
    """Display key analytics insights from the last run."""
    from verra.config import VERRA_HOME, ensure_data_dir
    from verra.analytics.batch import BatchAnalytics
    from verra.store.db import DatabaseManager
    from verra.store.entities import EntityStore
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    analysis_db = VERRA_HOME / "analysis.db"
    core_db = VERRA_HOME / "core.db"

    if not analysis_db.exists():
        console.print(
            "\n  [yellow]No analytics yet.[/yellow] Run [bold]verra analytics run[/bold] first.\n"
        )
        return

    if not core_db.exists():
        console.print(
            "\n  [yellow]No data indexed yet.[/yellow] Run [bold]verra ingest[/bold] first.\n"
        )
        return

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    entity_store = EntityStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    batch = BatchAnalytics(
        metadata_store=metadata_store,
        entity_store=entity_store,
        vector_store=vector_store,
        db_path=analysis_db,
    )

    # Top entities by mention count
    mention_rows = batch.get_analytics("entity_mention_counts")
    mention_rows.sort(key=lambda r: r["value_json"].get("count", 0), reverse=True)

    if mention_rows:
        console.print("\n  [bold]Top Entities by Mentions:[/bold]")
        for i, row in enumerate(mention_rows[:10], 1):
            entity = entity_store.get_entity(row["entity_id"])
            name = entity["canonical_name"] if entity else f"entity#{row['entity_id']}"
            count = row["value_json"].get("count", 0)
            console.print(f"    {i}. {name} [dim]({count} mentions)[/dim]")

    # Influence graph
    edges = batch.get_influence_edges()
    if edges:
        console.print("\n  [bold]Influence Graph:[/bold]")
        for edge in edges[:10]:
            from_entity = entity_store.get_entity(edge["from_entity_id"])
            to_entity = entity_store.get_entity(edge["to_entity_id"])
            from_name = (
                from_entity["canonical_name"]
                if from_entity
                else f"entity#{edge['from_entity_id']}"
            )
            to_name = (
                to_entity["canonical_name"] if to_entity else f"entity#{edge['to_entity_id']}"
            )
            score = edge["influence_score"]
            comms = edge["communication_count"]
            console.print(
                f"    {from_name} → {to_name} "
                f"[dim](score: {score:.2f}, {comms} comms)[/dim]"
            )

    # Knowledge gaps are in analysis.db
    import sqlite3

    gap_count = 0
    try:
        conn = sqlite3.connect(str(analysis_db))
        gap_count = conn.execute(
            "SELECT COUNT(*) FROM knowledge_gaps WHERE resolved = 0"
        ).fetchone()[0]
        conn.close()
    except Exception:
        pass
    console.print(f"\n  [bold]Knowledge Gaps:[/bold] {gap_count} unresolved\n")

    batch.close()
    db.close()




@main.command()
def briefing() -> None:
    """Show actionable insights from your data (stale leads, expiring contracts, etc.)."""
    from verra.briefing.detector import BriefingDetector
    from verra.config import VERRA_HOME, ensure_data_dir, load_config
    from verra.store.db import DatabaseManager

    ensure_data_dir()
    config = load_config()
    core_db = VERRA_HOME / "core.db"

    if not core_db.exists():
        console.print("\n  [yellow]No data indexed yet.[/yellow] Run [bold]verra ingest[/bold] first.\n")
        return

    db = DatabaseManager(VERRA_HOME)
    detector = BriefingDetector(
        core_conn=db.core,
        analysis_conn=db.analysis,
        config=config.briefing,
    )
    items = detector.detect_all()

    if not items:
        console.print("\n  [dim]No actionable insights right now.[/dim]\n")
        db.close()
        return

    console.print()
    console.print("  [bold cyan]Verra Briefing[/bold cyan]")
    console.print()

    urgency_icons = {5: "[red]!![/red]", 4: "[yellow]![/yellow]", 3: "[cyan]·[/cyan]", 2: "[dim]·[/dim]", 1: "[dim]·[/dim]"}
    for i, item in enumerate(items, 1):
        icon = urgency_icons.get(item.urgency, "·")
        console.print(f"  {icon} [bold]{item.title}[/bold]")
        console.print(f"    [dim]{item.detail}[/dim]")
        console.print()

    db.close()




@main.command()
@click.argument("account")
@click.option("--folder-id", default=None, help="Google Drive folder ID to ingest from.")
@click.option("--max-results", default=500, show_default=True, help="Max files to fetch.")
def drive(account: str, folder_id: str | None, max_results: int) -> None:
    """Ingest files from Google Drive.

    ACCOUNT: your Google account email (e.g. you@gmail.com).

    On first run this opens a browser for OAuth consent.
    Requires credentials.json in ~/.verra/oauth/.
    """
    from verra.config import VERRA_HOME, SourceConfig, ensure_data_dir, load_config, save_config
    from verra.ingest.drive import DriveIngestor, ingest_drive
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()

    console.print()
    console.print(f"  [bold cyan]Google Drive Ingest[/bold cyan]  {account}")
    console.print()

    ingestor = DriveIngestor(account=account)

    with console.status("  [dim]Authenticating with Google...[/dim]"):
        try:
            authenticated = ingestor.authenticate()
        except ImportError as exc:
            console.print(f"  [red]Missing dependency:[/red] {exc}")
            return

    if not authenticated:
        console.print("  [red]Authentication failed.[/red] Check that credentials.json exists in ~/.verra/oauth/")
        return

    console.print("  [green]Authenticated.[/green] Fetching files...")

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    try:
        with console.status("  [dim]Ingesting files from Drive...[/dim]"):
            stats = ingest_drive(
                ingestor=ingestor,
                metadata_store=metadata_store,
                vector_store=vector_store,
                folder_id=folder_id,
                max_results=max_results,
            )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_row("Files found", str(stats.files_found))
        table.add_row("Files processed", str(stats.files_processed))
        table.add_row("Skipped (unchanged)", str(stats.files_skipped))
        table.add_row("Chunks created", str(stats.chunks_created))
        table.add_row("Time elapsed", f"{stats.elapsed_seconds:.1f}s")
        console.print(table)

        if stats.errors:
            console.print(f"\n  [yellow]Warnings ({len(stats.errors)}):[/yellow]")
            for err in stats.errors[:10]:
                console.print(f"    [dim]{err}[/dim]")

        # Save to config
        existing = {s.path for s in config.sources if s.type == "drive"}
        if account not in existing:
            config.sources.append(SourceConfig(type="drive", account=account))
            save_config(config)
            console.print(f"\n  [green]Source saved to config.[/green]")
    finally:
        db.close()
    console.print()




@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind to.")
@click.option("--port", default=8484, show_default=True, help="TCP port to listen on.")
@click.option("--api-key", default=None, help="Require this API key for all requests (X-API-Key header).")
def serve(host: str, port: int, api_key: str | None) -> None:
    """Start the HTTP API server.

    By default the server listens on 127.0.0.1:8484 (localhost only).
    Pass --host 0.0.0.0 to listen on all network interfaces.

    API documentation is available at http://<host>:<port>/docs once the
    server is running.
    """
    import uvicorn
    from verra.server import create_app

    app = create_app(api_key=api_key)

    console.print()
    console.print("  [bold cyan]Verra API Server[/bold cyan]")
    console.print(f"  [dim]Listening on {host}:{port}[/dim]")
    console.print(f"  [dim]API docs: http://{host}:{port}/docs[/dim]")
    if api_key:
        console.print("  [dim]Auth: X-API-Key required[/dim]")
    else:
        console.print("  [yellow]Auth: disabled (no --api-key set)[/yellow]")
    console.print("  [dim]Press Ctrl+C to stop[/dim]")
    console.print()

    uvicorn.run(app, host=host, port=port, log_level="warning")




@main.command()
@click.argument("account")
@click.option("--client-id", default=None, help="Azure AD app client ID.")
@click.option("--since", default=None, help="Only ingest emails after this date (YYYY-MM-DD).")
@click.option("--folder", default="inbox", show_default=True, help="Mail folder: inbox, sentitems, etc.")
@click.option("--max-results", default=500, show_default=True, help="Max messages to fetch.")
def outlook(
    account: str,
    client_id: str | None,
    since: str | None,
    folder: str,
    max_results: int,
) -> None:
    """Ingest emails from Microsoft Outlook / Office 365.

    ACCOUNT: your Microsoft email (e.g. you@company.com).

    Requires an Azure AD app registration with Mail.Read permission.
    On first run, uses device-code flow (opens browser, enter code).
    """
    from verra.config import VERRA_HOME, SourceConfig, ensure_data_dir, load_config, save_config
    from verra.ingest.outlook import OutlookIngestor, ingest_outlook
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    config = load_config()

    # Look up saved client_id from config if not provided on the command line
    if not client_id:
        for src in config.sources:
            if src.type == "outlook" and src.account == account and src.client_id:
                client_id = src.client_id
                break

    console.print()
    console.print(f"  [bold cyan]Outlook Ingest[/bold cyan]  {account}")
    console.print(f"  [dim]folder={folder}, since={since or 'all time'}[/dim]")
    console.print()

    ingestor = OutlookIngestor(account=account, client_id=client_id)

    try:
        authenticated = ingestor.authenticate()
    except ImportError:
        console.print("  [red]msal is required.[/red] Install with: pip install msal")
        return

    if not authenticated:
        return

    console.print("  [green]Authenticated.[/green] Fetching emails...")

    db = DatabaseManager(VERRA_HOME)
    metadata_store = MetadataStore.from_connection(db.core)
    vector_store = VectorStore(VERRA_HOME / "chroma")

    try:
        with console.status("  [dim]Ingesting emails from Outlook...[/dim]"):
            stats = ingest_outlook(
                ingestor=ingestor,
                metadata_store=metadata_store,
                vector_store=vector_store,
                since=since,
                folder=folder,
                max_results=max_results,
            )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_row("Messages found", str(stats.files_found))
        table.add_row("Messages processed", str(stats.files_processed))
        table.add_row("Skipped (unchanged)", str(stats.files_skipped))
        table.add_row("Chunks created", str(stats.chunks_created))
        table.add_row("Time elapsed", f"{stats.elapsed_seconds:.1f}s")
        console.print(table)

        if stats.errors:
            console.print(f"\n  [yellow]Warnings ({len(stats.errors)}):[/yellow]")
            for err in stats.errors[:10]:
                console.print(f"    [dim]{err}[/dim]")

        # Save source to config, storing client_id so it doesn't need to be re-entered
        existing_sources = [s for s in config.sources if s.type == "outlook" and s.account == account]
        if not existing_sources:
            config.sources.append(SourceConfig(type="outlook", account=account, client_id=client_id))
            save_config(config)
            console.print(f"\n  [green]Source saved to config.[/green]")
        elif client_id:
            # Update client_id on the existing source if it changed or was newly provided
            for src in existing_sources:
                if src.client_id != client_id:
                    src.client_id = client_id
                    save_config(config)
                    console.print(f"\n  [green]Client ID saved to config.[/green]")
                    break
    finally:
        db.close()
    console.print()




@main.command(name="search")
@click.argument("query")
@click.option("--limit", default=10, show_default=True, help="Max results.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def search_cmd(query: str, limit: int, as_json: bool) -> None:
    """Search your data without using the LLM."""
    import json as json_mod

    from verra.config import VERRA_HOME, ensure_data_dir
    from verra.retrieval.router import parse_query
    from verra.retrieval.search import search as do_search
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    if not (VERRA_HOME / "core.db").exists():
        console.print("\n  [yellow]No data indexed yet.[/yellow]\n")
        return

    db = DatabaseManager(VERRA_HOME)
    meta = MetadataStore.from_connection(db.core)
    vec = VectorStore(VERRA_HOME / "chroma")

    classified = parse_query(query)
    results = do_search(classified, meta, vec, n_results=limit)

    if as_json:
        out = [
            {"text": r.text[:500], "score": round(r.score, 4),
             "metadata": r.metadata, "authority": r.authority_weight}
            for r in results
        ]
        click.echo(json_mod.dumps(out, indent=2, default=str))
    else:
        if not results:
            console.print("\n  [dim]No results found.[/dim]\n")
        else:
            console.print()
            for i, r in enumerate(results, 1):
                label = r.metadata.get("file_name", r.metadata.get("source_type", "?"))
                console.print(f"  [cyan]{i}.[/cyan] [bold]{label}[/bold] [dim](score: {r.score:.3f})[/dim]")
                preview = r.text[:200].replace("\n", " ").strip()
                console.print(f"     [dim]{preview}...[/dim]")
                console.print()
    db.close()




@main.command()
@click.argument("doc_id", type=int, required=False)
@click.option("--source", default=None, help="Delete all docs from a source type.")
@click.option("--path-pattern", default=None, help="Delete docs matching path pattern.")
def delete(doc_id: int | None, source: str | None, path_pattern: str | None) -> None:
    """Delete specific documents from the knowledge base."""
    from verra.config import VERRA_HOME, ensure_data_dir
    from verra.store.db import DatabaseManager
    from verra.store.metadata import MetadataStore
    from verra.store.vector import VectorStore

    ensure_data_dir()
    if not (VERRA_HOME / "core.db").exists():
        console.print("\n  [yellow]No data indexed yet.[/yellow]\n")
        return

    db = DatabaseManager(VERRA_HOME)
    meta = MetadataStore.from_connection(db.core)
    vec = VectorStore(VERRA_HOME / "chroma")

    try:
        if doc_id is not None:
            doc = db.core.execute(
                "SELECT id, file_name, source_type FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if not doc:
                console.print(f"\n  [yellow]Document {doc_id} not found.[/yellow]\n")
                return
            console.print(f"\n  Deleting: {doc['file_name']} (id={doc_id})")
            if click.confirm("  Are you sure?", default=False):
                vec.delete_by_document_id(doc_id)
                meta.delete_document(doc_id)
                console.print("  [green]Deleted.[/green]\n")

        elif source:
            rows = db.core.execute(
                "SELECT id FROM documents WHERE source_type = ?", (source,)
            ).fetchall()
            if not rows:
                console.print(f"\n  [yellow]No documents with source_type='{source}'.[/yellow]\n")
                return
            console.print(f"\n  Found {len(rows)} documents with source_type='{source}'.")
            if click.confirm(f"  Delete all {len(rows)}?", default=False):
                for row in rows:
                    vec.delete_by_document_id(row["id"])
                    meta.delete_document(row["id"])
                console.print(f"  [green]Deleted {len(rows)} documents.[/green]\n")

        elif path_pattern:
            rows = db.core.execute(
                "SELECT id, file_path FROM documents WHERE file_path LIKE ?",
                (f"%{path_pattern}%",),
            ).fetchall()
            if not rows:
                console.print(f"\n  [yellow]No documents matching '{path_pattern}'.[/yellow]\n")
                return
            console.print(f"\n  Found {len(rows)} documents matching '{path_pattern}'.")
            if click.confirm(f"  Delete all {len(rows)}?", default=False):
                for row in rows:
                    vec.delete_by_document_id(row["id"])
                    meta.delete_document(row["id"])
                console.print(f"  [green]Deleted {len(rows)} documents.[/green]\n")
        else:
            console.print("\n  Usage:")
            console.print("    verra delete <doc_id>            -- delete by ID")
            console.print("    verra delete --source folder     -- delete by source type")
            console.print("    verra delete --path-pattern acme -- delete by path match\n")
    finally:
        db.close()


@main.command(name="eval")
@click.option("--category", default=None, help="Only run cases in this category.")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output results as JSON.")
def eval_cmd(category: str | None, output_json: bool) -> None:
    """Run the automated evaluation suite against ingested data."""
    import os

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    _apply_api_key_from_config()

    from verra.eval import run_eval

    results = run_eval(category_filter=category, output_json=output_json)
    failures = sum(1 for r in results if not r["passed"])
    if failures:
        raise SystemExit(1)
