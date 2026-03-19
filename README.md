# Verra One

Chat with all your business data — emails, documents, spreadsheets. Runs locally with Ollama or connects to Claude/OpenAI.

## Install

```bash
pip install verra-one
verra
```

First run walks you through setup (pick a model, point at your data).

## What it does

- **Ingest** folders, Gmail, Google Drive, Outlook — keeps syncing in the background
- **Search** across everything with hybrid retrieval (vector + metadata + keyword)
- **Chat** with your data — answers cite sources, refuses to guess
- **Briefing** — surfaces stale leads, overdue commitments, expiring contracts
- **Actions** — draft emails, set reminders, take notes through the chat
- **Deploy** — `verra deploy user@server` sets up everything remotely via SSH

## Commands

```
verra                       # chat
verra ingest ~/Documents    # add a folder
verra gmail you@gmail.com   # connect Gmail
verra search "pricing"      # search without LLM
verra briefing              # what needs attention
verra serve --port 8484     # HTTP API for building UIs
verra status                # what's ingested
verra delete --source email # remove data
```

Full list: `verra --help`

## API

`verra serve` starts a FastAPI server with endpoints for chat, search, documents, entities, and briefing. OpenAPI docs at `/docs`.

## Models

Works with any LLM via [LiteLLM](https://github.com/BerriAI/litellm). Recommended local model: `llama3.1:8b` via Ollama.

## PDF extraction

Uses [pypdf](https://github.com/py-pdf/pypdf) (MIT) by default. For better quality on complex PDFs:

```bash
pip install "verra-one[pymupdf]"
```

## License

MIT
