# Verra One

Chat with all your business data — emails, documents, spreadsheets. Runs locally with Ollama or connects to Claude/OpenAI.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ConnorBerghoffer/verra-one/main/install.sh | sh
```

Or manually:

```bash
pipx install verra-one    # recommended
pip install verra-one     # if you don't have pipx
```

Then run `verra` — first run walks you through setup.

## Update

```bash
pipx upgrade verra-one    # if installed with pipx
pip install -U verra-one  # if installed with pip
```

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
