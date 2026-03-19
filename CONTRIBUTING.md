# Contributing

## Setup

```bash
git clone https://github.com/connorberghoffer/verra-one.git
cd verra-one
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest                              # unit tests
python tests/test_load.py --quick   # load tests
pytest tests/test_security.py       # security tests
```

## PR process

1. Fork, branch from `main`
2. Make changes, add tests if needed
3. Run `python tests/test_load.py --quick`
4. Open PR

## Known rough edges

- Math accuracy depends on the LLM model (8B models struggle with >4 number sums)
- Multi-hop tool calling only works well with 14B+ models
- No plugin system yet — adding a connector means editing several files
- The chunker creates large chunks (~800 tokens) which can dilute embeddings for long documents

## Layout

```
src/verra/
  cli.py        — CLI + REPL
  config.py     — config management
  server.py     — HTTP API (FastAPI)
  agent/        — chat engine, LLM, tools
  ingest/       — extractors, chunking, NER, connectors
  store/        — SQLite + ChromaDB stores
  retrieval/    — search, ranking, routing
  briefing/     — insight detection
  sync/         — background sync
  deploy/       — SSH deploy
  analytics/    — batch analytics
```
