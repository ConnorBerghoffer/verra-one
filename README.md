# Verra One

Chat with all your business data — emails, documents, spreadsheets. Runs locally with Ollama or connects to Claude/OpenAI.

### Setup

![Setup](assets/verra-setup.gif)

### Usage

![Usage](assets/verra-use.gif)

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
- **MCP** — `verra mcp` exposes your data as tools for Claude Desktop and other AI clients

## Commands

```
verra                       # chat
verra ingest ~/Documents    # add a folder
verra gmail you@gmail.com   # connect Gmail
verra search "pricing"      # search without LLM
verra briefing              # what needs attention
verra status                # what's ingested
verra docs                  # browse ingested documents
verra eval                  # run the eval suite
verra mcp                   # start MCP server for Claude Desktop
verra mcp-config            # print Claude Desktop config
verra serve --port 8484     # HTTP API
verra sync start            # background file watcher
verra update                # upgrade to latest version
```

Full list: `verra --help`

## How the retrieval works

I spent a lot of time on this. Most RAG tools do one thing — embed your text, search by cosine similarity, feed results to an LLM. That works for demos but falls apart on real business data where you have 800 files across 15 folders, half of them CSVs with 100K rows, and the user asks "which sales rep has the best win rate?"

Here's what Verra actually does when you ask a question:

**Six retrieval strategies, merged and reranked:**

1. **Embedding search** — nomic-embed-text-v1.5 (768-dim, way better than the default all-MiniLM-L6-v2 that every other tool uses). Query gets expanded with synonyms and converted to declarative form before embedding — poor man's HyDE without the LLM call.

2. **BM25 full-text search** — SQLite FTS5 index on every chunk. Catches exact terms, acronyms, and names that embeddings miss. "DataForge" doesn't embed well but BM25 finds it instantly.

3. **Keyword matching** — fallback that scans chunk text for literal query terms. Redundant with BM25 in theory but catches edge cases where FTS5 tokenization splits compound words.

4. **Filename search** — if you ask about "PTO" and there's a file called `pto_log_2025.csv`, that file gets guaranteed inclusion in results regardless of what the other strategies think. This sounds obvious but no other tool does it.

5. **Entity-based retrieval** — query terms are resolved against an entity registry (people, companies, projects extracted at ingest time). If "Jennifer Walsh" matches a known entity, we pull all chunks linked to that entity.

6. **SQL on tabular data** — CSVs are loaded as actual SQLite tables at ingest time. When a question looks analytical ("which", "best", "total", "compare"), we ask the LLM to generate a SQL query and execute it against the real data. The answer comes from `SELECT Owner, COUNT(*) ... GROUP BY Owner`, not from an 8B model squinting at CSV fragments.

**After retrieval:**

- **Cross-encoder reranking** — all candidates from all six strategies go through ms-marco-MiniLM-L-6-v2. This rescores every (query, passage) pair independently, which is way more accurate than bi-encoder similarity alone. ~30-40% accuracy improvement in the literature, and it holds in practice.

- **Authority weighting** — contracts outrank meeting notes outrank emails. When two sources say different things, the more authoritative one wins. Composite score: 75% relevance + 20% authority + 5% recency.

- **Diversity caps** — max 2 chunks per file, max 3 CSV chunks total. Prevents one large file from filling all result slots.

- **Reserved filename slots** — files matched by filename search can't be displaced by reranking. If the file is literally named after what you asked, it stays in the results.

- **Parent-child chunks** — chunks are ~800 chars for precise embedding matching, but the LLM sees a 2KB window around each match. For small files (<8KB), the entire document is pulled into context.

- **Query decomposition** — "Compare Q3 and Q4 financials" becomes two separate searches, results merged. No LLM call — regex-based splitting.

- **Relevance grading** — after the first search, we check if the results actually contain the query terms. If coverage is below 50%, the query gets expanded with synonyms and searched again. Both result sets merge.

## What I tried and what happened

**Things that worked:**

- Parallel ingest with ThreadPoolExecutor dropped ingest from 3 minutes to 17 seconds on 823 files. CPU-bound steps (chunking, NER, heuristic analysis) run in 4 threads, serial steps (SQLite writes, embedding) stay on the main thread. No race conditions because the hash check happens before submitting to the pool.

- CSV row capping at 500 rows was a huge win. A 100K-row transaction log was creating 4000 chunks that dominated the embedding space and drowned out the actual documents people were asking about. Capping to 500 rows + loading the full CSV as a SQL table gives you the best of both worlds.

- Cross-encoder reranking lived up to the literature claims. Before adding it, the right document was in the top 60 embedding results but buried under irrelevant stuff. After reranking, it surfaces to position 1-3 consistently.

- Temperature 0 for generation. The 8B model's variance at temp=0.1 was ±15 points on the eval. At temp=0 it's deterministic. For a tool where people need consistent answers about their business data, you want zero creativity.

**Things that didn't work:**

- LLM-based HyDE (generating a hypothetical answer before searching). Added 2 seconds per query for marginal improvement. Replaced it with template-based synonym expansion — 90% of the benefit at zero cost.

- LLM-based query decomposition. Same story — 2 seconds for something regex does fine. "Compare X and Y" splits on "and" just as well with a pattern match.

- LLM-based coreference resolution for follow-up questions. Three seconds to resolve "Who leads it?" into "Who leads Project Alpha?". Replaced with a heuristic that grabs the last mentioned entity from conversation history.

- Sending 16 results to an 8B model. It gets confused and ignores half of them. 10 results is the sweet spot — enough coverage, focused enough that the model actually reads everything.

- Large CSV chunks in the embedding index. Before diversity caps and the CSV row limit, 94% of all chunks were CSV rows. Every question returned CSV data regardless of what you asked. The fix was format-aware diversity caps (max 3 CSV chunks in any result set) plus loading CSVs as SQL tables for structured queries.

## Eval results

25-question eval suite across 23 categories (financial, entity, incident, org, HR, ops, etc.) on a corpus of 823 business documents:

```
llama3.1:8b on RTX 4090:  87-92% accuracy, 5-7s/question
```

The remaining failures are all LLM generation — the retrieval finds the right documents for every single question. The 8B model just phrases answers differently than the eval expects, or misses specific details that are in the context it was given.

Run `verra eval` to benchmark on your own data.

## API

`verra serve` starts a FastAPI server with endpoints for chat, search, documents, entities, and briefing. OpenAPI docs at `/docs`.

## MCP (Claude Desktop)

```bash
verra mcp-config    # prints the config to paste into Claude Desktop
```

Exposes three tools: `verra_search`, `verra_ask`, `verra_status`. Claude Desktop (or any MCP client) can search and query your data directly.

## Models

Works with any LLM via [LiteLLM](https://github.com/BerriAI/litellm). Recommended local model: `llama3.1:8b` via Ollama. Bigger models (70B, Claude, GPT-4o) will score higher on the eval but the retrieval quality is the same.

## PDF extraction

Uses [pypdf](https://github.com/py-pdf/pypdf) (MIT) by default. For better quality on complex PDFs:

```bash
pip install "verra-one[pymupdf]"
```

## License

MIT
