# RAG-Dagster

A RAG lab for PDF documents (text and tables). Upload a document, choose how it is chunked, embed it into a vector database, search it, ask questions of it in a chatbot, and benchmark embedding models, chunking strategies and search strategies against each other.

- **Parsing:** Docling (read through LlamaIndex's `DoclingReader`)
- **Chunking:** LlamaIndex splitters or our own (`hybrid`, `hierarchical`, `fixed`, `recursive`, `semantic`)
- **Embedding and reranking:** Ollama (`qwen3-embedding`, `Qwen3-Reranker`)
- **Vector database:** Qdrant (dense vectors plus an optional BM25 sparse vector)
- **Chatbot:** LangGraph and `langchain-ollama` with `gemma4:e4b-mlx`
- **UI:** Streamlit. **Batch ingestion:** Dagster. **Metrics:** PostgreSQL

Only PDFs with a text layer are supported. Images, figures and OCR are out of scope.

## Architecture

```
upload / data/raw/*.pdf -> parse (Docling) -> chunk -> embed (Ollama) -> index (Qdrant) -> search
                                                                               |
                                                         metrics -> PostgreSQL (rag_metrics)

Chatbot: question -> condense -> hybrid search + rerank -> grade -> answer with citations
                                      ^                       |
                                      +---- rewrite <---------+  (not enough: retry once, then say so)
```

| Service | Where | Address |
|---|---|---|
| Streamlit UI | Docker | http://localhost:8501 |
| Dagster | Docker | http://localhost:3000 |
| Qdrant | Docker | http://localhost:6333/dashboard |
| PostgreSQL | Docker | `localhost:5432` (databases `dagster` and `rag_metrics`) |
| Ollama | wherever `OLLAMA_BASE_URL` points (this project runs it on a Mac over Tailscale) | `OLLAMA_BASE_URL` |

## Requirements

- Docker with Compose
- An Ollama server, version 0.35 or newer (the reranker needs `logprobs`), reachable from the containers. If it is on another machine it must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`), and you should use its IP address, not a MagicDNS name.
- These Ollama models:

  ```
  ollama pull qwen3-embedding:0.6b                   # also :4b and :8b if you want to compare them
  ollama pull dengcao/Qwen3-Reranker-4B:Q8_0         # the reranker (Q4_K_M also works)
  ollama pull gemma4:e4b-mlx                         # the chatbot's model
  ```

  The 0.6B reranker builds do not work: they give every chunk a score of 0.

## Getting started

```powershell
copy .env.example .env      # then set OLLAMA_BASE_URL, POSTGRES_USER, POSTGRES_PASSWORD, SQL_LOADER_PASSWORD and SQL_READER_PASSWORD
docker compose up -d --build
```

The first start is slow: the code container takes about 50 s because it imports Docling, and Docling downloads its models on first use. Open http://localhost:8501.

## The UI

| Page | What it does |
|---|---|
| **Experiments** | Lists every experiment with the PDFs in it. Delete a PDF from an experiment, or a whole experiment. |
| **Upload** | Upload a PDF, pick a chunking strategy and its settings, see where the cuts fall in the document, and embed it into a new or existing experiment. Tick *Add a BM25 keyword vector* if you want hybrid search later; it cannot be added to an experiment afterwards. |
| **Try a query** | Pick an experiment and a search strategy (`dense`, `hybrid`, `dense+rerank`, `hybrid+rerank`), set top k, and see the chunks with their scores and timings. |
| **Chatbot** | Ask questions of your documents or of your tables. A new chat starts with *Search in*: **Documents** (an experiment with a BM25 vector; the answer is written from the top chunks of a hybrid search with reranking, with `[n]` citations) or **Database** (a schema of imported tables; the model writes a SELECT, it is checked and run read-only, and the answer is written from the rows, with the SQL shown under it; click **Good answer** under one that is right and it is kept as an example the model is shown for similar questions later). The choice is fixed for the whole chat: to search somewhere else, start a new chat. The page shows what happens as it happens: each step, the model's thinking, the retrieved chunks and scores, how full the model's context is, and how long each step took. If the documents do not answer the question it retries once with a different query, and otherwise says so. |
| **Database** | Import CSV files into a schema of the database for imported tables (UTF-8, first row the header, up to 50 MB and 1,000,000 rows each). Each file shows a preview and the column types it guessed, which you can change, and a table name. A bad row stops the import and leaves nothing behind. Below, what each table holds (with descriptions you can write), what the model is given about the schema, the good answers saved for it (turn one off or delete it), and deletes for a table or a schema. |
| **Benchmark** | Runs one document through every chosen embedding model, chunking strategy and search strategy, and saves a report with a PDF download. |

## Batch ingestion with Dagster

Drop PDFs in `data/raw/`. A sensor registers each as a Dagster partition (its id is the first 16 hex characters of the file's SHA-256). Launch `ingest_job` for a partition in the Dagster UI (http://localhost:3000) with the experiment set in the run config under `resources.experiment.config`; it runs parse, chunk, embed and index. See `CLAUDE.md` for the config keys.

## Command line

```powershell
# search an experiment
docker compose exec dagster-code python -m rag_lab.search "query text" --experiment <name> --top-k 5

# the chatbot, with every step printed (no question = a chat loop that keeps history)
docker compose exec dagster-code python -m rag_lab.agent documents "question" --experiment <name>

# the text-to-SQL agent over a schema of imported tables, every step printed
docker compose exec dagster-code python -m rag_lab.agent sql "question" --schema <name>
```

`notebook/search.ipynb` runs searches from a notebook with `rag_lab.search.quick.ask`.

## Common commands

```powershell
docker compose up -d --build          # start the stack (rebuild after changing dependencies)
docker compose restart ui             # after editing a UI module other than a page script
docker compose logs -f dagster-code   # code location logs
docker compose down                   # stop; add -v to wipe the Qdrant and Postgres volumes
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect the metrics
uv run pytest                         # the few pure-logic tests; no containers needed
```

`./src` and `./ui` are mounted into the containers, so code changes show up without a rebuild. Qdrant's data lives in a Docker named volume, not a bind mount; a bind mount to the Windows filesystem can corrupt it.

## How it is organised

```
src/rag_lab/
  config.py       experiment, search and agent settings (also Dagster run config)
  ingest.py       the stage bodies shared by Dagster, the Upload page and the benchmark
  parsing/  chunking/  embedding/  reranking/  storage/  search/
  agent/          the chatbot: shared events, model and runner; documents/ is the flow for documents; a CLI per flow
  benchmark/      runner, report data, PDF
  metrics/        timings, retrieval metrics, the Postgres store and its migrations
  assets/         Dagster assets, jobs and the sensor
ui/               the Streamlit app, one script per page
data/             raw/ (Dagster input), uploads/ (UI uploads), artifacts/ (per-stage outputs); git-ignored
tests/            a few pure-logic tests
```

Each stage is plain Python with no Dagster or Streamlit imports, and every stage choice is a config value. There is one Qdrant collection per experiment, named after it, so runs with different settings never mix vectors.

## More documentation

- `CLAUDE.md`: the technical reference (design rules, how each part works, the facts that are easy to get wrong).
- `PLAN.md`: the plan for the current work (the BM25 option on Upload and the chatbot, Phases 14 to 18), the ideas and the deferred items.
- `COMPLETED_PLAN.md`: the history of Phases 0 to 13.
