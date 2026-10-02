# RAG-Dagster

A lab for experimenting with PDF ingestion (text and tables) into a vector database.
Docling parses documents, Ollama embeds them, Qdrant stores and searches them, Dagster orchestrates the pipeline, and PostgreSQL keeps the metrics.

**Status: Phases 0 to 4 done.** The Docker stack (Qdrant, Postgres, Dagster) runs. Config, the Ollama embedder, the Qdrant store, the metrics store with migrations and their Dagster resources exist, PDF parsing (a sensor registers PDFs in `data/raw/` as partitions; `parsed_document` parses one with Docling) chunking (the `chunks` asset, five strategies), embedding (`embeddings`) and Qdrant indexing (`qdrant_index`). `ingest_job` runs all four stages for a document partition. Search (CLI and metrics logging) and benchmarking are still planned. `PLAN.md` is the build plan; update this file as phases land so that it describes what is actually in the repo.

## Architecture

```
data/raw/*.pdf -> parse (Docling) -> chunk -> embed (Ollama) -> index (Qdrant) -> search + benchmark
                                                                                          |
                                                                          metrics -> PostgreSQL (rag_metrics)
```

| Component | Where it runs | Address |
|---|---|---|
| Dagster webserver, daemon, code location | Docker on this Windows machine | http://localhost:3000 |
| Qdrant | Docker on this Windows machine | HTTP `6333` (dashboard at `/dashboard`), gRPC `6334` |
| PostgreSQL | Docker on this Windows machine | `5432`; databases `dagster` (Dagster's own storage) and `rag_metrics` (ours) |
| Ollama with `qwen3-embedding:0.6b` | MacBook, reached over Tailscale | `http://<mac-tailscale-ip>:11434` |

Docling runs in-process inside the Dagster code-location container. This machine has no NVIDIA GPU, so everything local is CPU-only.

## Scope

- **In:** PDF documents; text and table content.
- **Out for now:** image and figure ingestion, and OCR. Pictures are dropped at parse time, and picture and OCR features in Docling stay off (`do_ocr = False`) and are not exposed as config. Only PDFs with a text layer are supported. Do not add image handling or OCR unless asked.

## Working style

This project is kept minimal. Prefer the smallest change that works, and do not add code, config, abstractions or docs that nothing needs yet.

- **Few tests.** Write tests only for pure logic where a silent bug would corrupt results: retrieval metric formulas (Recall, MRR, nDCG), vector truncation and re-normalisation, and config hashing. No tests for Dagster assets, resources, Docling, Docker or anything needing Qdrant, Postgres or Ollama, and no mocks or fakes of them. Do not add test coverage to a change unless asked.
- **No re-testing after minor edits.** After a small change (rename, config tweak, comment, log line, doc edit, small refactor) do not run the test suite, rebuild containers or re-materialise assets. Run something only when a change alters the logic of one of the tested areas above, or when asked. Never rerun a passing check "to be sure".
- **Verify by running the real thing, once.** When a phase is finished, confirm it with a single real run (the "Done when" line in `PLAN.md`). Do not build extra verification around it.
- **Keep dependencies and services few.** Do not add a library, container or tool without a stated need in `PLAN.md`.

## Design rules

- **This is a lab.** Every stage choice (Docling options, chunker, embedding dimension, distance metric) is a config value, never a constant. A new experiment is a new config, not a code edit.
- **Stages are modular.** Parsing, chunking, embedding, storage and search live in separate packages under `src/rag_lab/` as plain Python with no Dagster imports. Dagster assets in `assets/` are thin wrappers that call them. Search must be usable without Dagster running.
- **One Qdrant collection per experiment.** The collection name comes from the experiment name, so runs with different settings never mix vectors. Each chunk also stores the config hash that produced it.
- **Vectors are supplied by us.** Qdrant only stores and searches vectors; embeddings come from the Ollama resource. Do not use the Qdrant client's built-in FastEmbed helpers for dense vectors.
- **Metrics go to PostgreSQL.** Ingestion, search and benchmark metrics are written to the `rag_metrics` database, keyed by experiment and config hash. Dagster asset metadata shows a summary of the same numbers; Postgres is the source of truth for comparisons.
- **Never write to the `dagster` database.** It belongs to Dagster. Our tables live only in `rag_metrics`.
- **Intermediates are kept on disk** under `data/artifacts/<experiment>/<stage>/` so that a stage can be re-run without redoing the ones before it.

## Key technical facts

- `qwen3-embedding:0.6b` outputs 1024-dimension vectors and supports smaller dimensions through Matryoshka truncation. Vectors must be L2-normalised again after truncation.
- Qwen3 embedding is asymmetric: **queries** get an instruction prefix (`Instruct: <task>\nQuery: <text>`), **documents** are embedded as-is. Mixing these up silently lowers retrieval quality.
- Tables are embedded as text (Markdown serialisation). Each chunk carries a `modality` of `text` or `table`.
- Chunking strategies are `hybrid`, `hierarchical`, `fixed`, `recursive` and `semantic`, registered with `@register("name")` in `chunking/`. `fixed`, `recursive` and `semantic` split text only and handle tables themselves by `table_handling`; `hybrid` and `hierarchical` are Docling's chunkers (Markdown table serialiser, `row-wise` unsupported). `overlap` applies to `fixed` only.
- `semantic` chunking calls Ollama for sentence embeddings (document mode, no query prefix), so it is the only chunk strategy that needs the Mac. Limits are measured in tokens with the Qwen3 tokenizer, not characters.
- Use Ollama's `/api/embed` endpoint (batch input), not the older `/api/embeddings`.
- The Python client is `qdrant-client`. Search goes through `query_points`.
- With the cosine metric, Qdrant's `score` is the cosine similarity itself (higher is better). Report `similarity = score` and `distance = 1 - score`.
- Qdrant point IDs must be unsigned integers or UUIDs. Chunk IDs are turned into deterministic UUIDs (uuid5) so re-ingestion overwrites instead of duplicating.
- Filtering on a payload field (`doc_id`, `modality`, `source_file`) needs a payload index on that field; create them when the collection is created.
- Qdrant storage must be a Docker named volume. A bind mount to the Windows filesystem can corrupt its data.
- Containers reach the Mac by Tailscale IP. MagicDNS names may not resolve inside containers, so `OLLAMA_BASE_URL` uses the `100.x.y.z` address.
- Ollama on the Mac must listen on all interfaces (`OLLAMA_HOST=0.0.0.0`), otherwise it only answers on localhost.
- A document's id is the first 16 hex characters of the SHA-256 of its bytes; it is also the Dagster partition key. Renaming a PDF does not re-ingest it.
- The code container takes about 50 s to start because it imports Docling. Reload the code location in the UI if it shows "could not reach user code server" right after `docker compose up`.
- The `model_cache` volume (mounted at `/root/.cache`) holds Docling's models. Ad hoc `uv pip install` in that container needs `UV_NO_CACHE=1`, or uv's cache (GBs) lands there too.
- Run config is set once per run on the shared `experiment` resource, not per asset: launchpad key `resources.experiment.config` with `name` (required) and optional `parse`, `chunk`, `embed`, `index` sections. Assets read it with `experiment.config()`. Over GraphQL: `{"resources": {"experiment": {"config": {...}}}}`.
- Ingest end to end: launch `ingest_job` for a partition with `resources.experiment.config` set. Changing `embed.dimension` needs a new experiment name, because a collection's vector size is fixed.
- A stage reads the previous stage's files from `data/artifacts/<experiment>/<stage>/`, so `chunks` needs `parsed_document` run first under the same experiment name (selecting both assets in one run does this).
- Chunk sizes are counted with the Qwen3 Hugging Face tokenizer (`EmbedConfig.tokenizer`), cached in the `model_cache` volume. Headings are embedded with each chunk and use part of `max_tokens`.
- Semantic chunking caches sentence-group embeddings in `data/artifacts/_cache/embeddings/`, shared across experiments.
- The Postgres init script that creates `rag_metrics` only runs when the data volume is first created. Schema changes go through new numbered files in `src/rag_lab/metrics/migrations/`, never by editing an applied one.
- `sqlalchemy<2.1` is pinned: SQLAlchemy 2.1 switches the default Postgres driver to psycopg 3, which breaks dagster-postgres.
- Running tests without a local `uv`: `docker compose run --rm --no-deps -e UV_NO_CACHE=1 -v ./tests:/app/tests dagster-code sh -c "uv pip install --system -q pytest && python -m pytest /app/tests -q"`. In Git Bash prefix it with `MSYS_NO_PATHCONV=1`.
- Retrieval quality metrics need expected results in `eval/queries.yaml`. Until those are filled in, the benchmark records latency and scores only and leaves quality metrics null.

## Layout (planned)

```
docker-compose.yml
.env.example                 # OLLAMA_BASE_URL, POSTGRES_USER/PASSWORD (copy to .env)
                             # QDRANT_URL and METRICS_DATABASE_URL are set in docker-compose.yml
docker/dagster/              # Dockerfile, dagster.yaml, workspace.yaml
docker/postgres/init/        # creates the rag_metrics database on first start
src/rag_lab/
  definitions.py             # Dagster Definitions entry point
  config.py                  # experiment + stage config models
  resources/                 # OllamaResource, QdrantResource, MetricsStoreResource
  parsing/                   # Docling converter factory and options
  chunking/                  # segment.py, sectioned.py, one module per strategy, base.py registry, tokens.py
  embedding/                 # ollama.py (embedder), vectors.py, cache.py (on-disk embedding cache)
  storage/                   # Qdrant collection setup and upsert
  search/                    # vector search + CLI
  metrics/                   # timing, latency stats, retrieval metrics, Postgres store,
                             # migrations/*.sql (numbered; applied at start-up by MetricsStoreResource)
  assets/                    # Dagster assets, jobs, sensors
data/raw/                    # drop PDFs here (git-ignored)
data/artifacts/              # per-stage outputs (git-ignored)
eval/queries.yaml            # query set; expected results to be filled in later
tests/                       # a few pure-logic tests only (see "Working style")
```

## Commands (planned)

```powershell
docker compose up -d --build          # start the stack
docker compose logs -f dagster-code   # code location logs
docker compose down                   # stop; add -v to wipe Qdrant and Postgres volumes
docker compose exec postgres psql -U <user> -d rag_metrics   # inspect metrics
uv run pytest                         # the few pure-logic tests; no containers needed
uv run python -m rag_lab.search "query text" --experiment <name> --top-k 5
```

## Conventions

- Python 3.12, managed with `uv`. Ruff for lint and format.
- Config sections in `config.py` subclass `dagster.Config` (a pydantic model), so the same classes are the launchpad's run config. Plain pydantic models cannot be nested in Dagster config.
- Secrets and host addresses come from `.env`; never hard-code the Tailscale IP or database credentials.
- The host is Windows: use PowerShell syntax in docs and scripts, and forward slashes in paths inside containers.
