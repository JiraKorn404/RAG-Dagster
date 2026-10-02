# Plan

Build a modular, experiment-friendly ingestion pipeline: PDFs in, searchable text and table vectors out, with similarity scores and a full set of performance metrics stored in PostgreSQL.

Each phase ends with something that runs. Do them in order; Phase 0 exists to find network problems before any pipeline code is written.

## Working approach

Keep the project minimal. Testing is deliberately light:

- **Tests:** only the two small files listed in Phases 1 and 6 (config hashing and vector truncation, retrieval metric formulas). Nothing else gets a test file.
- **Verification:** each phase's "Done when" is checked once, by a real run against the real stack. No mocks, no fake embedders, no integration test suite.
- **After small edits:** do not rerun tests, rebuild containers or re-materialise assets. Rerun only when the logic of a tested area changed.
- **Optional items** (in Phase 5, under "Ideas" and under "Deferred") are built only when you ask for them.

## Scope

- **Documents:** PDF.
- **Content:** text and tables. Image and figure ingestion is deferred (see "Deferred").
- **Hardware:** no NVIDIA GPU on this machine. Docling runs on CPU; embedding runs on the MacBook.

## Decisions taken

| Topic | Decision | Reason |
|---|---|---|
| Docling placement | In-process in the Dagster code-location container, CPU-only torch | Fewest moving parts. `docling-serve` as a separate container is a later experiment. |
| Experiment isolation | One Qdrant collection per experiment name | Different dimensions or chunkers cannot share a collection cleanly. |
| Partitioning | One Dagster dynamic partition per document | Re-run a single document; failures are isolated. |
| Embedding | Computed by our Ollama client; Qdrant only stores and searches vectors | Keeps embedding swappable and measurable. |
| Tables | Serialised to Markdown and embedded as text, tagged `modality = table` | The embedding model is text-only; tagging allows filtering and per-modality metrics. |
| Metrics storage | PostgreSQL in Docker, database `rag_metrics` | Queryable history across experiments. |
| Postgres instances | One container, two databases: `dagster` and `rag_metrics` | One less service to run. Separate databases keep Dagster's tables apart from ours. Splitting into two containers later only means changing a connection string. |
| Intermediates | Files under `data/artifacts/<experiment>/<stage>/` | Stages re-run independently; outputs are inspectable. |
| Search interface | Python module + CLI first; UI is optional later | Search must work without Dagster. |
| Quality metrics | Built now, computed once `eval/queries.yaml` has expected results | The query set will be filled in later; latency and score metrics work without it. |

---

## Phase 0 — Connectivity and skeleton

Goal: every service is reachable from where it needs to be reached.

- [x] On the Mac: `ollama pull qwen3-embedding:0.6b`; set `OLLAMA_HOST=0.0.0.0` (`launchctl setenv OLLAMA_HOST "0.0.0.0:11434"`, then restart Ollama); note the Tailscale IP.
- [x] From Windows PowerShell: `curl http://<mac-ip>:11434/api/tags` lists the model.
- [x] `git init`, `.gitignore` (`.env`, `data/`, `.venv/`), `pyproject.toml` with `uv`, `.env.example`.
- [x] `docker-compose.yml` with `qdrant`, `postgres`, `dagster-webserver`, `dagster-daemon`, `dagster-code`. Named volumes for Qdrant storage (never a bind mount on Windows), Postgres, and the Docling/HuggingFace model cache. Bind-mount `./data` and `./src`.
- [x] `docker/postgres/init/` script that creates the `rag_metrics` database beside `dagster`. Health check on Postgres; Dagster services wait for it.
- [x] **From inside a container**, call the Mac's `/api/embed`. This is the main risk in the project: Docker Desktop normally routes container traffic to the tailnet through the host, but it must be confirmed.
  - Confirmed working (2026-10-02): a container reaches the Mac at its Tailscale IP, so the fallback is not needed. If it ever fails: a Tailscale sidecar container, with `dagster-code` using `network_mode: service:tailscale`.
- [x] Trivial Dagster asset that embeds one string, writes one point to Qdrant, and inserts one row into `rag_metrics`.

Done when: the Dagster UI is at `localhost:3000` and the smoke asset materialises.

**Status: done (2026-10-02).** The smoke asset ran through the webserver and succeeded: 1024-dimension embedding in about 2.5 s (cold model), one point in Qdrant collection `smoke`, one row in `rag_metrics.smoke_check`. `smoke.py` and the `smoke_check` table are throwaway and go in Phase 1. Notes:
- `sqlalchemy<2.1` is pinned in `pyproject.toml`: SQLAlchemy 2.1 defaults to psycopg 3, which breaks dagster-postgres.
- Docling and the Hugging Face cache volume are wired but Docling is not installed yet (Phase 2).
- `uv` is not installed on this machine; install it (`winget install astral-sh.uv`) before running `uv run` commands or tests locally. Containers do not need it locally.

## Phase 1 — Resources, config and metrics store

- [x] `config.py`: `ExperimentConfig` holding `name`, `ParseConfig`, `ChunkConfig`, `EmbedConfig`, `IndexConfig`; a stable `config_hash()` that ignores `name`. `ChunkConfig` has the shared settings only; per-strategy settings are added in Phase 3.
- [x] `OllamaResource`: batch `embed(texts, kind="document"|"query")`, applies the query instruction prefix, optional dimension truncation with re-normalisation, retries, timeout. Returns vectors plus timings (Ollama reports `total_duration`, `load_duration`, `prompt_eval_count`).
- [x] `QdrantResource`: client lifecycle (gRPC preferred), `ensure_collection(experiment)` including payload indexes, batch upsert, delete-by-document (filter on `doc_id`).
- [x] `MetricsStoreResource`: connection to `rag_metrics`, one write method per table below. Plain Python underneath so the search CLI can use it without Dagster.
- [x] Migrations for the `rag_metrics` schema: numbered SQL files in `src/rag_lab/metrics/migrations/`, applied by `MetricsStoreResource` at start-up under an advisory lock (no Alembic).
- [x] `metrics/`: a `timed()` context manager and a percentile helper.
- [x] One small test file covering only config hashing and truncation with re-normalisation.

Done when: each piece works against the live stack. **Status: done (2026-10-02).** One real check covered migrations (idempotent), document and query embedding with timings, truncation to 256 dimensions at unit length, Qdrant over gRPC (create, double upsert gives no duplicates, payload indexes, delete by document, vector-size mismatch rejected), and a write to every `rag_metrics` table. Dagster loads the resources. The two config/truncation tests pass. The throwaway smoke asset and its table are removed, and `OLLAMA_MODEL` is gone from the env files because the model is now `EmbedConfig.model`. `definitions.py` has no assets until Phase 2.

### `rag_metrics` schema

| Table | One row per | Main columns |
|---|---|---|
| `experiments` | experiment config | `config_hash` (key), `name`, `config` (jsonb), `created_at` |
| `ingestion_stage_metrics` | document × stage × run | `config_hash`, `doc_id`, `stage` (`parse` / `chunk` / `embed` / `index`), `dagster_run_id`, `duration_ms`, `items` (pages, chunks, vectors or points), `throughput`, `details` (jsonb), `created_at` |
| `eval_queries` | query in the query set | `query_id`, `query_text`, `expected` (jsonb, nullable), `query_set_version` |
| `benchmark_runs` | benchmark execution | `benchmark_run_id`, `config_hash`, `dagster_run_id`, `top_k`, `repeats`, `query_set_version`, `collection_point_count`, `started_at`, `finished_at` |
| `query_results` | query × repeat in a benchmark | `benchmark_run_id`, `query_id`, `repeat`, `embed_ms`, `search_ms`, `total_ms`, `hits` (jsonb: chunk id, doc, page, modality, distance, similarity, rank), `first_relevant_rank` (nullable) |
| `benchmark_metrics` | metric value in a benchmark | `benchmark_run_id`, `metric`, `k` (nullable), `modality` (nullable), `value` |
| `search_log` | ad-hoc search from the CLI | `config_hash`, `query_text`, `top_k`, `embed_ms`, `search_ms`, `total_ms`, `top_similarity`, `created_at` |

`benchmark_metrics` is deliberately long-format so a new metric needs no schema change.

## Phase 2 — Parsing with Docling

- [x] `ParseConfig` fields, each mapped onto Docling's PDF pipeline options:
  - OCR is fixed off (`do_ocr = False`) and not exposed. Only PDFs with a text layer are supported; scanned PDFs will parse to little or no text.
  - `do_table_structure`, `table_mode` (`fast` / `accurate`), `table_cell_matching`
  - `do_formula_enrichment`, `do_code_enrichment`
  - `num_threads` (CPU), `document_timeout`
  - Picture options (image generation, description, classification) are fixed off and not exposed.
- [x] `parsing/`: `build_converter(ParseConfig)` and `parse(path) -> ParsedDocument`. Writes the Docling JSON and a Markdown export to the artifacts folder.
- [x] Sensor on `data/raw/*.pdf` that registers a dynamic partition per new file (keyed by content hash, so a renamed file is not re-ingested).
- [x] `parsed_document` asset. Metrics to Postgres and asset metadata: pages, tables found, parse seconds, pages per second, characters extracted per page, Markdown preview. The per-page character count makes a scanned PDF (near zero) easy to spot.
- [x] Confirm Docling's model downloads land in the cache volume, not the container layer.

Done when: dropping a PDF in `data/raw/` produces parsed output, and changing `table_mode` in the launchpad changes the result.

**Status: done (2026-10-02).** Checked once against the live stack with two generated PDFs (`data/raw/sample-report.pdf`: 2 pages, headings, one table; `sample-scanned.pdf`: image only):
- The sensor registered both files as partitions within one tick, keyed by content hash.
- Launched from the webserver with run config, `parsed_document` wrote `<doc_id>.json`, `.md` and `.meta.json` under `data/artifacts/<experiment>/parse/` and one `parse` row per run in `ingestion_stage_metrics`. The table came out as a clean 5 by 4 Markdown table.
- `table_mode` `accurate` versus `fast` gave different config hashes and timings (18.4 s versus 12.4 s for the same PDF) but identical Markdown for this simple table. Expect differences only on harder tables.
- The scanned PDF parsed to 0 characters and logged the scanned-PDF warning.
- Models downloaded into the `model_cache` volume (about 500 MB), not the image. The first run took 218 s to load models; later runs 13 to 22 s.

How to use it: add a PDF to `data/raw/`, wait for the sensor (30 s), then in the Dagster UI materialise `parsed_document` for that partition. Run config is set once per run on the shared `experiment` resource (see Phase 3): `resources.experiment.config` with `name` and optional `parse`, `chunk`, ... sections.

Notes:
- The sample PDFs were generated because the earlier `report.pdf` and `scan.pdf` had disappeared from `data/raw/`. Replace them with real documents.
- The code location takes about 50 s to start (it imports Docling), so the Dagster UI can show a brief "could not reach user code server" error after `docker compose up`; reload the location.
- Config sections now inherit `dagster.Config` so one definition serves Python code and the launchpad (plain pydantic models are not accepted as nested Dagster config).

## Phase 3 — Chunking

The `fixed`, `recursive` and `semantic` strategies run in two steps, so that tables stay whole and every chunk keeps its page and headings:

1. **Segment** the Docling document into blocks: text blocks (with their headings and page) and table blocks. Picture items are skipped.
2. **Split** the text blocks with the chosen strategy. Table blocks bypass the strategy and follow `table_handling`.

`hybrid` and `hierarchical` are Docling's own chunkers and do both steps themselves. Docling's default table text ("row, column = value" triplets) embeds poorly, so both are given a Markdown table serialiser. They cannot honour `row-wise` (it falls back to Markdown, with a warning), `skip` drops table-only chunks, and `hybrid` with `merge_peers` may merge a small table into its neighbouring text. Set `merge_peers` to false for isolated tables.

| Strategy | How it splits | Own settings |
|---|---|---|
| `hybrid` | Docling `HybridChunker`: follows document structure, then splits or merges to fit the token limit | `merge_peers` |
| `hierarchical` | Docling `HierarchicalChunker`: one chunk per structural element | none |
| `fixed` | Fixed token windows | none |
| `recursive` | Tries separators in order (paragraph, line, sentence, word), recursing to the next one only for pieces still over the limit | `separators` |
| `semantic` | Splits into sentences, embeds them, and breaks where the distance between neighbouring sentence groups jumps | `buffer_size` (sentences per group), `breakpoint_type` (`percentile` / `stddev` / `absolute`), `breakpoint_threshold`, `min_tokens` |

- [x] `ChunkConfig`: `strategy`, shared settings (`max_tokens`, `overlap`, `table_handling` (`markdown` / `row-wise` / `skip`), `include_headings_in_text`), and nested settings for `hybrid`, `recursive` and `semantic`. Only the chosen strategy's settings count: the others are left out of `config_hash()`. `overlap` is used by `fixed` only; the other strategies ignore it and log a warning. Headings are part of the embedded text and use up part of the `max_tokens` budget.
- [x] `chunking/base.py`: a `Chunker` protocol and a registry keyed by strategy name. Adding a strategy means adding one file and one registry entry.
- [x] `chunking/segment.py`: the shared segmenter.
- [x] `recursive`: own implementation measured in tokens, not characters. No splitter library dependency.
- [x] `semantic`:
  - Sentence embeddings come from `OllamaResource` in document mode (no query prefix), batched per document.
  - A semantic chunk over `max_tokens` is split again with `recursive`; one under `min_tokens` is merged into its neighbour.
  - The breakpoint threshold is taken over the whole document, not per section, so a short section is not forced to split. The threshold's meaning depends on `breakpoint_type` (percentile 0-100, standard-deviation multiples, or an absolute cosine distance).
  - Sentence-group embeddings are cached on disk (`data/artifacts/_cache/embeddings/`), keyed by model, dimension and text, and shared across experiments, so changing only a breakpoint setting does not call Ollama again.
- [x] Chunk model: `chunk_id` (deterministic), `doc_id`, `text`, `modality` (`text` / `table`), `page`, `headings`, `bbox`, `token_count`, `strategy`.
- [x] Token counting uses the Qwen3 tokenizer so `max_tokens` means what the embedder sees.
- [x] `chunks` asset. Metrics: chunk count, counts per modality, token-length min/mean/max, chunking seconds, a sample chunk. For `semantic`, also sentences embedded, embedding seconds and breakpoints found.
- [x] No per-strategy tests. Check strategies by running them on a real PDF and reading the chunk metrics.
- [x] One `ExperimentResource` (nested config) shared by every asset, so `resources.experiment.config` is set once per run instead of once per asset. `parsed_document` was switched to it.

Done when: the same PDF chunked with all five strategies gives five comparable rows in `ingestion_stage_metrics`.

**Status: done (2026-10-02).** Checked once against the live stack with a generated 2-page PDF (`data/raw/handbook.pdf`: five topic sections of two paragraphs, a closing 12-row table). Parse plus chunk ran for five experiments with `max_tokens` 128:

| Strategy | Chunks (text / table) | Tokens min / mean / max |
|---|---|---|
| hybrid | 13 (10 / 3) | 59 / 85 / 127 |
| hierarchical | 12 (11 / 1) | 13 / 88 / 294 |
| fixed (overlap 16) | 14 (11 / 3) | 13 / 85 / 128 |
| recursive | 14 (11 / 3) | 13 / 80 / 120 |
| semantic (percentile 75) | 15 (12 / 3) | 13 / 75 / 120 |

- Every strategy gave a `chunk` row in `ingestion_stage_metrics`, with counts, token statistics and (for `semantic`) sentences, embeddings, cache hits, threshold and breakpoints.
- No chunk is over the limit except `hierarchical`, which has no size limit and keeps the whole table as one chunk (294 tokens).
- The oversize table was split by rows with the header repeated in each piece. Headings and page numbers are on every chunk.
- Semantic: 41 sentence groups embedded in 5.7 s. Rerunning with threshold 90 embedded none (41 cache hits) and found different breakpoints (4 instead of 9). `fixed` overlap was confirmed by the last words of one chunk reappearing at the start of the next.
- A `semantic` run needs the Mac; the other four do not.

Notes:
- Each experiment has its own `parse/` folder, so comparing chunkers means re-parsing the PDF once per experiment (about 20 s each). Fine for a lab; if it becomes annoying, let a chunk run read another experiment's parse output.
- Docling treated a larger first test table (60 rows across a page break) as a picture, and found no table. This is Docling's layout model, not the chunker; a table that Docling does not detect is never chunked as a table.
- `config_hash` changed with the new settings, so earlier experiment rows have different hashes. `EmbedConfig` gained `tokenizer` (Hugging Face id of the Qwen3 tokenizer used to count tokens).
- One extra test assertion covers the new hashing rule (3 tests pass); the chunkers have none, as planned.

## Phase 4 — Embedding and indexing

- [x] `embeddings` asset: batches chunks through `OllamaResource`, stores vectors as `embed/<doc_id>.npy` (float32, one row per chunk) in the experiment's artifacts folder. Metrics: batch size, total seconds, chunks per second, tokens per second, model load time, dimension.
- [x] Collection setup: vector size from `EmbedConfig`, cosine distance, HNSW parameters (`m`, `ef_construct`). Quantisation is not built; it is a Phase 7 option. Payload holds the chunk model fields plus `experiment`, `config_hash`, `source_file`, `ingested_at`. Payload indexes on `doc_id`, `modality`, `source_file`.
- [x] `qdrant_index` asset: deletes the document's existing points in that collection, then batch-upserts with deterministic UUIDs (uuid5 of the chunk ID) and `wait=True` so re-runs are idempotent and timings are honest. Metrics: points written, batches, upsert seconds, points per second, collection size. A failed batch raises out of the asset, so there is no partial-failure count.
- [x] `ingest_job` covering parse through index for selected partitions.

Done when: one job run takes a PDF from `data/raw/` to queryable points, running it twice leaves the same point count, and each stage has a row in `ingestion_stage_metrics`.

**Status: done (2026-10-02).** Checked against the live stack with `handbook.pdf`, launching `ingest_job` through the webserver:
- One run produced a row for each of `parse`, `chunk`, `embed` and `index` in `ingestion_stage_metrics`. The embed row has dimension, tokens, tokens per second and model load time; the index row has points written and collection size.
- The collection was created with 1024-dimension cosine vectors, HNSW m=16 and ef_construct=100, and payload indexes on `doc_id`, `modality` and `source_file`.
- Running the same job again left the same point count (6).
- An experiment with `embed.dimension` 256 and the `recursive` chunker gave a 256-dimension collection with 14 points, so truncation works through the assets.
- A query embedded with the instruction prefix found the right chunks: the Dagster section for a Dagster question (similarity 0.64) and the table chunk for a benchmark question (0.71).

Notes:
- To ingest: in the Dagster UI launch `ingest_job` for a document partition and set `resources.experiment.config` (`name` plus any `parse`, `chunk`, `embed`, `index` settings). Individual assets can still be run alone, in order.
- Changing `embed.dimension` under an existing experiment name fails at indexing (vector size mismatch) after the embeddings are rewritten, so use a new experiment name for a new dimension.
- `ingest_job` is partitioned per document; selecting several partitions in the UI launches a backfill with one run each.

## Phase 5 — Search with scores

- [x] `search/`: `search(query, experiment, top_k, filters)` embeds the query with the instruction prefix and runs `query_points`, with `hnsw_ef` taken from config. Each hit returns text, source file, page, modality, `similarity` (Qdrant's cosine score) and `distance` (`1 - similarity`).
- [x] Per-query timing breakdown: embed ms, Qdrant ms, total ms.
- [x] CLI: `python -m rag_lab.search "..." --experiment <name> --top-k 5 [--modality table] [--json] [--no-log]`. Writes to `search_log` unless `--no-log`.
- [ ] Optional: `--exact` flag (brute-force search) to measure how much recall the HNSW index gives up.
- [ ] Optional: hybrid mode for comparison. In Qdrant this needs a sparse (BM25) vector stored on each point at index time, fused with the dense result at query time, so it touches Phase 4 as well.

Done when: the CLI returns ranked hits with similarity and distance, filters by modality, and logs to `search_log`. **Status: done (2026-10-02).** Checked once against the live stack with the `docs` experiment (`aiayn.pdf`):
- "How does multi-head attention work?" returned the Multi-Head Attention section first (similarity 0.732, 339 ms embed, 16 ms Qdrant, 355 ms total) and wrote one `search_log` row.
- `--modality table --json --no-log` returned only table chunks (BLEU query: Table 2 first, 0.636) as JSON and wrote no row.
- `--exact` and hybrid mode (the optional items) are not built.

How to use it: `docker compose exec dagster-code python -m rag_lab.search "..." --experiment <name> --top-k 5`. The experiment's settings (embedding model, dimension, query instruction, `hnsw_ef`) are read from the newest row for that name in `rag_metrics.experiments`, so ingest at least `parsed_document` and `chunks` under the name first. `rag_lab.search.search(query, config, embedder, store, ...)` is plain Python for the Phase 6 benchmark to reuse. Running outside the container needs `OLLAMA_BASE_URL`, `QDRANT_URL` and `METRICS_DATABASE_URL` set.

## Phase 5a — Notebook search helper

Goal: try similarity searches from a Jupyter notebook with one function call, instead of the CLI's `docker compose exec ... --experiment ... --top-k ...` line. Nothing is rebuilt: the notebook is a thin front end over the Phase 5 `search()`.

Where it runs: the notebook kernel runs on the Windows host using the existing `.venv`, not in a container (a Jupyter container would be one more service). The host already reaches everything: Qdrant on `localhost:6333/6334` and Postgres on `localhost:5432` are published by `docker-compose.yml`, and Ollama is reached at the Tailscale IP from `.env`. Open the notebook in VS Code (or `uv run jupyter lab`) and pick the `.venv` kernel.

- [x] `search/quick.py` with one function, `ask(query, experiment, top_k=5, modality=None, log=False)`:
  - Builds the Ollama, Qdrant and metrics clients itself, so the notebook needs no setup cell. Settings come from the environment, with host defaults when unset: `QDRANT_URL` falls back to `http://localhost:6333`, `METRICS_DATABASE_URL` is built from `POSTGRES_USER` and `POSTGRES_PASSWORD` at `localhost:5432/rag_metrics`, and `OLLAMA_BASE_URL` has no default. The three values are read from `.env` if the variables are not already set (a few lines of parsing, no `python-dotenv`).
  - Looks the experiment's settings up in `rag_metrics.experiments` (the same lookup as the CLI), so the embedding model, dimension and query instruction always match what was ingested. The lookup moves from `__main__.py` into a shared helper so the CLI and `ask()` use one copy.
  - Calls `search()`, prints one compact line per hit (rank, similarity, modality, file and page, first 100 characters) followed by the three timings, and returns the `SearchResult` so cells can inspect `result.hits[0].text`.
  - `log=True` writes to `search_log`; the default is off, so notebook experiments do not fill the log.
- [x] `notebook/search.ipynb`: a first cell `from rag_lab.search.quick import ask`, then example cells: a plain query, a `modality="table"` query, and the same query against two experiments to compare scores. Outputs are cleared before committing.
- [x] Add `ipykernel` to the `dev` dependency group in `pyproject.toml`. This is the stated need for it; nothing else is added.
- [x] `CLAUDE.md`: add `notebook/` to the layout and the `ask()` line to the commands.

Not included: a rewrite of the CLI (it stays as it is), `--exact` or hybrid search, filters other than `modality`, pandas tables or charts, and tests (the function is glue over code that is already checked).

Done when: after `docker compose up`, `ask("...", "docs")` in the notebook prints ranked hits from the host with no extra setup, and the same query gives the same top hit and similarity as the CLI. One real run, no more.

Open points to confirm before building:
- The kernel runs on the host. If you would rather run it in a container, that needs a `jupyter` service (or an install in `dagster-code`) and adds weight to the Docker setup.
- `ask()` reads `.env` for the Ollama address, because the Tailscale IP is not in the host's environment by default.

**Status: done (2026-10-02).** Checked once: from the host `.venv`, `ask("How does multi-head attention work?", "docs")` returned the same top hit as the CLI (similarity 0.732, 62 ms total) and the `modality="table"` query returned only table chunks (0.636 first). The CLI still works after the shared-lookup refactor. Notes:
- `ipykernel` ended up in the main `dependencies` of `pyproject.toml`, not the dev group.
- `.env` has `OLLAMA_BASE_URL=http://host.docker.internal:11434` (Ollama on this machine), which only resolves inside containers. `ask()` swaps that name for `localhost`.
- Postgres and Qdrant default to `127.0.0.1`, not `localhost`: Docker publishes them on IPv4 only and Postgres connections hung on the IPv6 attempt.
- A hit whose text has characters the Windows console cannot encode (such as `ϵ`) makes `print` fail in a plain cp1252 terminal. Notebooks print UTF-8, so it only matters outside them.

## Phase 6 — Metrics and benchmarking

All metrics below are implemented in this phase. Quality metrics are computed only for queries that have expected results; otherwise they are stored as null and the run still completes.

- [x] `eval/queries.yaml` format: `id`, `query`, optional `expected` list of `{source_file, page}` or `{chunk_id}`, optional `modality`. Ship it with placeholder queries and empty `expected`. Loading it syncs `eval_queries`.
- [x] `metrics/retrieval.py`, with one small test of the formulas against a hand-computed case (the one place a silent error would look plausible).
- [x] `search_benchmark` asset per experiment: warm-up pass, then N repeats of the query set. Writes `benchmark_runs`, `query_results`, `benchmark_metrics`.
- [x] `experiment_summary` asset: a SQL view over `rag_metrics` giving one row per experiment, shown as a Markdown table in the Dagster UI.

| Group | Metrics |
|---|---|
| Search latency | p50, p95, p99, mean, min, max for embed, Qdrant search and total; queries per second; cold versus warm first query |
| Similarity scores | top-1 similarity, mean similarity over top-k, gap between rank 1 and rank k, score distribution per experiment |
| Retrieval quality (needs expected results) | Hit rate@k, Recall@k, Precision@k, MRR, MAP, nDCG@k for k in 1, 3, 5, 10; also split by modality (text versus table) |
| Ingestion performance | parse seconds and pages per second; chunks per document; embed chunks per second and tokens per second; model load time; index points per second; end-to-end seconds per document |
| Index state | point count, indexed vector count, vector dimension, segment count |

Done when: two experiments (for example table mode fast versus accurate, or two chunking strategies) can be compared with one SQL query, and filling in `expected` then re-running the benchmark produces quality metrics with no code change.

**Status: done (2026-10-02).** Checked against the live stack with the `docs` experiment (`aiayn.pdf`, 41 points):
- `search_benchmark` for `docs` (default 10 top-k, 5 repeats, the 3 shipped queries) wrote 1 `benchmark_runs` row, 15 `query_results` and 69 `benchmark_metrics`. Total latency p50 35 ms, p95 56 ms, 26.9 queries per second; the first query cost 172 ms cold and 37 ms warm. Top-1 similarity averaged 0.704. The 36 quality metrics (overall and for `table`) were stored as null because the shipped queries have no expected results.
- `SELECT * FROM experiment_summary` gives one row per experiment with ingestion performance (15 pages parsed at 0.47 pages per second, 41 chunks, 3.3 chunks and 887 tokens per second embedding, 10.5 s model load, 47 s per document) next to the benchmark columns.
- Filling in `expected` and re-running gave quality metrics with no code change, checked with a temporary query file: `attention` pointing at page 4 got rank 1, and a `bleu-table` query with one real and one made-up page got recall 0.5 and nDCG@5 0.613, which matches the hand calculation. Those test rows were deleted afterwards.
- The retrieval formulas have one test (`tests/test_retrieval.py`, hand-computed case including duplicate hits on one expected page); all 4 tests pass.

How to use it:
- Launch `search_benchmark` (and `experiment_summary`) from the Dagster UI with run config `resources.experiment.config.name: <experiment>`; optional `ops.search_benchmark.config` with `top_k`, `repeats`, `queries_file`. From a shell: `docker compose exec dagster-code dagster asset materialize -m rag_lab.definitions --select search_benchmark,experiment_summary --config-json '{"resources":{"experiment":{"config":{"name":"docs"}}}}'`.
- The benchmark takes only the experiment name from the run config. Its settings are read from the `experiments` table (as the search CLI does), so a benchmark always shares the config hash of the ingestion it measures. Benchmarking with a changed `index.hnsw_ef` is therefore not possible from the run config.
- To get quality metrics, add `expected` to entries in `eval/queries.yaml` (`{source_file, page}` or `{chunk_id}`) and re-run. The query set version is a hash of the file, so each edit is a new version. A query's `modality` restricts its search and also splits the metrics: the `table` rows cover queries marked `modality: table`.

Notes:
- Each expected item is matched by at most one hit (the best ranked), so two chunks from the same expected page cannot push a score above 1. MRR and MAP cover the whole retrieved top-k list, so their `k` is null. Precision, recall, hit rate and nDCG are stored for k in 1, 3, 5, 10 up to `top_k`.
- "Cold versus warm first query": `first_query_cold_ms` is the first query of the warm-up pass, `first_query_warm_ms` the mean of the same query across the timed repeats.
- `experiment_summary` is a SQL view (migration `0002_experiment_summary.sql`) over the latest run of each stage per document and the latest finished benchmark. The asset shows it transposed (one column per experiment) as Markdown.
- `indexed_vectors_count` is 0 for a small collection: Qdrant does not build the HNSW index below its indexing threshold, so a small lab collection is searched by brute force. This also means `--exact` would show no difference until the collection is large.
- `docker-compose.yml` mounts `./eval` into `dagster-code`; recreate that container (`docker compose up -d dagster-code`) once after pulling this change.
- A Dagster asset takes its run config only through a parameter named `config`, so the benchmark settings are `config: BenchmarkConfig`.

## Phase 6a — Notebook summary helper

Goal: see the `experiment_summary` view from the notebook with one call, `summary()`, instead of the `docker compose exec postgres psql ...` line or opening the Dagster UI. Same idea as Phase 5a: a thin helper over code that already exists, run from the host `.venv`.

- [x] `metrics/summary.py`: `summary_markdown(columns, rows)` builds the transposed Markdown table (one column per experiment, one row per metric). It moves out of the `experiment_summary` asset, which then calls it, so the notebook and the Dagster UI show the same table from one copy.
- [x] `search/quick.py`: `summary(names=None)`. It fetches the view through `MetricsStore.experiment_summary()` and renders the table in the notebook (`IPython.display.Markdown`; IPython comes with `ipykernel`). `names` is an optional list of experiment names to show; the default is all. The connection settings code that `ask()` already has (`.env`, `127.0.0.1` defaults) is split out into one helper that both functions use. It returns nothing, so the table is not shown twice.
- [x] `notebook/search.ipynb`: a final section with `from rag_lab.search.quick import ask, summary` and a `summary()` cell. Existing cells stay as they are.
- [x] `CLAUDE.md`: mention `summary()` next to `ask()`.

Not included: a pandas DataFrame (no new dependency), charts, or a way to run a benchmark from the notebook (benchmarks stay Dagster runs).

Done when: `summary()` in the notebook shows the same numbers as `SELECT * FROM experiment_summary`. One real run, no tests (the table builder moves unchanged, and there is no new logic to check).

**Status: done (2026-10-02).** Checked once: from the host `.venv`, the table `summary()` builds has the same numbers as `SELECT * FROM experiment_summary` for `docs` (15 pages, 41 chunks, search p50 35.2 ms, top-1 similarity 0.704, quality columns empty), and `summary(["nope"])` raises a clear error. The `experiment_summary` asset still materialises after the table builder moved. I did not open the notebook itself; in Jupyter `display(Markdown(...))` renders the table, while a plain script only prints the object.

How to use it: in `notebook/search.ipynb`, `from rag_lab.search.quick import ask, summary`, then `summary()` or `summary(["docs", "other"])`. It reads the view directly, so it is always current and needs no re-materialising.

## Phase 7 — Embedding model × chunking strategy comparison

Goal: find out which embedding model and which chunking strategy retrieve best on your documents, and see the answer in one place. The matrix is 3 models (`qwen3-embedding:0.6b`, `:4b`, `:8b`) × 5 strategies (`hybrid`, `hierarchical`, `fixed`, `recursive`, `semantic`) = 15 experiments over the same documents and the same query set. Two UIs sit on top: a dashboard over `rag_metrics`, and a page for trying a query against many experiments at once.

### Decisions

| Topic | Decision | Reason |
|---|---|---|
| What varies | Only the model and the strategy. Everything else (`max_tokens`, `overlap`, `table_handling`, HNSW, batch size, query instruction, parse settings) is one shared block in `experiments/matrix.yaml`. | A difference in the results must come from the model or the chunker, not from a stray setting. |
| Experiment names | `<model label>-<strategy>`, with labels set in the matrix file: `q3-0-6b-hybrid`, `q3-4b-fixed`, `q3-8b-semantic`, and so on. | Names double as Qdrant collection names, so they must match the existing name pattern. |
| Dimension | Each model at its native size (1024, 2560, 4096). | One collection per experiment, so sizes never clash. A fixed-dimension axis (Matryoshka) is an idea, not part of this phase. |
| Tokenizer | The same Qwen3 tokenizer for all three. | Same model family, so `max_tokens` means the same thing everywhere. |
| How relevance is judged | Expected items are `{source_file, page}`, optionally with `contains: "<text>"`. `chunk_id` is not used. | Chunk ids differ between chunkers, so they cannot be compared. A page is coarse (two chunkers both hit the right page), so `contains` adds a case-insensitive text check on the hit. |
| What is compared across models | Quality metrics (recall, MRR, nDCG) and ranks. Similarity scores only within one model. | Cosine scores of different models are on different scales; a higher score from the 8b model does not mean a better result. |
| Parsing | Each experiment parses the PDF again, as today. | Simplest, and keeps every experiment's metrics complete. About 20 to 30 s per experiment per document. Reusing one parse across experiments is an idea. |
| How the matrix runs | A command inside `dagster-code` that calls `dagster.materialize` once per experiment and document. | Runs show up in the Dagster UI, with no new service and no GraphQL client. |
| UI | One Streamlit app with two pages (dashboard, query), run as a Docker service `ui` built from the shared image, at `http://localhost:8501`. | One new dependency and one new service. In the container it already has the Qdrant, Postgres and Ollama addresses, the same as `dagster-code`, so it needs no `.env` parsing. |
| Documents | `aiayn.pdf` only ("Attention Is All You Need", 15 pages, 4 tables). | The chosen test document. One paper means the results describe how the models and chunkers behave on this kind of paper, not on documents in general. |

### 7.1 Groundwork

- [x] On the Mac: `ollama pull qwen3-embedding:4b` and `ollama pull qwen3-embedding:8b`. Check the 8b model fits in the Mac's memory next to the others; Ollama loads one at a time, but a swap-heavy Mac will distort the latency numbers. Embed one string with each and confirm the dimensions (1024, 2560, 4096).
- [x] Document: `data/raw/aiayn.pdf`, already there and ingested as the `docs` experiment.
- [x] Query set: about 30 queries in `eval/queries.yaml` replacing the three placeholders, with `expected` filled in: roughly 20 about the text (architecture, attention, training, results) and 10 marked `modality: table` (the paper's complexity, BLEU, model-variation and parsing tables). Each expected item is `{source_file, page}` plus a `contains` snippet taken from the paper. Without expected results the comparison has only latency and scores, which cannot rank models (see the decisions).
- [x] I draft the queries from the parsed paper and check every expected item against it (the page exists and the `contains` text is on that page) before handing the file to you; an item that matches nothing would silently score zero. You review the file. The whole comparison rests on it.
- [x] `metrics/retrieval.py`: `relevance()` also accepts a `contains` key in an expected item (a substring of the hit's text). Hits carry their text into the matching only; `query_results` still stores no text. One added assertion in `tests/test_retrieval.py`, since this is the formula area the project does test.

### 7.2 Matrix runner

- [x] `experiments/matrix.yaml`: `models` (label to Ollama model), `strategies`, `shared` (the fixed settings above), and `documents` (all partitions, or a list of file names). Mounted into `dagster-code` like `eval/`.
- [x] `src/rag_lab/experiments/`: `matrix.py` expands the file into a list of `ExperimentConfig` (plain Python, no Dagster). `__main__.py` runs them: `docker compose exec -d dagster-code python -m rag_lab.experiments run experiments/matrix.yaml [--only <pattern>] [--dry-run]`.
- [x] Run order is model by model, so each model is loaded on the Mac once. For each experiment: `parsed_document`, `chunks`, `embeddings`, `qdrant_index` for every document, then `search_benchmark` once.
- [x] Resumable: an experiment that already has a finished benchmark for the current query-set version is skipped. A failed experiment is logged and the run moves on; a summary of failures is printed at the end.
- [x] `--dry-run` prints the 15 experiment names and config hashes without running anything.

### 7.3 Dashboard page

- [x] Migration `0003`: recreate `experiment_summary` with `model`, `strategy` and `dimension` columns read from `experiments.config`, so the dashboard and SQL can group by them.
- [x] `streamlit` added to the dependencies in `pyproject.toml` (its stated need); the shared image is rebuilt once (`docker compose up -d --build`). A new `ui` service in `docker-compose.yml`: the same image, `streamlit run /app/ui/app.py`, port `127.0.0.1:8501`, `./src` and `./ui` bind-mounted so edits reload, the same environment block as `dagster-code`, started after Postgres and Qdrant.
- [x] `ui/app.py` (Streamlit entry with two pages), `ui/dashboard.py`, and `ui/style.py` (the shared theme, below). The UI reads its addresses from the environment, like the search CLI.
- [x] Look and feel (applies to both pages, since "cool" is a requirement):
  - A dark theme set in `ui/.streamlit/config.toml`, with one accent colour, and a small block of custom CSS in `style.py`: a gradient page title, rounded cards, a quiet background.
  - Headline cards at the top of the dashboard (best model, best strategy, best nDCG@5, fastest search p50), each with the value large and the experiment name beneath it.
  - Charts in Altair (it comes with Streamlit; no plotting library is added), with one palette chosen for the three models and used on every chart, a single-hue scale for the heatmap grid, readable labels, and tooltips. The `dataviz` skill is used for the palette and chart rules when this is built.
  - Text and table hits in the query page are cards with a coloured badge for the modality and a similarity bar; the agreement marking below uses a second badge colour.
- [x] Dashboard contents, all read from `rag_metrics`:
  - A grid with strategy as rows and model as columns, filled with a metric you pick (nDCG@5, recall@5, MRR, search p50, embed chunks per second, and so on). This is the main view.
  - Bar charts by experiment for quality, search latency (p50 and p95) and ingestion throughput.
  - The raw `experiment_summary` table, sortable.
  - A per-query drill-down: for two chosen experiments, the first relevant rank of each query side by side, to see where one wins.
  - A query-set version selector, with a warning when the experiments shown were benchmarked on different versions.

### 7.4 Query page

- [x] `ui/query.py`: a query box, top-k, an optional modality filter, and a choice of experiments (all 15, or filter by model or strategy). Each experiment gets a column of ranked hits with rank, similarity, source file and page, and text. Hits whose page also appears in other experiments' results are marked, so agreement and disagreement are visible at a glance. Each hit also shows its `{source_file, page}` as a ready-to-paste line for `eval/queries.yaml`.
- [x] `search()` takes an optional precomputed query vector, so the page embeds the query once per distinct embedding setup (three times for the matrix, not fifteen).
- [x] Nothing is written to `search_log` from this page.

### 7.5 The comparison run

- [x] A slice first: one model and two strategies (for example `q3-0-6b` with `hybrid` and `fixed`), checked end to end through the dashboard and the query page.
- [x] Then the full 15 experiments. Read the dashboard and note what you find (best model, best strategy, whether the larger models are worth their latency and memory) in this section.

Done when: `--dry-run` lists 15 experiments, the full run leaves each with a finished benchmark on the same query-set version, the UI at `localhost:8501` shows a quality metric for every model and strategy pair in the dashboard grid, and the query page shows the same query's results from several experiments side by side. Checked once, on the real stack.

**Status: done (2026-10-02).** Checked on the real stack with `aiayn.pdf` and a 30-query set (20 text, 10 table):
- `--dry-run` lists the 15 experiments with their config hashes. The full run, started from the code container, finished all 15 in 11.7 minutes (a first slice of two experiments had been run and checked before). Every experiment has a finished benchmark on the same query-set version (`1ec485bcb7c0`).
- A coverage check confirmed that all 30 expected items are findable in the chunks of every one of the 15 experiments, so no experiment is penalised by where its chunker happened to cut.
- The UI is at `http://localhost:8501`. The dashboard grid shows nDCG@5 for every model and strategy pair; the query page shows one question across all 15 experiments, with one query embedding per model. Both pages were rendered in a real browser and checked for exceptions with Streamlit's app tester.

Results (30 queries on one paper, so read them as a guide, not a verdict):

| Model | Vector size | Mean nDCG@5 | Mean recall@5 | Search p50 | Embed tokens/s | Seconds per document |
|---|---|---|---|---|---|---|
| `qwen3-embedding:0.6b` | 1024 | 0.896 | 0.967 | 29 ms | 5436 | 36 |
| `qwen3-embedding:4b` | 2560 | 0.924 | 0.993 | 42 ms | 1870 | 41 |
| `qwen3-embedding:8b` | 4096 | 0.942 | 0.993 | 57 ms | 1199 | 45 |

| Strategy | Mean nDCG@5 | Mean MRR |
|---|---|---|
| `semantic` | 0.928 | 0.908 |
| `recursive` | 0.927 | 0.909 |
| `fixed` | 0.920 | 0.898 |
| `hierarchical` | 0.918 | 0.896 |
| `hybrid` | 0.910 | 0.896 |

- A larger model helps more than a different chunker: each step up in model size adds about 0.02 to 0.03 nDCG@5, at roughly 1.4 times the search latency and 0.3 to 0.5 times the embedding speed. The strategies span only 0.02, which is within what one or two queries can move (one query is 0.033 of recall@5), so no strategy can be declared best on this data.
- Best single experiments (nDCG@5 0.955): `q3-4b-hierarchical`, and `q3-8b` with `fixed`, `recursive` or `semantic`. `hierarchical` is the least stable: best at 4b, worst of the 8b row (0.893). It keeps formula placeholders as chunks (the `<!-- formula-not-decoded -->` chunk ranks first for the optimizer question at 0.6b).
- Recall@5 is 0.93 to 1.00 in every experiment, so the right chunk is nearly always in the top five. Most of the difference is in how high it ranks (MRR 0.855 to 0.940), not whether it is found.

Changes from the plan, and notes:
- Expected items use `{source_file, contains}` only, without `page`: Docling assigns a chunk's page differently depending on the chunker (Table 1 is on page 5 in the hybrid chunks, page 6 in the PDF), so a page match would have penalised some experiments unfairly. The 30 expected snippets were drafted from the parsed paper and checked against the chunks; they still need your review.
- The `hostenv.py` step was dropped: the UI runs in a container that already has the connection settings in its environment.
- Ollama is on this machine here (`OLLAMA_BASE_URL=http://host.docker.internal:11434`) with all three models pulled, so step 7.1's Mac check did not apply. Memory was not a problem for the 8b model.
- The runner skips ingestion when the experiment's Qdrant collection already holds the document (not when metrics rows exist), and skips an experiment whose config hash already has a finished benchmark on the current query set. This is needed because `q3-0-6b-hybrid` has the same settings, hence the same config hash, as the earlier `docs` experiment.
- That shared hash exposed a bug, now fixed: the experiments row kept the old config JSON (with the old name) when a second name claimed the hash, so a benchmark of `q3-0-6b-hybrid` silently ran against the `docs` collection. The upsert now updates the config with the name. The row was renamed to `q3-0-6b-hybrid`, the experiment was rerun, and the mislabelled benchmark run was deleted. The `docs` Qdrant collection and `data/artifacts/docs/` are now unused and can be deleted. The notebook examples use `q3-0-6b-hybrid` and `q3-0-6b-fixed`.
- `streamlit` was added to the shared image, so all Dagster services restarted once. After editing a module other than a page script (for example `ui/style.py`), restart the UI with `docker compose restart ui`.
- Migration `0003` recreates the `experiment_summary` view with `model`, `strategy` and `vector_dimension` columns.

Decided (2026-10-02): the document is `aiayn.pdf`; I draft the query set and you review it; the UI is one Streamlit app with two pages, run as a Docker service; the design should look polished (see 7.3).

## Ideas

Not planned and not in any phase. Each is built only when you ask for it.

- Reuse one parse across experiments instead of re-parsing for each (copy or point to the parse output, and copy its metrics row).
- A fixed-dimension axis in the matrix (all models truncated to the same size by Matryoshka) to separate model quality from vector size.
- Embedding models outside the Qwen3 family.
- Preset experiment configs in `experiments/*.yaml`, selectable in the launchpad.
- Reranking stage.
- Quantisation settings in Qdrant.
- `--exact` search to measure what HNSW gives up, and hybrid (dense plus BM25) search. See Phase 5.
- `docling-serve` as a separate container, compared against in-process parsing.
- A dashboard in Grafana or Metabase pointed at Postgres, in place of or beside Streamlit.
- Marking hits as expected directly in the query page and writing them to `eval/queries.yaml`.

## Deferred

- **Image and figure ingestion.** When wanted: enable picture extraction in `ParseConfig`, add a `picture` modality, and describe images with a vision model so they can be embedded as text.
- **OCR** (scanned PDFs). Docling's OCR also runs on CPU, only slowly, so this is a speed trade-off rather than a hard limit. When wanted: expose `do_ocr`, `ocr_engine`, `ocr_languages` and `force_full_page_ocr` in `ParseConfig`.
- **Other document types** (DOCX, PPTX, HTML). Docling handles them; the sensor and `ParseConfig` would need format-specific options.

## Risks

| Risk | Mitigation |
|---|---|
| Containers cannot reach the Mac over Tailscale | Tested in Phase 0; Tailscale sidecar as fallback. |
| The Mac sleeps or leaves the tailnet mid-run | Retries with backoff in `OllamaResource`; a health check at job start that fails fast. |
| Docling is slow on CPU, especially with accurate table mode | OCR is not offered, `num_threads` configurable, cached model volume, per-document partitions, `document_timeout`. |
| A scanned PDF is dropped in and parses to almost no text | Characters-per-page metric makes it visible; no OCR fallback for now. |
| Semantic chunking makes the chunk stage depend on the Mac and adds one embedding per sentence | Batched calls, on-disk sentence embedding cache, cost recorded in chunk metrics. The other four strategies stay offline. |
| Sentence splitting is language-dependent, which affects `semantic` and `recursive` | The sentence splitter is a config value; check it against the language of the actual PDFs before trusting results. |
| Ollama unloads the model between batches | Set `keep_alive` on requests; record `load_duration` so cold starts are visible in metrics. |
| Changing embedding dimension in an existing collection | Dimension is part of the config hash; a mismatch against the collection raises before any insert. |
| The 8b embedding model does not fit the Mac's memory, or Ollama swaps models and distorts latency | Check memory in Phase 7.1; the matrix runs model by model so each is loaded once; `model_load_ms` is recorded per experiment. |
| Models are ranked by similarity scores, which differ in scale between models | The dashboard compares models on recall, MRR and nDCG only; the plan requires a filled-in query set before the comparison run. |
| Qdrant data corruption from a Windows bind mount | Qdrant storage uses a Docker named volume only. |
| Wiping the Postgres volume loses metrics history along with Dagster's run history | Both live in one volume; `docker compose down -v` is called out in `CLAUDE.md`. Add a `pg_dump` script if the history starts to matter. |
