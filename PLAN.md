# Plan

Build a modular, experiment-friendly ingestion pipeline: PDFs in, searchable text and table vectors out, with similarity scores and a full set of performance metrics stored in PostgreSQL.

Each phase ends with something that runs. Do them in order; Phase 0 exists to find network problems before any pipeline code is written.

## Working approach

Keep the project minimal. Testing is deliberately light:

- **Tests:** only the two small files listed in Phases 1 and 6 (config hashing and vector truncation, retrieval metric formulas). Nothing else gets a test file.
- **Verification:** each phase's "Done when" is checked once, by a real run against the real stack. No mocks, no fake embedders, no integration test suite.
- **After small edits:** do not rerun tests, rebuild containers or re-materialise assets. Rerun only when the logic of a tested area changed.
- **Optional items** (in Phases 5 and 7 and under "Deferred") are built only when you ask for them.

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

- [ ] `search/`: `search(query, experiment, top_k, filters)` embeds the query with the instruction prefix and runs `query_points`, with `hnsw_ef` taken from config. Each hit returns text, source file, page, modality, `similarity` (Qdrant's cosine score) and `distance` (`1 - similarity`).
- [ ] Per-query timing breakdown: embed ms, Qdrant ms, total ms.
- [ ] CLI: `python -m rag_lab.search "..." --experiment <name> --top-k 5 [--modality table] [--json] [--no-log]`. Writes to `search_log` unless `--no-log`.
- [ ] Optional: `--exact` flag (brute-force search) to measure how much recall the HNSW index gives up.
- [ ] Optional: hybrid mode for comparison. In Qdrant this needs a sparse (BM25) vector stored on each point at index time, fused with the dense result at query time, so it touches Phase 4 as well.

## Phase 6 — Metrics and benchmarking

All metrics below are implemented in this phase. Quality metrics are computed only for queries that have expected results; otherwise they are stored as null and the run still completes.

- [ ] `eval/queries.yaml` format: `id`, `query`, optional `expected` list of `{source_file, page}` or `{chunk_id}`, optional `modality`. Ship it with placeholder queries and empty `expected`. Loading it syncs `eval_queries`.
- [ ] `metrics/retrieval.py`, with one small test of the formulas against a hand-computed case (the one place a silent error would look plausible).
- [ ] `search_benchmark` asset per experiment: warm-up pass, then N repeats of the query set. Writes `benchmark_runs`, `query_results`, `benchmark_metrics`.
- [ ] `experiment_summary` asset: a SQL view over `rag_metrics` giving one row per experiment, shown as a Markdown table in the Dagster UI.

| Group | Metrics |
|---|---|
| Search latency | p50, p95, p99, mean, min, max for embed, Qdrant search and total; queries per second; cold versus warm first query |
| Similarity scores | top-1 similarity, mean similarity over top-k, gap between rank 1 and rank k, score distribution per experiment |
| Retrieval quality (needs expected results) | Hit rate@k, Recall@k, Precision@k, MRR, MAP, nDCG@k for k in 1, 3, 5, 10; also split by modality (text versus table) |
| Ingestion performance | parse seconds and pages per second; chunks per document; embed chunks per second and tokens per second; model load time; index points per second; end-to-end seconds per document |
| Index state | point count, indexed vector count, vector dimension, segment count |

Done when: two experiments (for example table mode fast versus accurate, or two chunking strategies) can be compared with one SQL query, and filling in `expected` then re-running the benchmark produces quality metrics with no code change.

## Phase 7 — Lab ergonomics (as wanted)

- [ ] Preset experiment configs in `experiments/*.yaml`, selectable in the launchpad.
- [ ] A job that runs a list of experiments over the same documents.
- [ ] Dashboard over `rag_metrics` (Streamlit, or Grafana/Metabase pointed at Postgres).
- [ ] Small UI for trying queries across experiments side by side.
- [ ] `docling-serve` as a separate container, compared against in-process parsing.
- [ ] Reranking stage; alternative embedding models; quantisation settings in Qdrant.

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
| Qdrant data corruption from a Windows bind mount | Qdrant storage uses a Docker named volume only. |
| Wiping the Postgres volume loses metrics history along with Dagster's run history | Both live in one volume; `docker compose down -v` is called out in `CLAUDE.md`. Add a `pg_dump` script if the history starts to matter. |
