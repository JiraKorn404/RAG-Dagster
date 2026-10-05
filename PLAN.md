# Plan

Phase 14 and the RAG chatbot (Phases 15 to 18) are planned (below). Phases 0 to 13 are done: the project is a RAG platform with an Upload page (chunk preview, embed into a new or existing experiment), a Try a query page, an Experiments page (list and delete what is in each experiment) and a Benchmark page that compares embedding models, chunking strategies and search strategies (dense, hybrid, reranked) and produces a PDF report. What each phase planned and found is in `COMPLETED_PLAN.md`; the current state of the repo is in `CLAUDE.md`.

A new phase is written here as `## Phase 14 — ...`: a goal, a table of decisions, numbered steps with checkboxes, and a "Done when" line that is checked once on the real stack. When it is done, its section moves to `COMPLETED_PLAN.md`.

## Phase 14 — BM25 option on the Upload page

**Goal:** an experiment made on the Upload page can be searched with `hybrid` and `hybrid+rerank` on Try a query. Today the page never sets `index.sparse`, so every uploaded experiment is dense-only, and a collection made without the BM25 vector cannot get it later.

| Decision | Choice |
|---|---|
| Control | One checkbox on the Upload page, "Add a BM25 keyword vector (needed for hybrid search)", off by default, so uploads behave as before. |
| Where | Next to the other settings in `ui/upload.py`; `build_config` passes it as `index=IndexConfig(sparse=sparse)`. |
| Backend | No change. `IndexConfig.sparse`, `ensure_collection`, `index_chunks` and Try a query's `has_sparse` check already handle it. |
| Config identity | `sparse` is in `settings_hash()` only while true, so a sparse experiment never matches a dense-only one. The "existing experiment with the same settings" list then offers only sparse experiments, and a new experiment is the way to add BM25 to a document already uploaded. |
| Preview | The hash covers `index`, so ticking the box marks the chunk preview stale and the page asks for Chunk again. Accepted; no change to that logic. |
| Existing experiments | Left as they are. They stay dense-only; delete them on the Experiments page if they are no longer wanted. |
| Tests | None. The change is UI wiring; hashing of `sparse` is already covered by its existing behaviour. |

### Steps

- [x] 1. `ui/upload.py`: import `IndexConfig`; add the checkbox with a `help` text (keyword search over the same chunks; costs a little extra index time; cannot be added to an experiment later); pass `index=IndexConfig(sparse=sparse)` in `build_config`.
- [x] 2. `CLAUDE.md`: in the Upload description and the "BM25 sparse vectors" fact, note that the Upload page has the option.
- [x] 3. Restart the UI: `docker compose restart ui` (`ui/upload.py` is a page script, so Streamlit may pick it up without it).

**Done when:** a PDF is uploaded with the box ticked into a new experiment, and on Try a query that experiment offers `hybrid` and `hybrid+rerank`, and a `hybrid` search returns hits. Checked once, on the real stack.

## RAG chatbot (Phases 15 to 18)

**Goal:** a Chatbot page where you ask questions about the documents in an experiment and get an answer written by `gemma4:e4b-mlx` (served by Ollama on the Mac) from retrieved chunks, with everything the system does shown: which step it is in, what the model is thinking, which chunks it found and how they scored, and how long each step took. The agent is built with LangGraph (the graph) and `langchain-ollama` (the chat model). Main retrieval is `hybrid+rerank`.

Four parts, each shippable on its own: 15 the agent without a UI, 16 the event stream and trace that make it transparent, 17 the page, 18 smarter behaviour (retry, follow-ups, abstaining). Phase 14 (the BM25 checkbox) is a prerequisite.

### What was checked while drafting

- `gemma4:e4b-mlx` on the Mac's Ollama (0.35.0) reports capabilities `completion, vision, audio, tools, thinking`, 8.1B parameters, `nvfp4`, 131072 context. So tool calling and a separate thinking stream are available; vision and audio are not used (Scope).
- Ollama's default `num_ctx` is small and silently cuts a long prompt, so the agent always sets `num_ctx` itself and the page shows prompt tokens against it.
- **The installed reranker is `dengcao/Qwen3-Reranker-0.6B:Q8_0`, which scores every chunk 0.0** (see the Try a query finding). `hybrid+rerank` as the main strategy needs `dengcao/Qwen3-Reranker-4B:Q4_K_M` pulled first. This blocks Phase 15's "Done when".
- `hybrid` needs an experiment made with `index.sparse` (Phase 14). The chatbot offers only such experiments.
- One Mac will hold the chat model, the embedding model and the 4B reranker. Ollama may unload and reload them between steps; the first answer after a pause can be slow. Measure it in Phase 15 and set `keep_alive` / `OLLAMA_MAX_LOADED_MODELS` on the Mac only if it hurts.

### Architecture

A fixed graph, not a free-roaming tool-calling agent. The 8B model decides only what a small model does reliably (is this a follow-up, is the evidence enough, what to rewrite the query as, how to word the answer); retrieval itself is deterministic code, always `hybrid+rerank`. A tool-calling variant is under Ideas.

```
                 +----------------------------- retry, at most max_rewrites times (Phase 18) ---+
                 |                                                                              |
question -> condense -> retrieve -> grade --enough--> generate -> answer + citations            |
 (+ history)  (standalone  (hybrid + rerank,   |                                                |
               query)       top k chunks)      +--not enough--> rewrite --------------------------+
                                               |                  (still not enough after the retries)
                                               +------------------> abstain: "not in the documents", chunks still shown
```

| Node | Does | Model call | Thinking |
|---|---|---|---|
| `condense` | Turns the question and the last few turns into one standalone query ("and the second one?" becomes a full question). Skipped on the first turn. | yes, short | off |
| `retrieve` | `rag_lab.search.search(..., options=SearchConfig(method="hybrid+rerank"), reranker=...)`. Not a LangChain retriever: our own search keeps the vectors, filters and scores under our rules. | none | |
| `grade` | Phase 15: passes through. Phase 18: the reranker's best score against a threshold first (free), then a yes/no from the model when the score is in between. | Phase 18 | off |
| `rewrite` | Phase 18: a different query when the evidence is not enough. | yes, short | off |
| `generate` | Writes the answer from the numbered chunks only, citing `[1]`, `[2]`; says so when the chunks do not contain the answer. Streams tokens. | yes | on (setting) |
| `abstain` | Phase 18: a fixed "I could not find this in the documents" plus what was searched. | none | |

State (`TypedDict`): `question`, `history`, `query`, `hits`, `rewrites`, `grade`, `answer`, `events`. Settings go in `config.py` as `AgentConfig` (`model`, `temperature`, `num_ctx`, `think`, `top_k`, `candidates`, `history_turns`, `max_rewrites`, `grade_threshold`), not part of `ExperimentConfig` or its hash, like `SearchConfig`.

Code lives in `src/rag_lab/agent/` as plain Python with no Streamlit or Dagster imports: `graph.py` (state, nodes, `build_graph`), `prompts.py`, `events.py`, `__main__.py` (CLI). `ui/chat.py` is a thin caller. Retrieval reuses `search()`, `OllamaEmbedder`, `QdrantStore`, `OllamaReranker` unchanged. New dependencies: `langgraph` and `langchain-ollama` (which brings `langchain-core`); needed because the graph, the streaming of node updates and the Ollama chat model with a thinking stream are what they provide. Needs `docker compose up -d --build`.

### Phase 15 — Agent core, no UI

**Goal:** `condense -> retrieve -> generate` runs end to end from a command line, answering with citations from `hybrid+rerank` hits.

| Decision | Choice |
|---|---|
| Graph | The linear graph above. There is no `grade` node yet; Phase 18 adds it with its edges, so no empty node is carried. `condense` is one node that returns the question unchanged when there is no history. |
| Chat model | `ChatOllama(model, base_url=OLLAMA_BASE_URL, num_ctx, temperature, reasoning=think)`; the exact thinking parameter and where the thinking text arrives are checked against the installed `langchain-ollama` before writing `generate`. |
| Structured output | Not needed in this phase (`condense` returns plain text). When Phase 18 needs it (`grade`), thinking is off for that call, because JSON-schema output and thinking can clash. |
| Prompt | Chunks are numbered with source, page and headings; the model answers only from them and cites by number. Chunk text is data, not instructions. |
| Memory | The page and the CLI keep the message list and pass `history` in; no checkpointer yet. |
| Tests | None. |

Steps:

- [x] 1. A working reranker on the Mac and an experiment with a BM25 vector. Done with `dengcao/Qwen3-Reranker-4B:Q8_0` (already pulled; P(yes) 0.999 relevant, 0.0 irrelevant, so the `Q4_K_M` pull was not needed) and the existing `full-ingest-dense-sparse`.
- [x] 2. Add `langgraph` and `langchain-ollama` to `pyproject.toml`; rebuild.
- [x] 3. `AgentConfig` in `config.py`.
- [x] 4. `agent/prompts.py`, `agent/graph.py` (state, `condense`, `retrieve`, `generate`, `build_graph(embedder, store, metrics, reranker, llm)`).
- [x] 5. `agent/__main__.py`: `python -m rag_lab.agent "question" --experiment <name>` prints the answer and the numbered sources.
- [x] 6. `CLAUDE.md`: the `agent/` package, `AgentConfig`, the dependencies.

Built differently from the draft: `AgentConfig` nests a `SearchConfig` (method `hybrid+rerank`, reranker `Q8_0` 4B) instead of repeating `candidates` and the reranker; `grade_threshold` and `max_rewrites` are left to Phase 18; `SearchConfig`'s own default reranker is untouched.

**Done when:** on the real stack, a question whose answer is in the document gets a correct answer citing the right chunk, and the hits show nonzero reranker scores. Checked once.

**Result (checked once):** `python -m rag_lab.agent "what to do when there is a door opening mid flight" --experiment full-ingest-dense-sparse` answered correctly from the chunk "Door Opening In-Flight" (page 17), cited `[1]`, and the reranker scored it 0.999 (the next hits 0.18 to 0.08). 1682 tokens in, 645 out. Search took 12.4 s, 11.1 s of it reranking 20 candidates (the first call also loads the 4B model, so a warm run is expected to be faster). The follow-up path (`condense` with history) is wired but was not exercised; Phase 18 checks it.

### Phase 16 — Transparency: events and trace

**Goal:** every step the graph takes is a typed event, streamed while it happens and saved afterwards, so a page (or a CLI) can show it without knowing the graph.

| Decision | Choice |
|---|---|
| Events | Plain dataclasses in `agent/events.py`: `StepStarted(node)`, `StepFinished(node, ms)`, `Thinking(text)` (token chunks), `Query(text, kind)` (original or rewritten), `Retrieved(hits, method, candidates, embed_ms, search_ms, rerank_ms)`, `AnswerToken(text)`, `ModelState(...)`, `Done(answer, citations, timings)`. |
| How they are produced | Every node reports through LangGraph's custom stream (`get_stream_writer()`), including the thinking and answer tokens, which `generate` reads with `llm.stream`. That replaces the `updates` and `messages` modes the first draft listed: one channel, and the node already knows what it is doing. `run(graph, experiment, question, history, ...)` streams `custom` and `values` and yields the events, then `Done`. |
| Model state | Per model call: which model, thinking on or off, prompt tokens against `num_ctx` (a warning event when the prompt reached it), output tokens, tokens per second (from Ollama's eval counts), and whether the model was already loaded (`/api/ps` before the call). |
| Citations | The answer's `[n]` markers are resolved to hits; a number with no hit is flagged. |
| Persistence | `run()` does it, given `metrics` and a `session_id`, so the CLI and the page share it. A numbered migration adds `chat_turns` to `rag_metrics` (session id, experiment, question, standalone query, model, hits as JSON, answer, timings, created time). The retrieval also goes to the existing search log. Postgres stays the source of truth. A session id is a random id per browser session. |
| Tests | None. |

Steps:

- [x] 1. `agent/events.py`; `agent/graph.py` emits them and exposes `run(...)`.
- [x] 2. `ModelState` collection around each model call (including the `/api/ps` check).
- [x] 3. Migration `rag_lab/metrics/migrations/NNN_chat_turns.sql` and a `MetricsStore` method to write a turn.
- [x] 4. `agent/__main__.py` prints the events live (steps, thinking dimmed, retrieved chunks, timings).
- [x] 5. `CLAUDE.md`: events, `chat_turns`.

**Done when:** one CLI run prints every step with its time, the thinking text, the retrieved chunks with scores, the model state line, and writes one `chat_turns` row. Checked once.

Built differently from the draft: `Query(text, rewritten)` carries a flag instead of a `kind`; the `usage` entry is gone from the graph state (the `ModelState` event carries the counts); the CLI applies the migrations itself on start, as the UI does.

**Result (checked once):** `python -m rag_lab.agent` printed each step with its time, the standalone query, the five chunks with reranker scores (rerank 2.2 s warm), the dimmed thinking, the streamed answer, the model state (`1593 of 8192 context in, 236 out, 39.3 tok/s`, model was loaded) and the citations, and wrote one `chat_turns` row and one `search_log` row per turn (migration `0007` applied by the CLI). A two-question chat rewrote "and what causes that kind of accident?" to "what causes an open door accident in flight?" and answered it from the same chunk. A bug found on the way: the token counts are on the chunk that ends the generation, not the last chunk, so the first run showed 0 in and 0 out until that was fixed. Checked the thinking path and the saved thinking text on the first run (before that fix); the later runs had thinking off.

### Phase 17 — Chatbot page

**Goal:** a Chatbot page in the Streamlit UI that shows the answer and, with it, how it was produced.

| Decision | Choice |
|---|---|
| Page | `ui/chat.py`, added to `ui/app.py` next to Try a query. A thin caller of `agent.run`. |
| Settings | In the sidebar: experiment (only ones with a BM25 vector), embedding model is the experiment's, chat model (installed models that report `completion` and `tools`, default `gemma4:e4b-mlx`), reranker (as on Try a query), top k, candidates, thinking on/off. `hybrid+rerank` is fixed and said so. |
| Conversation | `st.chat_input` and `st.chat_message`; history in `st.session_state`; a "New chat" button. |
| While answering | An `st.status` panel shows the current state ("Rewriting the question", "Searching: hybrid, 20 candidates", "Reranking", "Writing the answer") and fills in live: the standalone query, then the thinking text, then the answer streaming below. |
| After answering | Under each answer an expander "How this was answered" with: the steps and times as a timeline, the thinking text, the model state line, and the retrieved chunks as the same cards Try a query uses (rank, reranker score, source, page, headings, modality, text), with the cited ones marked. Clicking a citation `[n]` is not needed; the numbers match the cards. |
| Reuse | `hit_card` and the score colours from `ui/query.py` move to a shared module (`ui/style.py` or `ui/hits.py`) rather than being copied. |
| Failure | Ollama unreachable, model missing or reranker returning nothing shows as an error in the panel with the step it failed in, never a blank answer. |
| Tests | None; the app tester can drive the page but not streaming meaningfully. |

Steps:

- [x] 1. Move the hit card out of `ui/query.py` into a shared module; Try a query imports it.
- [x] 2. `ui/chat.py` with the sidebar, the chat loop and the live status panel.
- [x] 3. The "How this was answered" expander.
- [x] 4. Add the page to `ui/app.py`; `docker compose restart ui`.
- [x] 5. `CLAUDE.md`: the fifth page, the layout list.

**Done when:** in the browser, a question gets a streamed answer, the status panel showed each step while it ran, and the expander lists the thinking, the timeline and the chunks with nonzero scores. Checked once.

Built differently from the draft: the tabs inside the expander are "Steps and model", "Thinking" and "Retrieved chunks" (Streamlit has no nested expanders, so the thinking is a tab, not a second expander); `data.chat_models()` was added for the model list; the page starts a new conversation when the experiment changes; a failed turn is kept in the display with its error but not in the history the agent sees.

**Result:** driven with Streamlit's app tester inside the `ui` container against the real stack: the page loads without an exception, lists `full-ingest-dense-sparse`, `gemma4:e4b-mlx` (thinking on) and the 4B reranker, answers a question with an expander holding the three tabs and five chunks, and answers a follow-up (the history carries both turns). **Not checked: how the live status panel looks in a browser.** The Chrome tab froze on two screenshot attempts (the server was healthy: health check 200, no errors in the log), so the layout and the live streaming have not been seen by eye; look at the page once before moving Phase 17 to `COMPLETED_PLAN.md`.

### Phase 18 — Better behaviour: retry, follow-ups, abstaining

**Goal:** the agent notices when retrieval did not find the answer, tries once more, and otherwise says so instead of inventing one.

| Decision | Choice |
|---|---|
| `grade` | The best reranker score against two scores, `enough_score` and `missing_score` (one threshold cannot have a band): at or above the first is enough, far below is not, the band between asks the model for a yes/no on whether the chunks answer the question. The threshold is chosen from real scores on a few answerable and unanswerable questions, not guessed. |
| `rewrite` | One retry by default (`max_rewrites = 1`) with a different query; the event stream shows both queries and both hit lists. |
| `abstain` | After the retries, a fixed message plus what was searched. The chunks are still shown. |
| Follow-ups | `condense` already handles them; this phase checks it on a real multi-turn chat and trims `history` to `history_turns`. |
| Grounding check | Not here (Ideas). |
| Tests | None. |

Steps:

- [x] 1. `grade`, `rewrite`, `abstain` nodes and the conditional edges.
- [x] 2. Pick `grade_threshold` from a handful of real queries; record the numbers here.
- [x] 3. The page shows retries and the abstain case in the timeline.
- [x] 4. `CLAUDE.md`.

**Done when:** a question that is not in the document ends in the abstain message with the chunks it looked at, a question needing a reworded query is answered after one retry, and a follow-up question ("and the second one?") is answered correctly. Checked once.

Built differently from the draft:

- Two scores instead of `grade_threshold`: `enough_score` 0.5 and `missing_score` 0.1, with the model asked only in between. `max_rewrites` is 1.
- `rewrite` is given the section titles of the closest chunks. Its first version, without them, turned "the stick does nothing when I pull it" into "joystick unresponsive when pulled" (score 0.047); with the titles it wrote "Flight Control Malfunction" (0.932).
- The answer is written for the standalone question, not for the rewrite; the retry's hits replace the first search's (the latest search is what the answer and the notice use).
- New events `Graded` and `Rewrote`; `Done.abstained`; step times are summed over a retry. The page lists every search with its score and verdict, shows the abstain message as a notice, and keeps the abstain message in the history like any answer.

**Thresholds, from the real document (FAA handbook chapter 18, `full-ingest-dense-sparse`), best chunk's reranker score:**

| Kind | Questions | Scores |
|---|---|---|
| Answered by the document | door opening, engine failure after takeoff, in-flight fire, landing gear, flaps, elevator, ditching and others (about 20) | 0.74 to 1.00, except three below: "the stick does nothing when I pull it" 0.357, "what if a wheel will not come down" 0.107, "the undercarriage is stuck up and will not lower" 0.258 |
| Not in the document | capital of France, bread, football, tyre, Boeing 747 weight, pilot licence cost, VNE of a Cessna 172, how pilots are paid, carburetor icing (never mentioned), spiral dive (mentioned once, not explained) | 0.000 to 0.051 |
| Not covered but near the topic | "what causes icing" (mentioned three times, causes not explained) | 0.006 |

So 0.5 and 0.1 leave a band of 0.1 to 0.5 that held three answerable questions, and the model judged all three answerable. It also judged "what to do if the aircraft shakes violently and the nose drops" (a stall; best score 0.385, about a ditching) answerable and the answer was off-topic. The grading model without thinking is lenient; raising `missing_score` would trade that against losing the three real ones. Not tuned further on one document.

**Result (checked on the real stack):**

- Not in the document: "what is the capital of France" and "what causes icing" each searched twice (the second query rewritten), both scored under 0.01, and ended in the abstain message listing both queries, with the chunks kept under "How this was answered". About 23 to 34 s, almost all of it two rerank passes.
- Needs a rewrite: retrieval was too robust on this document to give a natural weak first search (every rewording scored 0.74 or more). It was checked by raising both scores to 0.9 for one run: "the stick does nothing when I pull it" (0.357) was rewritten to "Flight Control Malfunction" (0.932), graded enough and answered correctly from the elevator-cable passage, citing `[2]`.
- Follow-up through the new graph: "and what causes that kind of accident?" after the door question became "what causes an open door accident in flight" (0.997) and was answered correctly.
- The Streamlit page, driven with the app tester: the off-topic question shows the abstain notice and both searches with scores; a normal question still answers.
- Not seen by eye: the page in a browser (the Chrome tab froze in Phase 17, and was not retried).
- Slow on the Mac: a rerank pass took 14 to 18 s on the off-topic questions against 2 to 4 s in other runs, and one grade call took 10 s for one output token. It looks like Ollama swapping the 4B reranker and gemma in and out of memory; not investigated.

### Open choices (defaults used above unless you say otherwise)

- Fixed graph rather than a tool-calling agent (above), because of the small model.
- Chunks in the answer prompt are the top k after reranking (default 5); the answer cites by number.
- Chat history is kept in the browser session only; no checkpointer, no chat history page.
- The page works on one experiment at a time.

## Working approach

Keep the project minimal. Testing is deliberately light:

- **Tests:** only for pure logic where a silent bug would corrupt results (retrieval metric formulas, vector truncation and re-normalisation, config hashing). Nothing else gets a test file.
- **Verification:** each phase's "Done when" is checked once, by a real run against the real stack. No mocks, no fake embedders, no integration test suite.
- **After small edits:** do not rerun tests, rebuild containers or re-materialise assets. Rerun only when the logic of a tested area changed.
- **Ideas and deferred items** are built only when you ask for them.

## Scope

- **Documents:** PDF with a text layer.
- **Content:** text and tables. Image and figure ingestion and OCR are deferred (see "Deferred").
- **Hardware:** no NVIDIA GPU on this machine. Docling runs on CPU; Ollama runs where `OLLAMA_BASE_URL` points.

## Ideas

Not planned and not in any phase. Each is built only when you ask for it.

- Chatbot: a tool-calling variant where the model chooses between searching, searching one document (`source_file` filter) and answering directly; a groundedness check that verifies each cited claim against its chunk; a Postgres checkpointer and a page of past chats; restricting a chat to chosen documents.
- A "Process with Dagster" option on the Upload page: copy the PDF to `data/raw/`, register its partition, launch `ingest_job` with the page's settings (and `tag`) as run config, and link to the run. The run would then show in the Dagster UI, at the cost of the page's chunk preview.
- Benchmark: several documents in one report; test queries generated by a local LLM from the document's own text; a Unicode font in the PDF so names with other characters are not replaced with `?`.
- Try a query: restrict the search to one document of an experiment (`source_file` already has a payload index).
- Experiments page: list the uploaded files and the parse cache (`data/uploads/`, `data/artifacts/_uploads/`) with the experiments that use each, and delete the ones nothing uses.
- On the upload page: parse options (`table_mode`, formulas), and picking a document already in `data/raw`.
- Parsing for the upload page as a Dagster run (a partition and `parsed_document`) instead of inside the Streamlit process, so it shows in the Dagster UI and does not use the UI container's memory.
- One `IngestionPipeline` with a docstore, to skip unchanged documents and dedupe chunks (see Phase 8 in `COMPLETED_PLAN.md`; it conflicts with per-stage files, so it would be a separate fast path).
- Embedding models outside the Qwen3 family.
- Quantisation settings in Qdrant.
- `--exact` search to measure what HNSW gives up. See Phase 5 in `COMPLETED_PLAN.md`.
- `docling-serve` as a separate container, compared against in-process parsing.

## Deferred

- **Image and figure ingestion.** When wanted: enable picture extraction in `ParseConfig`, add a `picture` modality, and describe images with a vision model so they can be embedded as text.
- **OCR** (scanned PDFs). Docling's OCR also runs on CPU, only slowly, so this is a speed trade-off rather than a hard limit. When wanted: expose `do_ocr`, `ocr_engine`, `ocr_languages` and `force_full_page_ocr` in `ParseConfig`.
- **Other document types** (DOCX, PPTX, HTML). Docling handles them; the sensor and `ParseConfig` would need format-specific options.
