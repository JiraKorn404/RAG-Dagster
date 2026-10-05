"""Upload a PDF, try a chunking strategy on it, look at where the cuts fall, and embed the result.

Everything runs in this process with the same plain functions the Dagster assets use: parse_pdf,
chunk_document, ingest_document. Uploads go to data/uploads, not data/raw, so the sensor
does not add them to the lab's documents."""

import hashlib
import json
import os
import time
from pathlib import Path

import data
import preview
import streamlit as st
import style
from docling_core.types.doc import DoclingDocument
from pydantic import ValidationError

from rag_lab.chunking import chunk_document, summarise
from rag_lab.chunking.segment import reference_text, segment
from rag_lab.config import (
    ChunkConfig,
    EmbedConfig,
    ExperimentConfig,
    HybridSettings,
    IndexConfig,
    RecursiveSettings,
    SemanticSettings,
)
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.ingest import (
    SCANNED_PDF_CHARS_PER_PAGE,
    check_new_experiment_name,
    clean_experiment_name,
    ensure_parsed,
    experiments_with_settings,
    ingest_document,
    save_upload,
)
from rag_lab.metrics.store import MetricsStore
from rag_lab.storage.qdrant import QdrantStore

PAGE_SIZE = 100
STRATEGIES = ["hybrid", "hierarchical", "fixed", "recursive", "semantic"]
DIMENSIONS = ["native", 256, 512, 768, 1024]


@st.cache_resource
def services() -> tuple[OllamaEmbedder, QdrantStore, MetricsStore]:
    return (
        OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]),
        QdrantStore(os.environ["QDRANT_URL"]),
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
    )


def register_upload(name: str, content: bytes) -> dict:
    """Save an upload and parse it once (cached on disk by document id)."""
    doc_id, path = save_upload(name, content)
    with st.status("Parsing with Docling (20 to 30 s on CPU, once per document; cached afterwards)…") as status:
        parse_dir, meta = ensure_parsed(path, doc_id)
        status.update(label="Parsed", state="complete")
    return {"doc_id": doc_id, "name": name, "dir": str(parse_dir), "meta": meta}


style.hero("Upload", "Upload a document, choose how it is chunked, embed it, then search it")

upload = st.file_uploader("PDF document", type=["pdf"])
if upload is not None:
    content = upload.getvalue()
    doc = st.session_state.get("up_doc")
    if doc is None or doc["doc_id"] != hashlib.sha256(content).hexdigest()[:16]:
        st.session_state.pop("up_chunks", None)
        st.session_state["up_doc"] = register_upload(upload.name, content)

doc = st.session_state.get("up_doc")
if doc is None:
    st.info("Upload a PDF to start. It is parsed once with Docling's default settings; changing the chunk settings never parses again.")
    st.stop()

meta = doc["meta"]
cols = st.columns(4)
for col, (label, value, sub) in zip(
    cols,
    [
        ("Document", meta["source_file"] if len(meta["source_file"]) <= 18 else doc["doc_id"], doc["doc_id"]),
        ("Pages", str(meta["pages"]), f"{meta['pages_per_second']:.2f} pages per second"),
        ("Tables", str(meta["tables"]), f"{meta['table_cells']} cells"),
        ("Characters per page", f"{meta['chars_per_page']:.0f}", f"parsed in {meta['parse_seconds']:.0f} s"),
    ],
):
    col.markdown(style.card(label, value, sub), unsafe_allow_html=True)
if meta["chars_per_page"] < SCANNED_PDF_CHARS_PER_PAGE:
    st.warning("Very little text per page: this looks like a scanned PDF. OCR is off, so there is little to chunk.")
st.write("")

# --- chunk settings ----------------------------------------------------------------------------
style.section("Chunking", "The same settings the chunk stage takes. Only the ones the chosen engine and strategy use are shown.")
models = data.embedding_models()
default_model = EmbedConfig().model
if not models:
    st.warning("Ollama did not list any qwen3-embedding model; using the default. Chunking `semantic` and embedding need Ollama.")
    models = [default_model]
c1, c2, c3 = st.columns([2, 1, 2])
model = c1.selectbox("Embedding model", models, index=models.index(default_model) if default_model in models else 0)
dimension_choice = c2.selectbox("Vector size", DIMENSIONS)
engine = c3.radio("Chunk engine", ["llamaindex", "native"], horizontal=True)

c1, c2, c3 = st.columns([2, 1, 2])
strategy = c1.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index(ChunkConfig().strategy))
max_tokens = c2.number_input("Max tokens", min_value=32, max_value=8192, value=ChunkConfig().max_tokens, step=32)
table_handling = c3.selectbox("Tables", ["markdown", "row-wise", "skip"], help="`row-wise` is not supported by hybrid and hierarchical (they fall back to markdown).")
include_headings = st.checkbox("Add the headings to each chunk's text", value=ChunkConfig().include_headings_in_text)
sparse = st.checkbox(
    "Add a BM25 keyword vector (needed for hybrid search)",
    value=IndexConfig().sparse,
    help="Keyword search over the same chunks, stored next to the embedding. Costs a little extra index time. It cannot be added to an experiment later.",
)

overlap, merge_peers = 0, HybridSettings().merge_peers
separators = RecursiveSettings().separators
semantic = SemanticSettings()
extra = {}
if strategy == "fixed":
    overlap = st.number_input("Overlap (tokens)", min_value=0, max_value=max(int(max_tokens) - 1, 0), value=64, step=8)
elif strategy == "hybrid":
    merge_peers = st.checkbox("Merge small neighbours under the same headings", value=merge_peers)
elif strategy == "recursive" and engine == "native":
    raw = st.text_input("Separators, tried in order (JSON list)", value=json.dumps(separators))
    try:
        separators = [str(s) for s in json.loads(raw)]
    except (ValueError, TypeError):
        st.error("The separators must be a JSON list of strings, for example [\"\\n\\n\", \"\\n\", \". \", \" \"].")
        st.stop()
elif strategy == "semantic":
    kinds = ["percentile", "stddev", "absolute"] if engine == "native" else ["percentile"]
    s1, s2, s3, s4 = st.columns(4)
    buffer_size = s1.number_input("Buffer (sentences each side)", min_value=0, max_value=10, value=semantic.buffer_size)
    kind = s2.selectbox("Breakpoint type", kinds)
    defaults = {"percentile": 90.0, "stddev": 1.0, "absolute": 0.3}
    threshold = s3.number_input("Breakpoint threshold", value=defaults[kind], key=f"threshold-{kind}", help="percentile 0 to 100; stddev in standard deviations; absolute a cosine distance")
    min_tokens = s4.number_input("Min tokens", min_value=0, value=semantic.min_tokens, disabled=engine != "native", help="Not applied by the llamaindex engine.")
    with st.expander("Sentence boundary"):
        pattern = st.text_input("Regular expression", value=semantic.sentence_pattern)
    semantic = SemanticSettings(
        buffer_size=int(buffer_size),
        breakpoint_type=kind,
        breakpoint_threshold=float(threshold),
        min_tokens=int(min_tokens),
        sentence_pattern=pattern,
    )


def build_config(name: str, tag: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        tag=tag,
        embed=EmbedConfig(model=model, dimension=None if dimension_choice == "native" else int(dimension_choice)),
        chunk=ChunkConfig(
            engine=engine,
            strategy=strategy,
            max_tokens=int(max_tokens),
            overlap=int(overlap),
            table_handling=table_handling,
            include_headings_in_text=include_headings,
            hybrid=HybridSettings(merge_peers=merge_peers),
            recursive=RecursiveSettings(separators=separators),
            semantic=semantic,
        ),
        index=IndexConfig(sparse=sparse),
    )


try:
    config = build_config("preview")
except (ValidationError, ValueError) as e:
    st.error(f"These settings are not valid: {e}")
    st.stop()

embedder, qdrant, metrics = services()
if st.button("Chunk", type="primary"):
    try:
        with st.spinner("Chunking…"):
            document = DoclingDocument.load_from_json(Path(doc["dir"]) / f"{doc['doc_id']}.json")
            started = time.perf_counter()
            chunks, ctx = chunk_document(document, doc["doc_id"], config, embedder)
            seconds = time.perf_counter() - started
            text = reference_text(segment(document))
    except Exception as e:  # noqa: BLE001  (a bad combination or Ollama not reachable: show it)
        st.error(f"Chunking failed: {e}")
        st.stop()
    st.session_state["up_chunks"] = {
        "hash": config.settings_hash(),
        "doc_id": doc["doc_id"],
        "chunks": chunks,
        "stats": ctx.stats,
        "warnings": ctx.warnings,
        "seconds": seconds,
        "text": text,
        "summary": summarise(chunks),
    }

result = st.session_state.get("up_chunks")
if result is None:
    st.caption("Choose the settings and press Chunk.")
    st.stop()
stale = result["hash"] != config.settings_hash() or result["doc_id"] != doc["doc_id"]
if stale:
    st.warning("The settings changed since this preview was made. Press Chunk again; Embed is disabled until then.")

chunks, text, summary = result["chunks"], result["text"], result["summary"]
if not chunks:
    st.warning("No chunks were produced (is everything skipped, or is the document scanned?).")
    st.stop()
for warning in result["warnings"]:
    st.warning(warning)

# --- preview -----------------------------------------------------------------------------------
st.write("")
cards = st.columns(4)
modalities = ", ".join(f"{n} {m}" for m, n in summary["by_modality"].items())
approx = sum(1 for c in chunks if c.span_approx)
nosrc = sum(1 for c in chunks if c.span and c.span[0] == c.span[1])
for col, (label, value, sub) in zip(
    cards,
    [
        ("Chunks", str(summary["chunks"]), modalities),
        ("Mean tokens", f"{summary['tokens_mean']:.0f}", f"min {summary['tokens_min']} · max {summary['tokens_max']}"),
        ("Chunking time", f"{result['seconds']:.2f} s", f"{summary['chunks'] / result['seconds']:.0f} chunks per second" if result["seconds"] else ""),
        ("Placed in the text", f"{summary['chunks'] - approx - nosrc} exact", f"{approx} approximate · {nosrc} with no source text"),
    ],
):
    col.markdown(style.card(label, value, sub), unsafe_allow_html=True)
st.write("")

pages = max(1, -(-len(chunks) // PAGE_SIZE))
page = 0
if pages > 1:
    page = st.selectbox(
        "Chunks shown",
        range(pages),
        format_func=lambda p: f"{p * PAGE_SIZE + 1} to {min(len(chunks), (p + 1) * PAGE_SIZE)} of {len(chunks)}",
    )
first, last = page * PAGE_SIZE, min(len(chunks), (page + 1) * PAGE_SIZE)

tab_doc, tab_embedded = st.tabs(["In the document", "As embedded"])
with tab_doc:
    st.markdown(preview.key_html(), unsafe_allow_html=True)
    lo = 0 if first == 0 else chunks[first].span[0]
    hi = len(text) if last >= len(chunks) else chunks[last].span[0]
    st.markdown(preview.document_html(text, chunks, lo, hi), unsafe_allow_html=True)
with tab_embedded:
    st.caption("What is embedded and stored: the chunk text, with the headings added in front when that is switched on.")
    st.markdown(preview.embedded_html(chunks[first:last], first), unsafe_allow_html=True)

# --- embed -------------------------------------------------------------------------------------
st.write("")
style.section(
    "Embed",
    "Writes the chunks, their vectors and Qdrant points, as ingest_job would: into a new experiment, "
    "or into an existing one with the same settings.",
)
NEW, EXISTING = "A new experiment", "An existing experiment with the same settings"
matches = [] if stale else experiments_with_settings(metrics, qdrant, config.settings_hash())
choice = NEW
if matches:
    choice = st.radio("Embed into", [NEW, EXISTING], horizontal=True)
elif not stale:
    st.caption("No experiment has these settings yet, so this makes a new one.")

target_name, target_tag, expected_hash = None, None, None
problem, confirmed = None, True
if choice == NEW:
    target_name = st.text_input(
        "Experiment name",
        value=clean_experiment_name(f"up-{Path(doc['name']).stem}-{strategy}"),
        key=f"name-{doc['doc_id']}-{strategy}",
        help="Also the Qdrant collection name: lower-case letters, digits, - and _. Any unused name works, "
        "even when another experiment has the same settings.",
    )
    target_tag = target_name  # the tag makes this experiment's hash its own
    if not stale:
        problem = check_new_experiment_name(metrics, qdrant, target_name)
        if problem:
            st.error(problem)
else:
    row = st.selectbox(
        "Experiment",
        matches,
        format_func=lambda r: (
            f"{r['name']} · {r['documents']} document(s) · "
            f"{'no collection yet' if r['points'] is None else str(r['points']) + ' points'}"
            + (" · in a benchmark report" if r["benchmarked"] else "")
        ),
    )
    target_name, target_tag, expected_hash = row["name"], row["config"].get("tag"), row["config_hash"]
    if qdrant.has_document(target_name, doc["doc_id"]):
        st.info("This document is already in the experiment; embedding it again replaces its points.")
    if row["benchmarked"]:
        st.warning(
            "This experiment belongs to a benchmark report. The report was measured without this document "
            "and is not redone, so it no longer describes everything in the experiment."
        )
        confirmed = st.checkbox("Add the document to it anyway")

if st.button("Embed and index", disabled=stale or problem is not None or not confirmed):
    final = build_config(target_name, target_tag)
    doc_id = doc["doc_id"]
    if expected_hash and final.config_hash() != expected_hash:
        st.error("The settings on the page do not give this experiment's hash, so nothing was written.")
        st.stop()
    progress = {"step": "starting"}

    def note(step: str) -> None:
        progress["step"] = step
        st.write(step)

    try:
        with st.status("Embedding…", expanded=True) as status:
            embedded, indexed = ingest_document(
                metrics, embedder, qdrant, final, doc_id, doc["name"], Path(doc["dir"]), meta, chunks,
                summary, result["stats"], result["warnings"], result["seconds"],
                register=choice == NEW,  # an existing experiment's row is left as it is
                on_step=note,
            )
            status.update(label="Embedded and indexed", state="complete")
    except Exception as e:  # noqa: BLE001  (show which step failed; earlier files stay, a rerun overwrites them)
        st.error(f"Failed while {progress['step']}: {e}")
        st.stop()
    st.success(
        f"{embedded['chunks']} chunks embedded ({embedded['tokens_per_second']} tokens per second) and "
        f"{indexed['points_written']} points written to the collection '{target_name}' "
        f"({indexed['collection_points']} points in it now)."
    )
    st.page_link("query.py", label="Try a query on it", icon=":material/search:")
