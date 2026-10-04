"""Benchmark one document: every chosen embedding model x chunking strategy, as a report that can be
reopened later and downloaded as a PDF.

The run is a background thread in this process. It saves each experiment's result to Postgres as it
finishes, so the page only reads from the database and a closed tab loses nothing."""

import os
import threading
from pathlib import Path

import charts
import data
import pandas as pd
import streamlit as st
import style
from docling_core.types.doc import DoclingDocument

from rag_lab.benchmark.pdf import report_pdf
from rag_lab.benchmark.report import build_report
from rag_lab.benchmark.runner import (
    CURRENT_STEP,
    BenchmarkSettings,
    Document,
    Services,
    TestQuery,
    delete_report_experiments,
    result_searches,
    run_report,
    snippets_missing_from,
    start_report,
)
from rag_lab.chunking.segment import reference_text, segment
from rag_lab.config import SEARCH_METHODS, EmbedConfig, SearchConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.ingest import ensure_parsed, save_upload
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.storage.qdrant import QdrantStore

STRATEGIES = data.STRATEGIES
QUERY_COLUMNS = ["Query", "Text that must be in the right chunk"]


@st.cache_resource
def services() -> Services:
    return Services(
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
        OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]),
        QdrantStore(os.environ["QDRANT_URL"]),
        OllamaReranker(os.environ["OLLAMA_BASE_URL"]),
    )


@st.cache_resource
def threads() -> dict[str, threading.Thread]:
    """Running report threads, kept here so a rerun of this script does not lose them."""
    return {}


@st.cache_data(show_spinner=False)
def document_text(parse_dir: str, doc_id: str) -> str:
    return reference_text(segment(DoclingDocument.load_from_json(Path(parse_dir) / f"{doc_id}.json")))


svc = services()
metrics = svc.metrics

style.hero("Benchmark", "Compare embedding models and chunking strategies on one document")

running = metrics.running_report()


def _first_p50(result_metrics: dict, per_query) -> float | None:
    """The median search time of the first search strategy an experiment has finished."""
    for entry in result_searches(result_metrics, per_query).values():
        if entry.get("search"):
            return entry["search"]["total_ms"]["p50"]
    return None


# --- progress of a running report -------------------------------------------------------------
def _progress(report_id: str) -> None:
    report = metrics.get_report(report_id)
    results = metrics.get_results(report_id)
    total = report["total_experiments"]
    st.progress(
        min(len(results) / total, 1.0),
        text=f"{len(results)} of {total} experiments · {CURRENT_STEP.get(report_id, 'working')}",
    )
    if results:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Experiment": r["experiment"],
                        "Status": r["status"],
                        "Chunks": (r["metrics"].get("chunking") or {}).get("chunks"),
                        "Search p50 (ms)": _first_p50(r["metrics"], r["per_query"]),
                    }
                    for r in results
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if report["status"] == "running":
        stop = st.button(
            "Stop after this experiment", key=f"stop-{report_id}", disabled=report["stop_requested"]
        )
        if stop:
            metrics.request_stop(report_id)
        if report["stop_requested"]:
            st.caption("Stopping after the current experiment…")
    else:
        st.rerun()  # the run ended: redraw the whole page with the finished report


if running:
    with st.container(border=True):
        style.section(
            f"Running: {running['document_name']}",
            f"Report {running['report_id']}. You can leave this page; the run goes on and the report is saved.",
        )
        st.fragment(run_every="3s")(_progress)(running["report_id"])

tab_new, tab_reports = st.tabs(["New benchmark", "Reports"])

# --- a new benchmark ---------------------------------------------------------------------------
def _new_benchmark() -> None:
    if running:
        st.info("A benchmark is running. Start another one when it has finished.")
    style.section("Document", "One PDF. It is parsed once with Docling's default settings.")
    uploaded = st.file_uploader("PDF document", type=["pdf"], key="bench-upload")
    if uploaded is not None:
        content = uploaded.getvalue()
        if st.session_state.get("bench_upload_key") != (uploaded.name, len(content)):
            doc_id, path = save_upload(uploaded.name, content)
            with st.status("Parsing with Docling (20 to 30 s on CPU, once per document)…") as status:
                parse_dir, meta = ensure_parsed(path, doc_id)
                status.update(label="Parsed", state="complete")
            st.session_state["bench_doc"] = {"doc_id": doc_id, "name": uploaded.name, "dir": str(parse_dir), "meta": meta}
            st.session_state["bench_upload_key"] = (uploaded.name, len(content))
    from_upload_page = st.session_state.get("up_doc")
    if from_upload_page and st.button(f"Use {from_upload_page['name']} from the Upload page"):
        st.session_state["bench_doc"] = from_upload_page
    doc = st.session_state.get("bench_doc")
    if doc is None:
        st.info("Upload a PDF to set up a benchmark.")
        return
    meta = doc["meta"]
    st.caption(
        f"{doc['name']} · {meta['pages']} pages · {meta['tables']} tables · parsed in {meta['parse_seconds']:.0f} s"
    )

    style.section("Settings", "What the report varies, and what it holds the same for every experiment.")
    available = data.embedding_models() or [EmbedConfig().model]
    c1, c2 = st.columns(2)
    models = c1.multiselect("Embedding models", available, default=available)
    strategies = c2.multiselect("Chunking strategies", STRATEGIES, default=STRATEGIES)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    engine = c1.radio("Chunk engine", ["llamaindex", "native"], horizontal=True)
    max_tokens = c2.number_input("Max tokens", min_value=32, max_value=8192, value=512, step=32)
    overlap = c3.number_input("Overlap (fixed only)", min_value=0, max_value=max(int(max_tokens) - 1, 0), value=64, step=8)
    table_handling = c4.selectbox("Tables", ["markdown", "row-wise", "skip"])
    c1, c2, c3, c4 = st.columns(4)
    headings = c1.checkbox("Headings in chunk text", value=True)
    top_k = c2.number_input("Top k", min_value=1, max_value=50, value=10)
    repeats = c3.number_input("Timed passes", min_value=1, max_value=10, value=3, help="Passes over the probe queries, after one warm-up pass.")
    probe_count = c4.number_input("Probe queries", min_value=1, max_value=100, value=20, help="Sentences taken evenly from the document, for search speed and score spread.")

    style.section(
        "Search strategies",
        "How each experiment's vectors are searched. All of them search the same collection, so a strategy "
        "adds searches, not experiments. Hybrid adds a BM25 vector to every point.",
    )
    labels = {
        "dense": "dense",
        "hybrid": "hybrid (dense + BM25)",
        "dense+rerank": "dense + rerank",
        "hybrid+rerank": "hybrid + rerank",
    }
    methods = st.multiselect(
        "Search strategies", SEARCH_METHODS, default=["dense"], format_func=labels.get, label_visibility="collapsed"
    )
    rerank_methods = [m for m in methods if m.endswith("+rerank")]
    rerankers = data.reranker_models()
    reranker, candidates, rerank_probes = SearchConfig().reranker, 20, 5
    if rerank_methods:
        if not rerankers:
            st.warning(
                "No reranker model is installed in Ollama. Pull one that can generate, for example "
                f"`ollama pull {SearchConfig().reranker}`, or drop the rerank strategies."
            )
        c1, c2, c3 = st.columns(3)
        reranker = c1.selectbox("Reranker", rerankers or [reranker])
        candidates = int(c2.number_input("Candidates", min_value=int(top_k), max_value=100, value=max(20, int(top_k)),
                                         help="Hits the first search hands to the reranker. It scores each one with a call to Ollama."))
        rerank_probes = int(c3.number_input("Rerank probe queries", min_value=1, max_value=int(probe_count), value=min(5, int(probe_count)),
                                            help="Rerank speed is measured on this many probe queries, in one pass."))

    style.section(
        "Test queries (optional)",
        "Add queries with the text that a right chunk must contain, and the report also measures retrieval quality.",
    )
    edited = st.data_editor(
        pd.DataFrame({QUERY_COLUMNS[0]: [""], QUERY_COLUMNS[1]: [""]}),
        num_rows="dynamic",
        width="stretch",
        key="bench-queries",
    )
    tests, incomplete = [], 0
    for _, row in edited.iterrows():
        q, snippet = str(row[QUERY_COLUMNS[0]] or "").strip(), str(row[QUERY_COLUMNS[1]] or "").strip()
        if q and snippet:
            tests.append(TestQuery(q, snippet))
        elif q or snippet:
            incomplete += 1
    if incomplete:
        st.warning(f"{incomplete} row(s) have only one of the two columns filled in and are ignored.")
    missing = snippets_missing_from(document_text(doc["dir"], doc["doc_id"]), [t.snippet for t in tests])
    for snippet in missing:
        st.warning(f"This text is not in the document, so its query would score zero: “{snippet[:80]}”")

    count = len(models) * len(strategies)
    st.caption(
        f"{len(models)} model(s) × {len(strategies)} strategy(ies) = {count} experiments, "
        f"{len(tests)} test queries."
        + (" Larger models and `semantic` take longer." if count else "")
    )
    if rerank_methods:
        calls = count * len(rerank_methods) * (1 + rerank_probes + len(tests)) * candidates
        st.caption(f"Reranking makes about {calls:,} calls to Ollama, one per chunk scored. Expect minutes per experiment.")
    blocked = not methods or (bool(rerank_methods) and not rerankers)
    if st.button("Start benchmark", type="primary", disabled=bool(running) or count == 0 or blocked):
        settings = BenchmarkSettings(
            models=models,
            strategies=strategies,
            engine=engine,
            max_tokens=int(max_tokens),
            overlap=int(overlap),
            table_handling=table_handling,
            include_headings=headings,
            top_k=int(top_k),
            repeats=int(repeats),
            probe_count=int(probe_count),
            search_methods=methods,
            reranker=reranker,
            candidates=candidates,
            rerank_probe_count=rerank_probes,
        )
        document = Document(doc["doc_id"], doc["name"], Path(doc["dir"]), meta)
        report_id, probes = start_report(document, settings, tests, metrics)
        thread = threading.Thread(
            target=run_report, args=(report_id, document, settings, tests, probes, svc), daemon=True
        )
        threads()[report_id] = thread
        thread.start()
        st.rerun()


with tab_new:
    _new_benchmark()


# --- saved reports -----------------------------------------------------------------------------
def _frame(chart) -> pd.DataFrame:
    return pd.DataFrame(chart.values).rename(columns={"experiment": "name"})


@st.cache_data(show_spinner=False)
def _pdf(report_id: str, results_count: int, status: str) -> bytes:
    stored = metrics.get_report(report_id)
    return report_pdf(build_report(stored, metrics.get_results(report_id)))


with tab_reports:
    reports = metrics.list_reports()
    if not reports:
        st.info("No reports yet. Start a benchmark on the first tab.")
        st.stop()
    chosen = st.selectbox(
        "Report",
        reports,
        format_func=lambda r: (
            f"{r['created_at']:%Y-%m-%d %H:%M} · {r['document_name']} · {r['status']} · "
            f"{r['total_experiments']} experiments"
        ),
    )
    stored = metrics.get_report(chosen["report_id"])
    results = metrics.get_results(chosen["report_id"])
    if stored["status"] == "running":
        st.info("This report is still running; see the progress at the top of the page.")
        st.stop()
    report = build_report(stored, results)

    cards = st.columns(4)
    for col, (label, value, sub) in zip(
        cards,
        [
            ("Document", report.document["name"] if len(report.document["name"]) <= 18 else report.document["id"], f"{report.document['pages']} pages"),
            ("Experiments", f"{report.experiments} / {stored['total_experiments']}", f"{len(report.failed)} failed"),
            ("Status", stored["status"], report.created),
            ("Test queries", str(len(stored["test_queries"] or [])), "quality measured" if report.has_quality else "speed and behaviour only"),
        ],
    ):
        col.markdown(style.card(label, value, sub), unsafe_allow_html=True)
    if stored["status"] in ("stopped", "interrupted", "failed"):
        st.warning(f"This report is {stored['status']}: it covers the experiments that finished." + (f" {stored['error']}" if stored["error"] else ""))
    st.write("")

    if report.rows:
        st.download_button(
            "Download PDF",
            _pdf(stored["report_id"], len(results), stored["status"]),
            file_name=f"benchmark-{Path(stored['document_name']).stem}-{stored['created_at']:%Y%m%d}.pdf",
            mime="application/pdf",
            type="primary",
        )
        style.section("Summary", "Green: the best value in a speed or quality column.")
        frame = pd.DataFrame(report.rows, columns=report.columns)
        best = set(report.best)

        def mark(_):
            return pd.DataFrame(
                [["background-color: rgba(25,158,112,.35)" if (r, c) in best else "" for c in range(len(report.columns))] for r in range(len(report.rows))],
                columns=report.columns,
            )

        st.dataframe(frame.style.apply(mark, axis=None), hide_index=True, width="stretch")

        method = report.methods[0]
        if len(report.methods) > 1:
            method = st.segmented_control(
                "Search strategy shown in the speed and quality charts and the test query ranks",
                report.methods,
                default=report.methods[0],
                key=f"method-{stored['report_id']}",
            ) or report.methods[0]
        shown = [c for c in report.charts if c.values and c.group in (None, "compare", method)]
        left, right = st.columns(2)
        for i, chart in enumerate(shown):
            with (left if i % 2 == 0 else right), st.container(border=True):
                suffix = f" ({method} search)" if chart.group == method and len(report.methods) > 1 else ""
                style.section(chart.title + suffix, chart.unit)
                charts.grouped_bars(
                    _frame(chart), chart.unit, fmt=chart.fmt, height=260,
                    order=report.methods if chart.group == "compare" else None,
                )

        if report.queries and method in report.queries["by_method"]:
            style.section("Test queries", "Rank of the first chunk that contains the snippet (1 is best; a dash means not found).")
            q, found = report.queries, report.queries["by_method"][method]
            table = pd.DataFrame(
                [["–" if r is None else str(r) for r in ranks] for ranks in found["ranks"]],
                columns=[e["label"] for e in found["experiments"]],
            )
            table.insert(0, "Query", [item["query"] for item in q["queries"]])
            st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.warning("No experiment finished in this report.")

    for name, error in report.failed:
        st.error(f"{name} failed: {error}")
    with st.expander("Notes"):
        for note in report.notes:
            st.markdown(f"- {note}")

    existing = [r["experiment"] for r in results if svc.qdrant.client.collection_exists(r["experiment"])]
    if existing:
        st.caption(f"{len(existing)} of this report's experiments are still in Qdrant and can be searched on the Try a query page.")
        if st.button(f"Delete this run's {len(existing)} experiments"):
            removed = delete_report_experiments(stored["report_id"], metrics, svc.qdrant)
            st.success(f"Deleted {len(removed)} experiments. The report stays.")
            st.rerun()
    elif results:
        st.caption("This report's experiments have been deleted; the report remains.")
