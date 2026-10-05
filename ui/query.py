import os

import data
import streamlit as st
import style
from hits import hit_card

from rag_lab.config import SEARCH_METHODS, ExperimentConfig, SearchConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import search
from rag_lab.storage.qdrant import QdrantStore


@st.cache_resource
def clients() -> tuple[OllamaEmbedder, QdrantStore, MetricsStore]:
    return (
        OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]),
        QdrantStore(os.environ["QDRANT_URL"]),
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
    )


@st.cache_resource
def reranker() -> OllamaReranker:
    return OllamaReranker(os.environ["OLLAMA_BASE_URL"])


SCORE_MEANING = {
    "dense": "cosine similarity",
    "hybrid": "rank-fusion score (dense and BM25 rankings combined), not a similarity",
    "dense+rerank": "the reranker's probability that the chunk answers the question",
    "hybrid+rerank": "the reranker's probability that the chunk answers the question",
}


@st.cache_data(ttl=15)
def experiments() -> list[dict]:
    """The experiments that have a Qdrant collection, with their settings and point counts."""
    _, qdrant, metrics = clients()
    existing = {c.name for c in qdrant.client.get_collections().collections}
    return [
        {**row, "points": qdrant.count(row["name"])}
        for row in metrics.list_experiments()
        if row["name"] in existing
    ]


def describe(row: dict) -> str:
    config = row["config"]
    return (
        f"{row['name']} · {data.model_label(config['embed']['model'])} · {config['chunk']['strategy']} · "
        f"{row['documents']} document(s) · {row['points']} points"
    )


style.hero("Try a query", "Search one experiment and see the chunks it returns")

available = experiments()
if not available:
    st.info("No experiment has a Qdrant collection yet. Upload and embed a document on the Upload page first.")
    st.stop()

# The experiment and the search strategy sit outside the form: which strategies an experiment allows
# depends on the experiment, and a form's widgets only update when it is submitted.
row = st.selectbox("Experiment", available, format_func=describe)
has_sparse = bool(row["config"].get("index", {}).get("sparse"))
rerankers = data.reranker_models()
choices = [m for m in SEARCH_METHODS if (has_sparse or not m.startswith("hybrid")) and (rerankers or not m.endswith("+rerank"))]
c1, c2, c3 = st.columns([2, 2, 1])
method = c1.selectbox("Search strategy", choices, format_func=lambda m: m.replace("+", " + "), help="dense: nearest vectors. hybrid: dense plus BM25 keyword search, fused. + rerank: a reranker model re-scores the first hits.")
reranker_model, candidates = SearchConfig().reranker, 20
if method.endswith("+rerank"):
    reranker_model = c2.selectbox("Reranker", rerankers)
    candidates = int(c3.number_input("Candidates", min_value=1, max_value=100, value=20, help="Hits the reranker scores; one call to Ollama each."))
options = SearchConfig(method=method, reranker=reranker_model, candidates=candidates)
if not has_sparse:
    st.caption("This experiment has no BM25 vector (it was made without `index.sparse`), so hybrid search is not offered. Benchmark experiments made with a hybrid strategy have one.")
if not rerankers:
    st.caption("No reranker model that can generate is installed in Ollama, so the rerank strategies are not offered.")

with st.form("query"):
    query = st.text_input("Question", placeholder="How does multi-head attention work?")
    c1, c2 = st.columns([3, 1])
    top_k = c1.slider("Chunks to return (top k)", 1, 20, 5)
    modality = c2.radio("Content", ["any", "text", "table"], horizontal=True)
    submitted = st.form_submit_button("Search", type="primary", width="stretch")

if submitted and query.strip():
    embedder, qdrant, _ = clients()
    try:
        with st.spinner("Searching…"):
            result = search(
                query,
                ExperimentConfig.model_validate(row["config"]),
                embedder,
                qdrant,
                top_k,
                {"modality": modality} if modality != "any" else None,
                options=options,
                reranker=reranker(),
            )
    except Exception as e:  # noqa: BLE001  (Ollama or Qdrant not reachable: show it, do not crash)
        st.error(f"Search failed: {e}")
        st.stop()
    st.session_state["shown"] = (query, row["name"], result)

if "shown" not in st.session_state:
    st.caption("Choose an experiment, enter a question and press Search.")
    st.stop()

query, name, result = st.session_state["shown"]
st.caption(
    f"“{query}” in {name} ({result.method.replace('+', ' + ')}): {len(result.hits)} chunks · "
    f"embed {result.embed_ms:.0f} ms · search {result.search_ms:.0f} ms"
    + (f" · rerank {result.rerank_ms:.0f} ms" if result.method.endswith("+rerank") else "")
    + f" · score: {SCORE_MEANING[result.method]}"
)
if not result.hits:
    st.warning("No chunks found.")
else:
    st.markdown("".join(hit_card(h, style.MODEL_COLORS[0]) for h in result.hits), unsafe_allow_html=True)
