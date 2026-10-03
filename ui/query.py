import html
import os

import data
import streamlit as st
import style

from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.search import search
from rag_lab.storage.qdrant import QdrantStore


@st.cache_resource
def clients() -> tuple[OllamaEmbedder, QdrantStore, MetricsStore]:
    return (
        OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]),
        QdrantStore(os.environ["QDRANT_URL"]),
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
    )


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


def hit_card(hit, color: str) -> str:
    body = " ".join(hit.text.split())
    snippet = html.escape(body[:330] + ("…" if len(body) > 330 else ""))
    where = f"{hit.source_file}, p. {hit.page}" if hit.page and hit.source_file else (
        f"p. {hit.page}" if hit.page else (hit.source_file or "page ?")
    )
    return (
        '<div class="hit"><div class="hit-top">'
        f'<span class="rank">{hit.rank}</span><span class="sim">{hit.similarity:.3f}</span>'
        f'<div class="bar"><span style="width:{max(0.0, min(1.0, hit.similarity)) * 100:.0f}%;background:{color}"></span></div>'
        "</div>"
        f'<div class="badges"><span class="badge {hit.modality}">{hit.modality}</span>'
        f'<span class="badge page">{html.escape(where)}</span></div>'
        + (f'<div class="heading">{html.escape(" › ".join(hit.headings))}</div>' if hit.headings else "")
        + f'<div class="snippet">{snippet}</div>'
        f"<details><summary>Full text</summary><pre>{html.escape(hit.text)}</pre></details></div>"
    )


style.hero("Try a query", "Search one experiment and see the chunks it returns")

available = experiments()
if not available:
    st.info("No experiment has a Qdrant collection yet. Upload and embed a document on the Upload page first.")
    st.stop()

with st.form("query"):
    row = st.selectbox("Experiment", available, format_func=describe)
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
    f"“{query}” in {name}: {len(result.hits)} chunks · embed {result.embed_ms:.0f} ms · "
    f"search {result.search_ms:.0f} ms"
)
if not result.hits:
    st.warning("No chunks found.")
else:
    st.markdown("".join(hit_card(h, style.MODEL_COLORS[0]) for h in result.hits), unsafe_allow_html=True)
