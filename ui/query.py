import html
import os
from collections import Counter

import data
import streamlit as st
import style

from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.search import search
from rag_lab.storage.qdrant import QdrantStore


@st.cache_resource
def clients() -> tuple[OllamaEmbedder, QdrantStore]:
    return OllamaEmbedder(os.environ["OLLAMA_BASE_URL"]), QdrantStore(os.environ["QDRANT_URL"])


@st.cache_data(ttl=15)
def experiments() -> list[dict]:
    """Newest config of each experiment name that has a Qdrant collection."""
    existing = {c.name for c in clients()[1].client.get_collections().collections}
    rows = data.frame("SELECT DISTINCT ON (name) name, config FROM experiments ORDER BY name, created_at DESC")
    return [r.config for r in rows.itertuples() if r.name in existing]


def hit_card(hit, color: str, agree: int, total: int) -> str:
    body = " ".join(hit.text.split())
    snippet = html.escape(body[:230] + ("…" if len(body) > 230 else ""))
    where = f"p. {hit.page}" if hit.page else "page ?"
    shared = (
        f'<span class="badge agree">same page in {agree}/{total}</span>'
        if total > 1 and agree > 1
        else '<span class="badge solo">only here</span>' if total > 1 else ""
    )
    entry = html.escape(f"{{source_file: {hit.source_file}, page: {hit.page}}}")
    return (
        '<div class="hit"><div class="hit-top">'
        f'<span class="rank">{hit.rank}</span><span class="sim">{hit.similarity:.3f}</span>'
        f'<div class="bar"><span style="width:{max(0.0, min(1.0, hit.similarity)) * 100:.0f}%;background:{color}"></span></div>'
        "</div>"
        f'<div class="badges"><span class="badge {hit.modality}">{hit.modality}</span>'
        f'<span class="badge page">{html.escape(where)}</span>{shared}</div>'
        + (f'<div class="heading">{html.escape(" › ".join(hit.headings))}</div>' if hit.headings else "")
        + f'<div class="snippet">{snippet}</div>'
        f'<details><summary>Full text</summary><pre>{html.escape(hit.text)}</pre>'
        f"<div>expected entry: <code>{entry}</code></div></details></div>"
    )


style.hero("Try a query", "One question, every experiment: see what each model and chunker retrieves")

configs = [ExperimentConfig.model_validate(c) for c in experiments()]
if not configs:
    st.info("No experiment has a Qdrant collection yet. Ingest a document first.")
    st.stop()

models = data.model_order(data.model_label(c.embed.model) for c in configs)
strategies = data.strategy_order(c.chunk.strategy for c in configs)
with st.form("query"):
    query = st.text_input("Question", placeholder="How does multi-head attention work?")
    c1, c2, c3, c4 = st.columns([1, 1, 2, 3])
    top_k = c1.slider("Results per experiment", 1, 10, 3)
    modality = c2.radio("Content", ["any", "text", "table"], horizontal=False)
    chosen_models = c3.multiselect("Models", models, default=models)
    chosen_strategies = c4.multiselect("Chunking strategies", strategies, default=strategies)
    submitted = st.form_submit_button("Search", type="primary", width="stretch")

if submitted and query.strip():
    selected = sorted(
        (
            c
            for c in configs
            if data.model_label(c.embed.model) in chosen_models and c.chunk.strategy in chosen_strategies
        ),
        key=lambda c: (models.index(data.model_label(c.embed.model)), strategies.index(c.chunk.strategy)),
    )
    embedder, store = clients()
    embedded, results = {}, {}
    try:
        with st.spinner("Searching…"):
            for c in selected:
                key = (c.embed.model, c.embed.dimension, c.embed.query_instruction)
                if key not in embedded:  # one embedding call per model, not per experiment
                    embedded[key] = embedder.embed([query], c.embed, kind="query")
                results[c.name] = search(
                    query,
                    c,
                    embedder,
                    store,
                    top_k,
                    {"modality": modality} if modality != "any" else None,
                    embedded=embedded[key],
                )
    except Exception as e:  # noqa: BLE001  (Ollama or Qdrant not reachable: show it, do not crash)
        st.error(f"Search failed: {e}")
        st.stop()
    st.session_state["shown"] = (query, selected, results, embedded)

if "shown" not in st.session_state:
    st.caption("Enter a question and press Search.")
    st.stop()

query, selected, results, embedded = st.session_state["shown"]
if not selected:
    st.warning("No experiments match the selected models and strategies.")
    st.stop()

pages = Counter(
    (h.source_file, h.page)
    for r in results.values()
    for h in {(h.source_file, h.page): h for h in r.hits}.values()  # count each page once per experiment
)
st.caption(
    f"“{query}” across {len(selected)} experiments. "
    "Similarity scores are only comparable within one model; the badges show how many experiments return the same page."
)
for model in models:
    group = [c for c in selected if data.model_label(c.embed.model) == model]
    if not group:
        continue
    color = style.MODEL_COLORS[models.index(model) % len(style.MODEL_COLORS)]
    first = group[0]
    dim = len(embedded[(first.embed.model, first.embed.dimension, first.embed.query_instruction)].vectors[0])
    st.markdown(
        f'<div class="model-head"><span class="dot" style="background:{color}"></span>'
        f'<span class="name">{html.escape(first.embed.model)}</span>'
        f'<span class="meta">{dim} dimensions</span></div>',
        unsafe_allow_html=True,
    )
    for col, c in zip(st.columns(len(group)), group):
        r = results[c.name]
        cards = "".join(
            hit_card(h, color, pages[(h.source_file, h.page)], len(selected)) for h in r.hits
        ) or '<div class="exp-meta">No hits.</div>'
        col.markdown(
            f'<div class="exp-head">{c.chunk.strategy}</div>'
            f'<div class="exp-meta">embed {r.embed_ms:.0f} ms · search {r.search_ms:.0f} ms</div>{cards}',
            unsafe_allow_html=True,
        )
