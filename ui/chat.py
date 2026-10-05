import os
import uuid

import data
import streamlit as st
import style
from hits import hit_card

from rag_lab.agent import build_graph, run
from rag_lab.agent.events import (
    AnswerToken,
    Done,
    Graded,
    ModelState,
    Query,
    Retrieved,
    Rewrote,
    StepStarted,
    Thinking,
)
from rag_lab.config import AgentConfig, ExperimentConfig, SearchConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.storage.qdrant import QdrantStore

STEP_NAMES = {
    "condense": "Reading the question",
    "retrieve": "Searching",
    "grade": "Checking the chunks",
    "rewrite": "Trying a different query",
    "generate": "Writing the answer",
    "abstain": "Not found",
}


@st.cache_resource
def services() -> tuple[OllamaEmbedder, QdrantStore, MetricsStore, OllamaReranker]:
    base_url = os.environ["OLLAMA_BASE_URL"]
    return (
        OllamaEmbedder(base_url),
        QdrantStore(os.environ["QDRANT_URL"]),
        MetricsStore(os.environ["METRICS_DATABASE_URL"]),
        OllamaReranker(base_url),
    )


@st.cache_data(ttl=15)
def experiments() -> list[dict]:
    """The experiments that have a Qdrant collection and a BM25 vector (hybrid search needs it)."""
    _, qdrant, metrics, _ = services()
    existing = {c.name for c in qdrant.client.get_collections().collections}
    return [
        {**row, "points": qdrant.count(row["name"])}
        for row in metrics.list_experiments()
        if row["name"] in existing and row["config"].get("index", {}).get("sparse")
    ]


def describe(row: dict) -> str:
    config = row["config"]
    return (
        f"{row['name']} · {data.model_label(config['embed']['model'])} · {config['chunk']['strategy']} · "
        f"{row['documents']} document(s) · {row['points']} points"
    )


def reset() -> None:
    st.session_state["messages"] = []
    st.session_state["history"] = []  # (role, text) pairs of the answered turns, for the agent
    st.session_state["session_id"] = uuid.uuid4().hex[:12]


def where(hit) -> str:
    name = hit.source_file or hit.doc_id
    return f"{name}, p. {hit.page}" if hit.page else name


def state_line(s: ModelState) -> str:
    loaded = {True: "already loaded", False: "had to be loaded", None: "load state unknown"}[s.loaded]
    speed = f" · {s.tokens_per_s:.1f} tok/s" if s.tokens_per_s else ""
    return (
        f"**{STEP_NAMES[s.node]}** · {s.model} ({loaded}) · thinking {'on' if s.think else 'off'} · "
        f"{s.prompt_tokens} of {s.num_ctx} context in, {s.output_tokens} out{speed}"
    )


def verdict(graded: Graded | None) -> str:
    if not graded:
        return ""
    answers = "the chunks answer it" if graded.enough else "the chunks do not answer it"
    return f" — best score {graded.best_score:.3f}, {answers} (decided by the {graded.by})"


def attempt_lines(trace: dict) -> str:
    return "\n".join(
        f"{n}. **Searched for:** {a['query']}{verdict(a['graded'])}" for n, a in enumerate(trace["attempts"], start=1)
    )


def show_answer(done: Done) -> None:
    (st.info if done.abstained else st.markdown)(done.answer)


def show_trace(trace: dict) -> None:
    """Everything the agent did for one answer."""
    done, retrieved = trace["done"], trace["retrieved"]
    with st.expander("How this was answered"):
        steps, thinking, chunks = st.tabs(
            ["Steps and model", "Thinking", f"Retrieved chunks ({len(retrieved.hits) if retrieved else 0})"]
        )
        with steps:
            st.markdown(f"**Question:** {trace['question']}")
            if trace["attempts"]:
                st.markdown(attempt_lines(trace))
            if done:
                rows = "".join(f"| {STEP_NAMES[node]} | {ms:.0f} ms |\n" for node, ms in done.timings.items())
                st.markdown(f"| Step | Time |\n|---|---|\n{rows}| **Total** | **{done.total_ms:.0f} ms** |")
            if retrieved:
                st.caption(
                    f"Search: {retrieved.method.replace('+', ' + ')}, {retrieved.candidates} candidates. "
                    f"Embedding the query {retrieved.embed_ms:.0f} ms, Qdrant {retrieved.search_ms:.0f} ms, "
                    f"reranking {retrieved.rerank_ms:.0f} ms."
                )
            for s in trace["states"]:
                st.markdown(state_line(s))
                if s.context_full:
                    st.warning("The prompt filled the context window, so Ollama cut it. Raise `num_ctx`.")
            if done and done.unknown_citations:
                st.warning(f"The answer cites passage(s) {done.unknown_citations}, which do not exist.")
        with thinking:
            if trace["thinking"]:
                st.markdown(trace["thinking"])
            else:
                st.caption("The model did not think for this answer.")
        with chunks:
            if not retrieved or not retrieved.hits:
                st.caption("Nothing was retrieved.")
            else:
                st.caption("The number on a card is the passage number the answer cites. Score: the reranker's probability that the chunk answers the question.")
                cited = done.cited if done else []
                st.markdown(
                    "".join(hit_card(h, style.MODEL_COLORS[0], cited=h.rank in cited) for h in retrieved.hits),
                    unsafe_allow_html=True,
                )


def answer_turn(question: str, graph, experiment: ExperimentConfig, cfg: AgentConfig, metrics: MetricsStore) -> dict:
    """Run one question, drawing each step as it happens. Returns what to keep for the history."""
    trace = {
        "question": question, "attempts": [], "retrieved": None,
        "thinking": "", "states": [], "done": None, "error": None,
    }
    live, box = st.empty(), st.empty()
    text, step = "", "starting"
    with live.container():
        status = st.status("Starting", expanded=True)
        with status:
            attempts_box, chunks_box, thinking_box = st.empty(), st.empty(), st.empty()
    try:
        for event in run(
            graph, experiment, question, st.session_state["history"],
            cfg=cfg, metrics=metrics, session_id=st.session_state["session_id"],
        ):
            if isinstance(event, StepStarted):
                step = event.node
                status.update(label=STEP_NAMES[step] + ("…" if step != "retrieve" else f" (hybrid, {cfg.search.candidates} candidates, then reranking)…"))
            elif isinstance(event, (Query, Rewrote)):
                query = event.text if isinstance(event, Query) else event.query
                trace["attempts"].append({"query": query, "retrieved": None, "graded": None})
                attempts_box.markdown(attempt_lines(trace))
            elif isinstance(event, Graded):
                trace["attempts"][-1]["graded"] = event
                attempts_box.markdown(attempt_lines(trace))
            elif isinstance(event, Retrieved):
                trace["retrieved"] = trace["attempts"][-1]["retrieved"] = event
                chunks_box.markdown(
                    "**Found**\n\n" + "\n".join(
                        f"{h.rank}. `{h.similarity:.3f}` {where(h)}" + (f" › {h.headings[-1]}" if h.headings else "")
                        for h in event.hits
                    )
                )
            elif isinstance(event, Thinking):
                trace["thinking"] += event.text
                thinking_box.markdown(f"**Thinking**\n\n{trace['thinking']}")
            elif isinstance(event, AnswerToken):
                text += event.text
                box.markdown(text + " ▌")
            elif isinstance(event, ModelState):
                trace["states"].append(event)
            elif isinstance(event, Done):
                trace["done"] = event
    except Exception as e:  # noqa: BLE001  (Ollama or Qdrant not reachable: show where it failed, do not crash)
        trace["error"] = f"{STEP_NAMES.get(step, 'Starting')} failed: {e}"
        status.update(label=trace["error"], state="error")
        return trace
    live.empty()
    with box.container():
        show_answer(trace["done"])
    return trace


style.hero("Chatbot", "Ask about the documents in an experiment and see how each answer was found")

available = experiments()
rerankers = data.reranker_models()
chat_models = data.chat_models()
if not available:
    st.info("No experiment with a BM25 vector exists yet. On the Upload page, tick the BM25 box when embedding a document.")
    st.stop()
if not rerankers or not chat_models:
    st.info("Ollama has no " + ("reranker" if not rerankers else "chat model that can use tools") + " installed, so the chatbot cannot run.")
    st.stop()

defaults = AgentConfig()
models = list(chat_models)
with st.sidebar:
    st.subheader("Chat settings")
    row = st.selectbox("Experiment", available, format_func=describe)
    model = st.selectbox("Chat model", models, index=models.index(defaults.model) if defaults.model in models else 0)
    think = st.toggle("Thinking", value=defaults.think and chat_models[model], disabled=not chat_models[model], help="The model thinks before it answers, and the page shows it. Slower.")
    rerank_model = st.selectbox("Reranker", rerankers, index=rerankers.index(defaults.search.reranker) if defaults.search.reranker in rerankers else 0)
    top_k = st.slider("Chunks given to the model (top k)", 1, 10, defaults.top_k)
    candidates = int(st.number_input("Candidates", min_value=1, max_value=100, value=defaults.search.candidates, help="Hits the reranker scores; one call to Ollama each."))
    st.caption("Search: hybrid (dense and BM25 keywords) with reranking.")
    st.button("New chat", on_click=reset, width="stretch")

if st.session_state.get("chat_experiment") != row["name"]:  # another experiment is another conversation
    reset()
    st.session_state["chat_experiment"] = row["name"]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            continue
        trace = message["trace"]
        if trace["error"]:
            st.error(trace["error"])
        else:
            show_answer(trace["done"])
        show_trace(trace)

question = st.chat_input("Ask a question about the documents")
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    embedder, qdrant, metrics, reranker = services()
    experiment = ExperimentConfig.model_validate(row["config"])
    cfg = AgentConfig(
        model=model,
        think=think,
        top_k=top_k,
        search=SearchConfig(method="hybrid+rerank", reranker=rerank_model, candidates=candidates),
    )
    graph = build_graph(experiment, embedder, qdrant, reranker, os.environ["OLLAMA_BASE_URL"], cfg)
    with st.chat_message("assistant"):
        trace = answer_turn(question, graph, experiment, cfg, metrics)
        if not trace["error"]:
            show_trace(trace)
    st.session_state["messages"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "trace": trace},
    ]
    if not trace["error"]:
        st.session_state["history"] += [("User", question), ("Assistant", trace["done"].answer)]
