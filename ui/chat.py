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
    citations,
    from_dict,
)
from rag_lab.config import AgentConfig, ExperimentConfig, SearchConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import Hit
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


# A chat is the id in the page's URL (?chat=<id>), so a refresh or a bookmark finds it again. Its turns
# are in the database; the page shows them from the saved events, the same way it shows a live answer.


def new_chat() -> str:
    chat_id = uuid.uuid4().hex[:12]
    st.query_params["chat"] = chat_id
    return chat_id


def open_chat(chat_id: str) -> None:
    st.query_params["chat"] = chat_id


def delete_chat(chat_id: str) -> None:
    services()[2].delete_chat_session(chat_id)
    new_chat()


def new_trace(question: str, settings: dict | None) -> dict:
    return {
        "question": question, "settings": settings, "attempts": [], "retrieved": None,
        "thinking": "", "states": [], "done": None, "error": None,
    }


def apply(trace: dict, event) -> None:
    """Fold one event into what is shown for a turn. A live answer and a saved turn both go through here."""
    if isinstance(event, (Query, Rewrote)):
        query = event.text if isinstance(event, Query) else event.query
        trace["attempts"].append({"query": query, "retrieved": None, "graded": None})
    elif isinstance(event, Graded):
        trace["attempts"][-1]["graded"] = event
    elif isinstance(event, Retrieved):
        trace["retrieved"] = trace["attempts"][-1]["retrieved"] = event
    elif isinstance(event, Thinking):
        trace["thinking"] += event.text
    elif isinstance(event, ModelState):
        trace["states"].append(event)
    elif isinstance(event, Done):
        trace["done"], trace["thinking"] = event, event.thinking


def trace_from_turn(turn: dict) -> dict:
    """What is shown for a saved turn. A turn saved before its events were kept is rebuilt from its
    columns: one search, with no times for it and no verdict."""
    settings = turn["settings"] or {"model": turn["model"], "think": turn["think"]}
    trace = new_trace(turn["question"], settings)
    if turn["events"] is not None:
        for saved in turn["events"]:
            apply(trace, from_dict(saved))
        trace["error"] = turn["error"]
        return trace
    hits = [Hit(**h) for h in turn["hits"]]
    cited, unknown = citations(turn["answer"], len(hits))
    events = [
        Query(turn["query"], rewritten=turn["query"] != turn["question"]),
        Retrieved(hits=hits, method="", candidates=0, embed_ms=0.0, search_ms=0.0, rerank_ms=0.0),
        *[ModelState(**s) for s in turn["model_states"]],
        Done(turn["answer"], turn["thinking"], cited, unknown, turn["timings"], sum(turn["timings"].values())),
    ]
    for event in events:
        apply(trace, event)
    return trace


def messages_from(turns: list[dict]) -> list[dict]:
    messages = []
    for turn in turns:
        messages += [
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "trace": trace_from_turn(turn)},
        ]
    return messages


def history_of(messages: list[dict]) -> list[tuple[str, str]]:
    """The answered turns as the agent's history; a failed turn has no answer to remember."""
    history = []
    for user, assistant in zip(messages[::2], messages[1::2]):
        if not assistant["trace"]["error"]:
            history += [("User", user["content"]), ("Assistant", assistant["trace"]["done"].answer)]
    return history


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


def settings_line(settings: dict | None) -> str:
    """What a turn ran with; a turn saved before the settings were kept only has the model."""
    if not settings:
        return ""
    search = settings.get("search") or {}
    parts = [settings.get("model"), f"thinking {'on' if settings.get('think') else 'off'}"]
    if search:
        parts += [
            search["method"].replace("+", " + "),
            search["reranker"],
            f"{search['candidates']} candidates",
            f"top {settings['top_k']}",
            f"context {settings['num_ctx']}",
        ]
    return "Settings: " + ", ".join(str(p) for p in parts)


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
            if retrieved and retrieved.method:
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
            if trace["settings"]:
                st.caption(settings_line(trace["settings"]))
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


def answer_turn(
    question: str, graph, experiment: ExperimentConfig, cfg: AgentConfig, metrics: MetricsStore,
    chat_id: str, history: list[tuple[str, str]],
) -> dict:
    """Run one question, drawing each step as it happens. Returns what is shown for the turn."""
    trace = new_trace(question, cfg.model_dump(mode="json"))
    live, box = st.empty(), st.empty()
    text, step = "", "starting"
    with live.container():
        status = st.status("Starting", expanded=True)
        with status:
            attempts_box, chunks_box, thinking_box = st.empty(), st.empty(), st.empty()
    try:
        for event in run(graph, experiment, question, history, cfg=cfg, metrics=metrics, session_id=chat_id):
            apply(trace, event)
            if isinstance(event, StepStarted):
                step = event.node
                status.update(label=STEP_NAMES[step] + ("…" if step != "retrieve" else f" (hybrid, {cfg.search.candidates} candidates, then reranking)…"))
            elif isinstance(event, (Query, Rewrote, Graded)):
                attempts_box.markdown(attempt_lines(trace))
            elif isinstance(event, Retrieved):
                chunks_box.markdown(
                    "**Found**\n\n" + "\n".join(
                        f"{h.rank}. `{h.similarity:.3f}` {where(h)}" + (f" › {h.headings[-1]}" if h.headings else "")
                        for h in event.hits
                    )
                )
            elif isinstance(event, Thinking):
                thinking_box.markdown(f"**Thinking**\n\n{trace['thinking']}")
            elif isinstance(event, AnswerToken):
                text += event.text
                box.markdown(text + " ▌")
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

embedder, qdrant, metrics, reranker = services()
by_name = {row["name"]: row for row in available}

# Which chat is this? The id in the URL, or a new one. A chat belongs to the experiment it started in.
chat_id = st.query_params.get("chat")
session = metrics.get_chat_session(chat_id) if chat_id else None
owner = next((r for r in available if session and r["config_hash"] == session["config_hash"]), None)
if session and not owner:
    st.warning("That chat belongs to an experiment that is not available here, so a new chat was started.")
    chat_id = None
if not chat_id:
    chat_id = new_chat()
elif owner:
    st.session_state["experiment"] = owner["name"]
if st.session_state.get("experiment") not in by_name:
    st.session_state.pop("experiment", None)

if st.session_state.get("loaded_chat") != chat_id:
    st.session_state["messages"] = messages_from(metrics.get_chat_turns(chat_id))
    st.session_state["loaded_chat"] = chat_id

defaults = AgentConfig()
models = list(chat_models)
with st.sidebar:
    st.subheader("Chat settings")
    name = st.selectbox("Experiment", list(by_name), key="experiment", format_func=lambda n: describe(by_name[n]), on_change=new_chat, help="Changing it starts a new chat: a chat belongs to one experiment.")
    row = by_name[name]
    model = st.selectbox("Chat model", models, index=models.index(defaults.model) if defaults.model in models else 0)
    think = st.toggle("Thinking", value=defaults.think and chat_models[model], disabled=not chat_models[model], help="The model thinks before it answers, and the page shows it. Slower.")
    rerank_model = st.selectbox("Reranker", rerankers, index=rerankers.index(defaults.search.reranker) if defaults.search.reranker in rerankers else 0)
    top_k = st.slider("Chunks given to the model (top k)", 1, 10, defaults.top_k)
    candidates = int(st.number_input("Candidates", min_value=1, max_value=100, value=defaults.search.candidates, help="Hits the reranker scores; one call to Ollama each."))
    st.caption("Search: hybrid (dense and BM25 keywords) with reranking.")
    st.button("New chat", on_click=new_chat, width="stretch")
    if st.session_state["messages"]:
        with st.popover("Delete this chat", width="stretch"):
            st.write("This deletes the chat and all its turns.")
            st.button("Yes, delete it", on_click=delete_chat, args=(chat_id,), key="delete-chat")
    past = metrics.list_chat_sessions(row["config_hash"])
    if past:
        st.subheader("Past chats")
        for s in past:
            title = s["title"] if len(s["title"]) <= 40 else s["title"][:40] + "…"
            st.button(
                f"{title} · {s['turns']}",
                key=f"chat-{s['session_id']}",
                on_click=open_chat,
                args=(s["session_id"],),
                type="primary" if s["session_id"] == chat_id else "secondary",
                width="stretch",
                help=f"{s['turns']} turn(s), last used {s['updated_at']:%d %b %Y %H:%M} UTC",
            )

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
        if trace["done"] and not trace["done"].saved:
            st.warning(f"This turn was not saved to the database: {trace['done'].save_error}")

question = st.chat_input("Ask a question about the documents")
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    experiment = ExperimentConfig.model_validate(row["config"])
    cfg = AgentConfig(
        model=model,
        think=think,
        top_k=top_k,
        search=SearchConfig(method="hybrid+rerank", reranker=rerank_model, candidates=candidates),
    )
    graph = build_graph(experiment, embedder, qdrant, reranker, os.environ["OLLAMA_BASE_URL"], cfg)
    with st.chat_message("assistant"):
        trace = answer_turn(question, graph, experiment, cfg, metrics, chat_id, history_of(st.session_state["messages"]))
    st.session_state["messages"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "trace": trace},
    ]
    st.rerun()  # draws the turn from what was kept, and brings the list of past chats up to date
