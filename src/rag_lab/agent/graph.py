"""The chatbot agent as a LangGraph graph: condense -> retrieve -> grade -> generate, with a retry
(rewrite, retrieve again) when the chunks do not answer the question, and abstain when the retries fail.

Retrieval is our own `search()` (hybrid with reranking by default), not a LangChain retriever, so the
vectors, filters and scores stay under the same rules as everywhere else. The chat model only decides
what a small model does reliably: the standalone query, a yes or no, a different query and the wording
of the answer.

Every node reports what it does as events (events.py) through LangGraph's custom stream; `run()` is the
way to use the graph: it yields the events and saves the turn."""

import functools
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rag_lab.agent.events import (
    AnswerToken,
    Done,
    Event,
    Graded,
    ModelState,
    Query,
    Retrieved,
    Rewrote,
    StepFinished,
    StepStarted,
    Thinking,
    citations,
    to_dict,
)
from rag_lab.agent.model import call_model, chat_model
from rag_lab.agent.prompts import (
    ANSWER_SYSTEM,
    CONDENSE_SYSTEM,
    GRADE_SYSTEM,
    REWRITE_SYSTEM,
    answer_prompt,
    condense_prompt,
    grade_prompt,
    not_found,
    rewrite_prompt,
)
from rag_lab.config import AgentConfig, ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import SearchResult, search
from rag_lab.storage.qdrant import QdrantStore


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]  # ("User" or "Assistant", text), oldest first
    standalone: str  # the question made standalone; what the answer is written for
    query: str  # what is searched for: the standalone question, then a rewrite of it on a retry
    tried: list[str]  # every query searched
    retrieval: SearchResult  # the latest search
    enough: bool  # the chunks answer the question
    rewrites: int
    abstained: bool
    answer: str
    thinking: str  # the model's thinking while it wrote the answer; empty when `think` is off


def step(node):
    """Report when a node starts and how long it took."""

    @functools.wraps(node)
    def reported(state: AgentState) -> dict:
        emit = get_stream_writer()
        emit(StepStarted(node.__name__))
        start = time.perf_counter()
        update = node(state)
        emit(StepFinished(node.__name__, (time.perf_counter() - start) * 1000))
        return update

    return reported


def build_graph(
    experiment: ExperimentConfig,
    embedder: OllamaEmbedder,
    store: QdrantStore,
    reranker: OllamaReranker,
    base_url: str,
    cfg: AgentConfig | None = None,
):
    cfg = cfg or AgentConfig()
    quick_llm = chat_model(base_url, cfg, think=False)  # condense, grade and rewrite never think
    answer_llm = chat_model(base_url, cfg, think=cfg.think)

    @step
    def condense(state: AgentState) -> dict:
        history = state.get("history", [])[-2 * cfg.history_turns :]
        query = state["question"]
        if history:
            query, _ = call_model(
                quick_llm,
                base_url,
                cfg,
                "condense",
                think=False,
                stream=False,
                messages=[("system", CONDENSE_SYSTEM), ("human", condense_prompt(query, history))],
            )
            query = query.strip() or state["question"]
        get_stream_writer()(Query(query, rewritten=query != state["question"]))
        return {"standalone": query, "query": query, "tried": [], "rewrites": 0}

    @step
    def retrieve(state: AgentState) -> dict:
        result = search(
            state["query"],
            experiment,
            embedder,
            store,
            top_k=cfg.top_k,
            options=cfg.search,
            reranker=reranker,
        )
        get_stream_writer()(
            Retrieved(
                hits=result.hits,
                method=result.method,
                candidates=cfg.search.candidates,
                embed_ms=result.embed_ms,
                search_ms=result.search_ms,
                rerank_ms=result.rerank_ms,
            )
        )
        return {"retrieval": result, "tried": [*state["tried"], state["query"]]}

    @step
    def grade(state: AgentState) -> dict:
        best = max((hit.similarity for hit in state["retrieval"].hits), default=0.0)
        by = "score"
        if best >= cfg.enough_score:
            enough = True
        elif best < cfg.missing_score:
            enough = False
        else:  # in between: the model reads the chunks
            by = "model"
            reply, _ = call_model(
                quick_llm,
                base_url,
                cfg,
                "grade",
                think=False,
                stream=False,
                messages=[
                    ("system", GRADE_SYSTEM),
                    ("human", grade_prompt(state["standalone"], state["retrieval"].hits)),
                ],
            )
            enough = reply.strip().lower().startswith("yes")
        get_stream_writer()(Graded(enough, best, by))
        return {"enough": enough}

    @step
    def rewrite(state: AgentState) -> dict:
        reply, _ = call_model(
            quick_llm,
            base_url,
            cfg,
            "rewrite",
            think=False,
            stream=False,
            messages=[
                ("system", REWRITE_SYSTEM),
                ("human", rewrite_prompt(state["standalone"], state["tried"], state["retrieval"].hits)),
            ],
        )
        lines = reply.strip().splitlines()
        query = lines[0].strip(' "“”') if lines else ""
        query = query or state["query"]
        get_stream_writer()(Rewrote(query, previous=state["query"]))
        return {"query": query, "rewrites": state["rewrites"] + 1}

    @step
    def generate(state: AgentState) -> dict:
        answer, thinking = call_model(
            answer_llm,
            base_url,
            cfg,
            "generate",
            think=cfg.think,
            stream=True,
            messages=[
                ("system", ANSWER_SYSTEM),
                ("human", answer_prompt(state["standalone"], state["retrieval"].hits)),
            ],
        )
        return {"answer": answer, "thinking": thinking}

    @step
    def abstain(state: AgentState) -> dict:
        answer = not_found(state["tried"])
        get_stream_writer()(AnswerToken(answer))
        return {"answer": answer, "thinking": "", "abstained": True}

    def after_grade(state: AgentState) -> str:
        if state["enough"]:
            return "generate"
        return "rewrite" if state["rewrites"] < cfg.max_rewrites else "abstain"

    graph = StateGraph(AgentState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def run(
    graph,
    experiment: ExperimentConfig,
    question: str,
    history: list[tuple[str, str]] | None = None,
    *,
    cfg: AgentConfig | None = None,
    metrics: MetricsStore | None = None,
    session_id: str | None = None,
) -> Iterator[Event]:
    """Answer one question, yielding the events as they happen and `Done` last. With `metrics` and a
    `session_id` the turn is saved to `chat_turns` (a failed turn too, with its error, before the error
    is raised again) and the retrieval to the search log. A save that fails does not hide the answer:
    `Done.saved` is false and `Done.save_error` says why. `cfg` is the one the graph was built with
    (it records the model, the settings and the top k)."""
    cfg = cfg or AgentConfig()
    start = time.perf_counter()
    timings: dict[str, float] = {}
    states: list[ModelState] = []
    stored: list[Event] = []  # what a saved turn replays from: every event but the token-by-token ones
    searched, found, step_now = question, [], "starting"  # the last query, the last hits, the running step
    final: AgentState = {}

    def save(answer: str, thinking: str, **extra) -> str | None:
        """Save the turn. Returns why that failed, or None."""
        if not (metrics and session_id):
            return None
        try:
            metrics.add_chat_turn(
                experiment.config_hash(),
                session_id,
                question,
                searched,
                cfg.model,
                cfg.think,
                answer,
                thinking,
                [asdict(h) for h in found],
                timings,
                [asdict(s) for s in states],
                events=[to_dict(e) for e in stored],
                total_ms=(time.perf_counter() - start) * 1000,
                settings=cfg.model_dump(mode="json"),
                **extra,
            )
        except Exception as e:  # noqa: BLE001  (a failed save is reported, never hides the answer)
            return str(e)
        return None

    try:
        for mode, payload in graph.stream(
            {"question": question, "history": history or []}, stream_mode=["custom", "values"]
        ):
            if mode == "values":
                final = payload
                continue
            if isinstance(payload, StepStarted):
                step_now = payload.node
            elif isinstance(payload, StepFinished):
                timings[payload.node] = timings.get(payload.node, 0.0) + payload.ms  # a retry repeats steps
            elif isinstance(payload, ModelState):
                states.append(payload)
            elif isinstance(payload, Query):
                searched = payload.text
            elif isinstance(payload, Rewrote):
                searched = payload.query
            elif isinstance(payload, Retrieved):
                found = payload.hits
            if not isinstance(payload, (Thinking, AnswerToken)):
                stored.append(payload)
            yield payload
    except Exception as e:
        save("", "", error=f"{step_now} failed: {e}")  # the original error is the one to raise
        raise

    hits = final["retrieval"].hits
    cited, unknown = citations(final["answer"], len(hits))
    done = Done(
        answer=final["answer"],
        thinking=final["thinking"],
        cited=cited,
        unknown_citations=unknown,
        timings=timings,
        total_ms=(time.perf_counter() - start) * 1000,
        abstained=final.get("abstained", False),
    )
    stored.append(done)
    failure = None
    if metrics and session_id:
        result = final["retrieval"]
        try:
            metrics.add_search_log(
                experiment.config_hash(),
                final["query"],
                cfg.top_k,
                result.embed_ms,
                result.search_ms,
                result.total_ms,
                hits[0].similarity if hits else None,
            )
        except Exception as e:  # noqa: BLE001
            failure = str(e)
        failure = save(done.answer, done.thinking, abstained=done.abstained, cited=cited) or failure
    yield replace(done, saved=False, save_error=failure) if failure else done
