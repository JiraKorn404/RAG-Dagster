"""The chatbot agent for documents as a LangGraph graph: condense -> retrieve -> grade -> generate, with a
retry (rewrite, retrieve again) when the chunks do not answer the question, and abstain when the retries fail.

Retrieval is our own `search()` (hybrid with reranking by default), not a LangChain retriever, so the
vectors, filters and scores stay under the same rules as everywhere else. The chat model only decides
what a small model does reliably: the standalone query, a yes or no, a different query and the wording
of the answer.

Every node reports what it does as events (agent/events.py) through LangGraph's custom stream;
`agent.run()` is the way to use the graph, with a `DocumentsFlow` that says what a saved turn needs."""

from dataclasses import asdict
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rag_lab.agent.documents.prompts import (
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
from rag_lab.agent.events import AnswerToken, Event, Graded, Query, Retrieved, Rewrote, citations
from rag_lab.agent.model import call_model, chat_model
from rag_lab.agent.run import Summary, step
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


class DocumentsFlow:
    """What a saved documents turn needs: the last query, the final hits, the citations and the search log."""

    kind = "documents"
    schema_name = None

    def __init__(self, experiment: ExperimentConfig, cfg: AgentConfig):
        self.experiment, self.cfg = experiment, cfg
        self.config_hash = experiment.config_hash()

    def summarise(self, question: str, events: list[Event], final: dict) -> Summary:
        if not final:  # the turn failed: what was searched and found until then
            searched, found = question, []
            for event in events:
                if isinstance(event, Query):
                    searched = event.text
                elif isinstance(event, Rewrote):
                    searched = event.query
                elif isinstance(event, Retrieved):
                    found = event.hits
            return Summary(query=searched, hits=[asdict(h) for h in found])

        result = final["retrieval"]
        hits = result.hits
        cited, unknown = citations(final["answer"], len(hits))

        def log(metrics: MetricsStore) -> None:
            metrics.add_search_log(
                self.config_hash,
                final["query"],
                self.cfg.top_k,
                result.embed_ms,
                result.search_ms,
                result.total_ms,
                hits[0].similarity if hits else None,
            )

        return Summary(
            query=final["query"],
            hits=[asdict(h) for h in hits],
            answer=final["answer"],
            thinking=final["thinking"],
            abstained=final.get("abstained", False),
            cited=cited,
            unknown_citations=unknown,
            log=log,
        )
