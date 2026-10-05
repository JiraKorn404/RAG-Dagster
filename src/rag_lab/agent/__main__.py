"""CLI: python -m rag_lab.agent ["question"] --experiment <name> [--top-k 5] [--model <name>] [--no-think]

Answers from the experiment's documents with hybrid search and reranking, printing every step as it
happens: the query, the retrieved chunks, the model's thinking, the answer and the model's state. Each
turn is saved to `chat_turns`. Without a question it reads questions from the prompt (an empty line
ends) and keeps the conversation, so follow-ups work.
Needs OLLAMA_BASE_URL, QDRANT_URL and METRICS_DATABASE_URL in the environment (already set inside the
dagster-code container). The experiment must have been made with `index.sparse`."""

import argparse
import os
import sys
import uuid

from rag_lab.agent import build_graph, run
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
)
from rag_lab.config import AgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.migrate import apply_migrations
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import load_experiment
from rag_lab.storage.qdrant import QdrantStore

DIM, RESET = "\033[2m", "\033[0m"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set")
    return value


class Printer:
    """Prints events as they arrive; thinking is dimmed and the answer follows it."""

    def __init__(self) -> None:
        self.streaming = ""  # "thinking" or "answer" while a stream is being printed

    def _stream(self, kind: str, text: str) -> None:
        if self.streaming != kind:
            self._end_stream()
            print(f"{DIM}thinking:{RESET}\n{DIM}" if kind == "thinking" else "answer:")
            self.streaming = kind
        print(text, end="", flush=True)

    def _end_stream(self) -> None:
        if self.streaming:
            print(RESET if self.streaming == "thinking" else "")
            self.streaming = ""

    def __call__(self, event: Event) -> None:
        if isinstance(event, Thinking):
            return self._stream("thinking", event.text)
        if isinstance(event, AnswerToken):
            return self._stream("answer", event.text)
        self._end_stream()
        if isinstance(event, StepStarted):
            print(f"\n-> {event.node}", flush=True)
        elif isinstance(event, StepFinished):
            print(f"   {event.node} took {event.ms:.0f} ms")
        elif isinstance(event, Query):
            print(f"   query{' (rewritten)' if event.rewritten else ''}: {event.text}")
        elif isinstance(event, Graded):
            verdict = "the chunks answer it" if event.enough else "the chunks do not answer it"
            print(f"   {verdict} (best score {event.best_score:.3f}, decided by the {event.by})")
        elif isinstance(event, Rewrote):
            print(f"   trying a different query: {event.query}")
        elif isinstance(event, Retrieved):
            print(
                f"   {event.method}, {event.candidates} candidates: embed {event.embed_ms:.0f} ms, "
                f"search {event.search_ms:.0f} ms, rerank {event.rerank_ms:.0f} ms"
            )
            for number, hit in enumerate(event.hits, start=1):
                where = f"{hit.source_file or hit.doc_id} p.{hit.page}" if hit.page else (hit.source_file or hit.doc_id)
                heading = f" > {' > '.join(hit.headings)}" if hit.headings else ""
                print(f"   [{number}] score {hit.similarity:.3f} [{hit.modality}] {where}{heading}")
        elif isinstance(event, ModelState):
            speed = f", {event.tokens_per_s:.1f} tok/s" if event.tokens_per_s else ""
            loaded = {True: "was loaded", False: "was cold", None: "load state unknown"}[event.loaded]
            print(
                f"   model {event.model} ({loaded}), thinking {'on' if event.think else 'off'}: "
                f"{event.prompt_tokens} of {event.num_ctx} context in, {event.output_tokens} out{speed}"
            )
            if event.context_full:
                print("   WARNING: the prompt filled the context window, so Ollama cut it")
        elif isinstance(event, Done):
            if event.abstained:
                print("\nabstained: the documents do not answer it", end="")
            print(f"\ncited: {event.cited or 'nothing'}", end="")
            print(f"; no such passage: {event.unknown_citations}" if event.unknown_citations else "")
            steps = ", ".join(f"{node} {ms:.0f}" for node, ms in event.timings.items())
            print(f"total {event.total_ms:.0f} ms ({steps})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.agent", description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--no-think", action="store_true", help="write the answer without thinking")
    args = parser.parse_args()

    overrides = {
        **({"top_k": args.top_k} if args.top_k else {}),
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
    }
    cfg = AgentConfig(**overrides)

    base_url = _env("OLLAMA_BASE_URL")
    database_url = _env("METRICS_DATABASE_URL")
    apply_migrations(database_url)
    metrics = MetricsStore(database_url)
    try:
        _, experiment = load_experiment(metrics, args.experiment)
    except ValueError as e:
        sys.exit(str(e))
    graph = build_graph(
        experiment,
        OllamaEmbedder(base_url),
        QdrantStore(_env("QDRANT_URL")),
        OllamaReranker(base_url),
        base_url,
        cfg,
    )

    session_id = uuid.uuid4().hex[:12]
    show = Printer()
    history: list[tuple[str, str]] = []
    question = args.question
    while True:
        if question is None:
            question = input("> ").strip()
            if not question:
                return
        answer = ""
        try:
            for event in run(graph, experiment, question, history, cfg=cfg, metrics=metrics, session_id=session_id):
                show(event)
                if isinstance(event, Done):
                    answer = event.answer
        except ValueError as e:  # for example an experiment without the BM25 vector
            sys.exit(str(e))
        if args.question:
            return
        history += [("User", question), ("Assistant", answer)]
        question = None


if __name__ == "__main__":
    main()
