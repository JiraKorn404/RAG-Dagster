"""The command line for the documents flow: python -m rag_lab.agent documents ["question"] --experiment <name>

Answers from the experiment's documents with hybrid search and reranking, printing every step as it
happens: the query, the retrieved chunks, the model's thinking, the answer and the model's state. Each
turn is saved to `chat_turns`. Without a question it reads questions from the prompt (an empty line
ends) and keeps the conversation, so follow-ups work.
Needs OLLAMA_BASE_URL, QDRANT_URL and METRICS_DATABASE_URL in the environment (already set inside the
dagster-code container). The experiment must have been made with `index.sparse`."""

import os
import sys
import uuid

from rag_lab.agent import run
from rag_lab.agent.documents.graph import DocumentsFlow, build_graph
from rag_lab.agent.events import Done
from rag_lab.agent.printer import Printer
from rag_lab.config import AgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.migrate import apply_migrations
from rag_lab.metrics.store import MetricsStore
from rag_lab.reranking import OllamaReranker
from rag_lab.search import load_experiment
from rag_lab.storage.qdrant import QdrantStore


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set")
    return value


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("documents", help="ask questions of the documents in an experiment")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--no-think", action="store_true", help="write the answer without thinking")
    parser.set_defaults(main=main)


def main(args) -> None:
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
    flow = DocumentsFlow(experiment, cfg)

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
            for event in run(graph, flow, question, history, metrics=metrics, session_id=session_id):
                show(event)
                if isinstance(event, Done):
                    answer = event.answer
        except ValueError as e:  # for example an experiment without the BM25 vector
            sys.exit(str(e))
        if args.question:
            return
        history += [("User", question), ("Assistant", answer)]
        question = None
