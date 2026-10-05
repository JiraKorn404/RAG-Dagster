"""The command line for the database flow: python -m rag_lab.agent sql ["question"] --schema <name>

Answers questions about the tables of an imported schema: the model writes a SELECT, a guard and the
database check it, it runs as the read-only role, and the answer is written from its rows. Every step is
printed as it happens, and each turn is saved to `chat_turns`. Without a question it reads questions from
the prompt (an empty line ends) and keeps the conversation, so follow-ups work.
Needs OLLAMA_BASE_URL, QDRANT_URL, METRICS_DATABASE_URL and SQL_READER_URL in the environment (already set
inside the dagster-code container). Good answers saved for the schema are shown to the model when a similar
question is asked; `--no-examples` leaves them out."""

import os
import sys
import uuid

from rag_lab.agent import run
from rag_lab.agent.events import Done, SqlChecked
from rag_lab.agent.printer import Printer
from rag_lab.agent.sql import SqlFlow, build_graph, remembered
from rag_lab.config import SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.migrate import apply_migrations
from rag_lab.metrics.store import MetricsStore
from rag_lab.storage.qdrant import QdrantStore


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set")
    return value


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("sql", help="ask questions of the tables in a schema")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model")
    parser.add_argument("--no-think", action="store_true", help="write the SQL without thinking")
    parser.add_argument("--no-examples", action="store_true", help="do not show the model good answers to similar questions")
    parser.set_defaults(main=main)


def main(args) -> None:
    overrides = {
        **({"model": args.model} if args.model else {}),
        **({"think": False} if args.no_think else {}),
        **({"use_examples": False} if args.no_examples else {}),
    }
    cfg = SqlAgentConfig(**overrides)
    base_url = _env("OLLAMA_BASE_URL")
    database_url = _env("METRICS_DATABASE_URL")
    apply_migrations(database_url)
    metrics = MetricsStore(database_url)
    if not metrics.get_db_schema(args.schema):
        sys.exit(f"There is no schema '{args.schema}'. Import a CSV file into one on the Database page.")
    graph = build_graph(metrics, args.schema, base_url, cfg, OllamaEmbedder(base_url), QdrantStore(_env("QDRANT_URL")))
    flow = SqlFlow(args.schema, cfg)

    session_id = uuid.uuid4().hex[:12]
    show = Printer()
    history: list[tuple[str, str]] = []
    question = args.question
    while True:
        if question is None:
            question = input("> ").strip()
            if not question:
                return
        answer, sql = "", ""
        try:
            for event in run(graph, flow, question, history, metrics=metrics, session_id=session_id):
                show(event)
                if isinstance(event, SqlChecked) and event.ok:
                    sql = event.sql
                if isinstance(event, Done):
                    answer = event.answer
        except ValueError as e:  # for example a schema that was deleted meanwhile
            sys.exit(str(e))
        if args.question:
            return
        history += [("User", question), ("Assistant", remembered(answer, sql))]
        question = None
