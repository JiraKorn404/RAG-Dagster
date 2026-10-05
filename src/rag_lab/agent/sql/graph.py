"""The text-to-SQL agent as a LangGraph graph:

    condense -> schema -> examples -> write_sql -> check -> run_sql -> answer
                                         ^        |
                                         +- repair <-+   (a check or the database refused the query)
    schema too large, "CANNOT", or repairs used up -> abstain

The model is given the whole schema (agent/../sql/catalog.py), writes one query, and a guard and the
database check it before it runs as the read-only role. It thinks while it writes and repairs the SQL and
does not when it rewrites the question or words the answer. Every node reports what it does as events
(agent/events.py); `agent.run()` is the way to use the graph, with a `SqlFlow`."""

from dataclasses import dataclass
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rag_lab.agent.events import (
    ROWS_KEPT,
    AnswerToken,
    Event,
    ExamplesFound,
    Query,
    Repairing,
    SchemaShown,
    SqlChecked,
    SqlRan,
    SqlWritten,
)
from rag_lab.agent.model import call_model, chat_model
from rag_lab.agent.run import Summary, step
from rag_lab.agent.sql.prompts import (
    ANSWER_SYSTEM,
    CANNOT,
    CONDENSE_SYSTEM,
    WRITE_SYSTEM,
    answer_prompt,
    condense_prompt,
    extract_sql,
    not_answered,
    repair_prompt,
    write_prompt,
)
from rag_lab.config import SqlAgentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.sql import catalog, examples as good_answers, execute
from rag_lab.sql.catalog import SchemaText, SchemaTooLarge
from rag_lab.sql.execute import QueryResult, SqlError
from rag_lab.sql.guard import Refused, guard
from rag_lab.storage.qdrant import QdrantStore


class SqlState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]  # ("User" or "Assistant", text); an answer ends with its SQL (`remembered`)
    standalone: str  # the question made standalone; what the SQL and the answer are written for
    schema: SchemaText
    examples: list  # good answers to similar questions (sql/examples.py), when there are any
    stop: str  # a reason to give up at once (the schema is too large, or has no tables)
    sql: str  # the latest query the model wrote, or CANNOT
    to_run: str  # what the guard made of it: what runs
    error: str  # why the latest query failed; empty when it did not
    repairs: int
    result: QueryResult
    answer: str
    thinking: str  # the model's thinking while it wrote and repaired the SQL
    abstained: bool


def remembered(answer: str, sql: str) -> str:
    """An answer as it goes into the history: with the SQL that produced it, so a follow-up can be understood."""
    return f"{answer}\nSQL used:\n{sql}" if sql else answer


def build_graph(
    metrics: MetricsStore,
    schema_name: str,
    base_url: str,
    cfg: SqlAgentConfig | None = None,
    embedder: OllamaEmbedder | None = None,
    store: QdrantStore | None = None,
):
    """`embedder` and `store` are for the good answers: without them, or with `use_examples` off, none are looked for."""
    cfg = cfg or SqlAgentConfig()
    quick_llm = chat_model(base_url, cfg, think=False)  # condense and answer never think
    sql_llm = chat_model(base_url, cfg, think=cfg.think)

    def give_up_or_repair(state: SqlState) -> str:
        return "repair" if state["repairs"] < cfg.max_repairs else "abstain"

    @step
    def condense(state: SqlState) -> dict:
        history = state.get("history", [])[-2 * cfg.history_turns :]
        question = state["question"]
        if history:
            question, _ = call_model(
                quick_llm,
                base_url,
                cfg,
                "condense",
                think=False,
                stream=False,
                messages=[("system", CONDENSE_SYSTEM), ("human", condense_prompt(question, history))],
            )
            question = question.strip() or state["question"]
        get_stream_writer()(Query(question, rewritten=question != state["question"]))
        return {"standalone": question, "repairs": 0, "error": "", "stop": ""}

    @step
    def schema(state: SqlState) -> dict:
        text = catalog.render(metrics, schema_name)
        get_stream_writer()(SchemaShown(text.tables, text.chars, text.text))
        try:
            text.check(cfg.schema_char_budget)
        except SchemaTooLarge as e:
            return {"schema": text, "stop": str(e)}
        if not text.tables:
            return {"schema": text, "stop": f"The schema '{schema_name}' has no tables. Import a CSV file on the Database page."}
        return {"schema": text}

    @step
    def examples(state: SqlState) -> dict:
        """Good answers to questions like this one, when the schema has any: no call is made when it has none."""
        if not (cfg.use_examples and embedder and store and good_answers.has_examples(store, schema_name)):
            return {"examples": []}
        found = good_answers.retrieve(embedder, store, cfg, schema_name, state["standalone"])
        get_stream_writer()(ExamplesFound([e.as_dict() for e in found]))
        return {"examples": found}

    @step
    def write_sql(state: SqlState) -> dict:
        reply, thinking = call_model(
            sql_llm,
            base_url,
            cfg,
            "write_sql",
            think=cfg.think,
            stream=True,
            tokens=False,  # the SQL is not the answer: it is reported as SqlWritten
            messages=[("system", WRITE_SYSTEM), ("human", write_prompt(state["schema"].text, state["standalone"], state.get("examples")))],
        )
        sql = extract_sql(reply)
        get_stream_writer()(SqlWritten(sql, attempt=1))
        return {"sql": sql, "thinking": thinking}

    @step
    def check(state: SqlState) -> dict:
        emit = get_stream_writer()
        try:
            to_run = guard(state["sql"], schema_name, state["schema"].tables, cfg.row_limit + 1)
            execute.explain(schema_name, to_run, cfg.statement_timeout_s)
        except (Refused, SqlError) as e:
            emit(SqlChecked(False, state["sql"], str(e)))
            return {"error": str(e), "to_run": ""}
        emit(SqlChecked(True, to_run))
        return {"error": "", "to_run": to_run}

    @step
    def run_sql(state: SqlState) -> dict:
        try:
            result = execute.run(schema_name, state["to_run"], cfg.row_limit, cfg.statement_timeout_s)
        except SqlError as e:
            return {"error": str(e)}
        get_stream_writer()(SqlRan(result.columns, result.rows[:ROWS_KEPT], result.row_count, result.truncated, result.ms))
        return {"result": result, "error": ""}

    @step
    def repair(state: SqlState) -> dict:
        attempt = state["repairs"] + 1
        get_stream_writer()(Repairing(state["error"], attempt))
        reply, thinking = call_model(
            sql_llm,
            base_url,
            cfg,
            "repair",
            think=cfg.think,
            stream=True,
            tokens=False,
            messages=[
                ("system", WRITE_SYSTEM),
                ("human", repair_prompt(state["schema"].text, state["standalone"], state["sql"], state["error"], state.get("examples"))),
            ],
        )
        sql = extract_sql(reply)
        get_stream_writer()(SqlWritten(sql, attempt=attempt + 1))
        earlier = state.get("thinking", "")
        return {"sql": sql, "repairs": attempt, "thinking": f"{earlier}\n\n{thinking}".strip() if thinking else earlier}

    @step
    def answer(state: SqlState) -> dict:
        text, _ = call_model(
            quick_llm,
            base_url,
            cfg,
            "answer",
            think=False,
            stream=True,
            messages=[
                ("system", ANSWER_SYSTEM),
                ("human", answer_prompt(state["standalone"], state["to_run"], state["result"], cfg.answer_rows)),
            ],
        )
        return {"answer": text}

    @step
    def abstain(state: SqlState) -> dict:
        if state.get("stop"):
            reason, sql = state["stop"], ""
        elif state.get("sql") == CANNOT:
            reason, sql = "The tables do not seem to hold what this question asks for.", ""
        else:
            reason, sql = f"No working query was found after {cfg.max_repairs} repair(s).", state.get("sql", "")
        text = not_answered(reason, sql, "" if state.get("sql") == CANNOT else state.get("error", ""))
        get_stream_writer()(AnswerToken(text))
        return {"answer": text, "abstained": True}

    graph = StateGraph(SqlState)
    for node in (condense, schema, examples, write_sql, check, run_sql, repair, answer, abstain):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "schema")
    graph.add_conditional_edges("schema", lambda s: "abstain" if s.get("stop") else "examples", ["abstain", "examples"])
    graph.add_edge("examples", "write_sql")
    graph.add_conditional_edges("write_sql", lambda s: "abstain" if s["sql"] == CANNOT else "check", ["abstain", "check"])
    graph.add_conditional_edges("check", lambda s: give_up_or_repair(s) if s["error"] else "run_sql", ["repair", "abstain", "run_sql"])
    graph.add_conditional_edges("run_sql", lambda s: give_up_or_repair(s) if s["error"] else "answer", ["repair", "abstain", "answer"])
    graph.add_conditional_edges("repair", lambda s: "abstain" if s["sql"] == CANNOT else "check", ["abstain", "check"])
    graph.add_edge("answer", END)
    graph.add_edge("abstain", END)
    return graph.compile()


@dataclass
class SqlFlow:
    """What a saved database turn needs: the SQL that ran (or the last one tried), the answer, the thinking."""

    schema_name: str
    cfg: SqlAgentConfig
    kind = "database"
    config_hash = None

    def summarise(self, question: str, events: list[Event], final: dict) -> Summary:
        sql = ""
        for event in events:
            if isinstance(event, SqlWritten):
                sql = event.sql
            elif isinstance(event, SqlChecked) and event.ok:
                sql = event.sql
        if not final:  # the turn failed: what was written until then
            return Summary(query=sql or question)
        return Summary(
            query=final.get("to_run") or sql or question,
            answer=final["answer"],
            thinking=final.get("thinking", ""),
            abstained=final.get("abstained", False),
        )
