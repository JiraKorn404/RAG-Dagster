"""What every chatbot flow shares: how a node reports itself, and the runner that streams a turn's events
and saves it. A flow is a LangGraph graph plus a small object that says what a saved turn needs from it
(see `Flow`), so the runner does not know what a chunk or a table is."""

import functools
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from langgraph.config import get_stream_writer

from rag_lab.agent.events import (
    AnswerToken,
    Done,
    Event,
    ModelState,
    StepFinished,
    StepStarted,
    Thinking,
    to_dict,
)
from rag_lab.config import ChatModelConfig
from rag_lab.metrics.store import MetricsStore


def step(node):
    """Report when a node starts and how long it took."""

    @functools.wraps(node)
    def reported(state: dict) -> dict:
        emit = get_stream_writer()
        emit(StepStarted(node.__name__))
        start = time.perf_counter()
        update = node(state)
        emit(StepFinished(node.__name__, (time.perf_counter() - start) * 1000))
        return update

    return reported


@dataclass
class Summary:
    """What a saved turn needs from a flow."""

    query: str  # the last thing searched for (documents) or run (database)
    hits: list[dict] = field(default_factory=list)  # for the `hits` column
    answer: str = ""
    thinking: str = ""
    abstained: bool = False
    cited: list[int] = field(default_factory=list)
    unknown_citations: list[int] = field(default_factory=list)
    log: Callable[[MetricsStore], None] | None = None  # extra writes, such as the search log


class Flow(Protocol):
    kind: str  # "documents" or "database": what a chat that uses this flow searches
    config_hash: str | None  # the experiment, for a documents flow
    schema_name: str | None  # the schema, for a database flow
    cfg: ChatModelConfig

    def summarise(self, question: str, events: list[Event], final: dict) -> Summary:
        """What to save for a turn. `final` is the graph's final state; it is empty when the turn
        failed, and then `events` (what happened up to the failure) is all there is."""


def run(
    graph,
    flow: Flow,
    question: str,
    history: list[tuple[str, str]] | None = None,
    *,
    metrics: MetricsStore | None = None,
    session_id: str | None = None,
) -> Iterator[Event]:
    """Answer one question, yielding the events as they happen and `Done` last. With `metrics` and a
    `session_id` the turn is saved to `chat_turns` (a failed turn too, with its error, before the error
    is raised again) and whatever the flow logs besides. A save that fails does not hide the answer:
    `Done.saved` is false and `Done.save_error` says why."""
    cfg = flow.cfg
    start = time.perf_counter()
    timings: dict[str, float] = {}
    states: list[ModelState] = []
    stored: list[Event] = []  # what a saved turn replays from: every event but the token-by-token ones
    step_now = "starting"
    final: dict = {}
    turn_id: int | None = None

    def save(summary: Summary, **extra) -> str | None:
        """Save the turn. Returns why that failed, or None."""
        nonlocal turn_id
        if not (metrics and session_id):
            return None
        try:
            turn_id = metrics.add_chat_turn(
                flow.config_hash,
                flow.schema_name,
                session_id,
                question,
                summary.query,
                cfg.model,
                cfg.think,
                summary.answer,
                summary.thinking,
                summary.hits,
                timings,
                [asdict(s) for s in states],
                kind=flow.kind,
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
            if not isinstance(payload, (Thinking, AnswerToken)):
                stored.append(payload)
            yield payload
    except Exception as e:
        save(flow.summarise(question, stored, {}), error=f"{step_now} failed: {e}")  # the original error is the one to raise
        raise

    summary = flow.summarise(question, stored, final)
    done = Done(
        answer=summary.answer,
        thinking=summary.thinking,
        cited=summary.cited,
        unknown_citations=summary.unknown_citations,
        timings=timings,
        total_ms=(time.perf_counter() - start) * 1000,
        abstained=summary.abstained,
    )
    stored.append(done)
    failure = None
    if metrics and session_id:
        if summary.log:
            try:
                summary.log(metrics)
            except Exception as e:  # noqa: BLE001
                failure = str(e)
        failure = save(summary, abstained=summary.abstained, cited=summary.cited) or failure
    yield replace(done, saved=False, save_error=failure) if failure else replace(done, turn_id=turn_id)
