"""What the agent reports while it works. A page or a CLI draws these without knowing the graph."""

import re
from dataclasses import dataclass, field

from rag_lab.search import Hit


@dataclass
class StepStarted:
    node: str


@dataclass
class StepFinished:
    node: str
    ms: float


@dataclass
class Query:
    text: str
    rewritten: bool  # false when the question was already standalone


@dataclass
class Graded:
    """Whether the retrieved chunks answer the question, and how that was decided."""

    enough: bool
    best_score: float  # the best chunk's reranker score
    by: str  # "score" or "model" (the score was in between, so the model was asked)


@dataclass
class Rewrote:
    query: str  # the different query tried next
    previous: str


@dataclass
class Retrieved:
    hits: list[Hit]
    method: str
    candidates: int
    embed_ms: float
    search_ms: float
    rerank_ms: float


@dataclass
class Thinking:
    text: str  # a piece of the model's thinking


@dataclass
class AnswerToken:
    text: str  # a piece of the answer


@dataclass
class ModelState:
    """One model call: what ran, how full its context was and how fast it generated."""

    node: str
    model: str
    think: bool
    loaded: bool | None  # was the model already in memory before the call; None when Ollama could not say
    prompt_tokens: int
    output_tokens: int  # includes the thinking
    num_ctx: int
    tokens_per_s: float | None
    context_full: bool  # the prompt reached num_ctx, so Ollama cut it


@dataclass
class Done:
    answer: str
    thinking: str
    cited: list[int]  # the passage numbers the answer cites that exist
    unknown_citations: list[int]  # cited numbers that no passage has
    timings: dict[str, float] = field(default_factory=dict)  # ms per step, summed over retries
    total_ms: float = 0.0
    abstained: bool = False  # the chunks did not answer it, so the answer is the fixed "not found" message


Event = (
    StepStarted | StepFinished | Query | Graded | Rewrote | Retrieved | Thinking | AnswerToken | ModelState | Done
)


def citations(answer: str, passages: int) -> tuple[list[int], list[int]]:
    """The passage numbers an answer cites as [n]: (those that exist, those that do not)."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [n for n in numbers if 1 <= n <= passages], [n for n in numbers if not 1 <= n <= passages]
