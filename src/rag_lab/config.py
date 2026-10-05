"""Experiment configuration. Every stage setting lives here; nothing is hard-coded in the stages."""

import hashlib
import json
from typing import Literal

from dagster import Config
from pydantic import ConfigDict, Field


NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


# Config is a pydantic model, so these also work as Dagster run config (the launchpad form)
# without a second definition. Stage code only needs the models, not Dagster's runtime.
class _Section(Config):
    model_config = ConfigDict(extra="forbid")  # a typo in an experiment config should fail loudly


class ParseConfig(_Section):
    # OCR and picture options are fixed off in the parser and deliberately not exposed.
    do_table_structure: bool = True
    table_mode: Literal["fast", "accurate"] = "accurate"
    table_cell_matching: bool = True
    do_formula_enrichment: bool = False
    do_code_enrichment: bool = False
    num_threads: int = 8
    document_timeout: float = 600.0


class HybridSettings(_Section):
    merge_peers: bool = True  # merge undersized neighbours that share the same headings


class RecursiveSettings(_Section):
    # Tried in order; a piece is split with the next separator only while it is over the limit.
    separators: list[str] = ["\n\n", "\n", ". ", " "]


class SemanticSettings(_Section):
    buffer_size: int = 1  # sentences on each side that are embedded together with a sentence
    breakpoint_type: Literal["percentile", "stddev", "absolute"] = "percentile"
    # percentile: 0-100 (default 90). stddev: multiples of the standard deviation above the mean
    # (try 1.0). absolute: a cosine distance (try 0.3). The default only suits "percentile".
    breakpoint_threshold: float = 90.0
    min_tokens: int = 64  # smaller chunks are merged into their neighbour
    # Sentence boundary: end punctuation plus whitespace, or a blank line. Not suited to
    # languages without sentence-final punctuation (Thai, for example).
    sentence_pattern: str = r"(?<=[.!?])\s+|\n{2,}"


class ChunkConfig(_Section):
    # Which implementation runs the strategy. `llamaindex` uses LlamaIndex's splitters, which differ
    # from ours in a few settings (see chunking/llamaindex.py); `native` is the original code. A config
    # that does not say is `llamaindex`, so experiments run before the default changed need `native`.
    engine: Literal["native", "llamaindex"] = "llamaindex"
    strategy: Literal["hybrid", "hierarchical", "fixed", "recursive", "semantic"] = "hybrid"
    max_tokens: int = 512
    overlap: int = 0  # tokens; only the `fixed` strategy uses it
    # For hybrid and hierarchical, Docling serialises tables itself: "row-wise" falls back to
    # markdown there, and "skip" drops table-only chunks.
    table_handling: Literal["markdown", "row-wise", "skip"] = "markdown"
    include_headings_in_text: bool = True
    # Settings for the chosen strategy only; the others are ignored (and left out of the hash).
    hybrid: HybridSettings = HybridSettings()
    recursive: RecursiveSettings = RecursiveSettings()
    semantic: SemanticSettings = SemanticSettings()


class EmbedConfig(_Section):
    model: str = "qwen3-embedding:0.6b"
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"  # Hugging Face id; chunk sizes are counted with it
    dimension: int | None = None  # None keeps the model's native size (1024); smaller truncates
    batch_size: int = 32
    query_instruction: str = (
        "Given a question, retrieve relevant passages from the documents that answer it"
    )
    keep_alive: str = "30m"


class IndexConfig(_Section):
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int | None = None  # search-time ef; None uses the Qdrant default
    # Also store a BM25 sparse vector on every point, which the hybrid search methods need. Left out
    # of the config hash while false, so the hashes of experiments made before it existed are unchanged.
    sparse: bool = False


SearchMethod = Literal["dense", "hybrid", "dense+rerank", "hybrid+rerank"]
SEARCH_METHODS: tuple[str, ...] = ("dense", "hybrid", "dense+rerank", "hybrid+rerank")


class SearchConfig(_Section):
    """How an experiment is searched. Not part of ExperimentConfig or its hash: changing how you search
    does not make a new experiment."""

    method: SearchMethod = "dense"
    # What the first stage hands on: each branch of a hybrid search fetches this many hits, and a
    # reranker scores this many. At least top k is always fetched.
    candidates: int = 20
    reranker: str = "dengcao/Qwen3-Reranker-0.6B:Q4_K_M"
    rerank_instruction: str = (
        "Given a question, retrieve relevant passages from the documents that answer it"
    )
    keep_alive: str = "30m"
    # The reranker's context window. Ollama's default (40k) makes the model take about 9 GB of memory,
    # which pushes the embedding model out. A prompt is the instruction, the query and one chunk.
    rerank_num_ctx: int = 2048

    @property
    def hybrid(self) -> bool:
        return self.method.startswith("hybrid")

    @property
    def rerank(self) -> bool:
        return self.method.endswith("+rerank")


class ChatModelConfig(_Section):
    """What every chatbot flow shares: the chat model and how it is called."""

    model: str = "gemma4:e4b-mlx"
    temperature: float = 0.2
    # Ollama's own default window is small and silently cuts a long prompt, so it is always set. The
    # prompt is the instructions, the history, what the flow retrieved and, when `think` is on, the thinking.
    num_ctx: int = 8192
    think: bool = True  # the answer is written with the model's thinking on; the other steps never think
    keep_alive: str = "30m"
    history_turns: int = 6  # earlier question-and-answer pairs the question is condensed with


class AgentConfig(ChatModelConfig):
    """The chatbot agent for documents. Not part of ExperimentConfig or its hash: it changes how an
    experiment is asked, not what is stored in it."""

    top_k: int = 5  # chunks given to the model
    # Judging the retrieval by the best chunk's reranker score (a probability of "yes"): at or above
    # `enough_score` the chunks answer it, below `missing_score` they do not, and in between the model is
    # asked. Chosen on one document: questions it answers scored 0.99 or more, questions it does not
    # cover 0.0 to 0.05 (carburetor icing, which it never mentions, 0.051).
    enough_score: float = 0.5
    missing_score: float = 0.1
    max_rewrites: int = 1  # different queries tried when the chunks do not answer it, before giving up
    # How the documents are searched. The 4B reranker works; the 0.6B builds give every chunk 0.0.
    search: SearchConfig = SearchConfig(method="hybrid+rerank", reranker="dengcao/Qwen3-Reranker-4B:Q8_0")


class SqlAgentConfig(ChatModelConfig):
    """The chatbot agent for a database schema (text-to-SQL). Not part of any experiment hash."""

    # Exact answers want no randomness, and the prompt (the whole schema, the repairs, the rows and, when
    # `think` is on, the thinking) is longer than a documents prompt.
    temperature: float = 0.0
    num_ctx: int = 16384
    # The model is given the description of every table in the schema. A schema whose description is
    # longer than this is refused with its size, not cut: a silently cut schema gives silently wrong SQL.
    schema_char_budget: int = 12_000
    max_repairs: int = 2  # a query the guard or the database refused is rewritten this many times
    row_limit: int = 200  # rows a query may return; more are cut, and the answer says so
    answer_rows: int = 30  # rows given to the model to write the answer from
    statement_timeout_s: float = 10
    # Good answers (rag_lab/sql/examples.py): a question like the one asked, answered correctly before, is
    # shown to the model with its SQL. They are found by the similarity of the questions (cosine, from
    # qwen3-embedding:0.6b). On six saved examples the best match scored 0.56 to 0.78 for a paraphrase or the
    # same kind of question, 0.36 to 0.67 for another kind of question about the same table, and 0.26 to 0.51
    # for an unrelated one; 0.55 is the gap between the last and the first.
    use_examples: bool = True
    examples_k: int = 3
    examples_min_score: float = 0.55
    embed: EmbedConfig = EmbedConfig(
        query_instruction="Given a question about a dataset, retrieve questions that ask for the same kind of information"
    )


class ImportConfig(_Section):
    """Importing a CSV file into the database for imported tables."""

    max_bytes: int = 50 * 1024 * 1024
    max_rows: int = 1_000_000
    sample_rows: int = 10_000  # rows read to guess the column types
    sample_bytes: int = 4 * 1024 * 1024  # the start of the file that is read for that
    preview_rows: int = 20
    sample_values: int = 10  # example values kept for a text column with few distinct values
    max_distinct_for_samples: int = 50


class ExperimentConfig(_Section):
    # The name doubles as the Qdrant collection name, so keep it simple.
    name: str = Field(pattern=NAME_PATTERN)
    parse: ParseConfig = ParseConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    index: IndexConfig = IndexConfig()
    # When set, part of the config hash, so the same settings under another tag are a separate
    # experiment (the Upload page tags the experiments it creates with their name). Empty is left out
    # of the hash, which keeps every hash from before `tag` existed.
    tag: str | None = None

    @property
    def collection(self) -> str:
        return self.name

    def config_hash(self) -> str:
        """Identifies the experiment: its settings and its tag, not its name. Experiments without a tag
        and with the same settings share a hash."""
        return self._hash(with_tag=True)

    def settings_hash(self) -> str:
        """Identifies the settings alone: the same for every name and tag."""
        return self._hash(with_tag=False)

    def _hash(self, with_tag: bool) -> str:
        data = self.model_dump(mode="json", exclude={"name"})
        for strategy in ("hybrid", "recursive", "semantic"):
            if strategy != self.chunk.strategy:
                data["chunk"].pop(strategy)  # unused strategy settings do not identify the experiment
        if self.chunk.engine == "native":
            data["chunk"].pop("engine")  # keeps the hashes of the experiments run before `engine` existed
        if not self.index.sparse:
            data["index"].pop("sparse")  # keeps the hashes of the experiments made before `sparse` existed
        if not (with_tag and self.tag):
            data.pop("tag")
        payload = json.dumps(data, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
