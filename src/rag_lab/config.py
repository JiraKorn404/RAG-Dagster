"""Experiment configuration. Every stage setting lives here; nothing is hard-coded in the stages."""

import hashlib
import json
from typing import Literal

from dagster import Config
from pydantic import ConfigDict, Field


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


class BenchmarkConfig(_Section):
    """Settings of one search benchmark run. Not part of the experiment, so not in the config hash."""

    top_k: int = 10  # quality metrics are computed for k in 1, 3, 5, 10 up to this value
    repeats: int = 5  # timed passes over the query set, after one warm-up pass
    queries_file: str = "eval/queries.yaml"


class ExperimentConfig(_Section):
    # The name doubles as the Qdrant collection name, so keep it simple.
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    parse: ParseConfig = ParseConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    index: IndexConfig = IndexConfig()

    @property
    def collection(self) -> str:
        return self.name

    def config_hash(self) -> str:
        """Identifies the settings, not the name: the same settings under two names share a hash."""
        data = self.model_dump(mode="json", exclude={"name"})
        for strategy in ("hybrid", "recursive", "semantic"):
            if strategy != self.chunk.strategy:
                data["chunk"].pop(strategy)  # unused strategy settings do not identify the experiment
        payload = json.dumps(data, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
