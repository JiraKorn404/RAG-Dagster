from dataclasses import dataclass

from rag_lab.config import ExperimentConfig
from rag_lab.embedding.ollama import EmbedResult, OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.metrics.timing import timed
from rag_lab.storage.qdrant import QdrantStore


@dataclass
class Hit:
    rank: int
    similarity: float  # Qdrant's cosine score
    distance: float  # 1 - similarity
    text: str
    source_file: str | None
    page: int | None
    modality: str
    headings: list[str]
    doc_id: str
    chunk_id: str


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    embed_ms: float
    search_ms: float
    total_ms: float


def load_experiment(metrics: MetricsStore, name: str) -> tuple[str, ExperimentConfig]:
    """(config_hash, config) recorded for an ingested experiment, so a query is embedded with the
    same model, dimension and instruction as the documents were."""
    found = metrics.get_experiment(name)
    if found is None:
        raise ValueError(f"No experiment named '{name}' in rag_metrics. Ingest a document first.")
    return found[0], ExperimentConfig.model_validate(found[1])


def search(
    query: str,
    config: ExperimentConfig,
    embedder: OllamaEmbedder,
    store: QdrantStore,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    embedded: EmbedResult | None = None,
) -> SearchResult:
    """Embed the query (with the instruction prefix) and search the experiment's collection.
    `filters` match payload fields exactly; only `doc_id`, `modality` and `source_file` have indexes.
    `embedded` is an already embedded query (same model, dimension and instruction as the experiment),
    so several experiments that share an embedding setup can reuse one call; its time counts as embed_ms."""
    reused = embedded is not None
    with timed() as total:
        if embedded is None:
            embedded = embedder.embed([query], config.embed, kind="query")
        with timed() as qdrant_time:
            points = store.query(
                config.collection, embedded.vectors[0], top_k, filters, config.index.hnsw_ef
            )

    hits = [
        Hit(
            rank=rank,
            similarity=p.score,
            distance=1 - p.score,
            text=p.payload["text"],
            source_file=p.payload.get("source_file"),
            page=p.payload.get("page"),
            modality=p.payload["modality"],
            headings=p.payload.get("headings") or [],
            doc_id=p.payload["doc_id"],
            chunk_id=p.payload["chunk_id"],
        )
        for rank, p in enumerate(points, start=1)
    ]
    return SearchResult(
        query=query,
        hits=hits,
        embed_ms=embedded.wall_ms,
        search_ms=qdrant_time.ms,
        total_ms=total.ms + (embedded.wall_ms if reused else 0),
    )
