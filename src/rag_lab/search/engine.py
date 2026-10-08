from dataclasses import dataclass

from rag_lab.config import ExperimentConfig, SearchConfig
from rag_lab.embedding.ollama import EmbedResult, OllamaEmbedder
from rag_lab.embedding.sparse import query_vector
from rag_lab.metrics.store import MetricsStore
from rag_lab.metrics.timing import timed
from rag_lab.reranking import OllamaReranker
from rag_lab.storage.qdrant import QdrantStore


@dataclass
class Hit:
    rank: int
    # The score of the method that ranked the hit: Qdrant's cosine similarity for `dense`, a rank-fusion
    # score for `hybrid`, the reranker's probability of "yes" for the rerank methods. Only the cosine
    # score is comparable between experiments, and `distance` means something only for it.
    similarity: float
    distance: float  # 1 - similarity
    text: str
    source_file: str | None
    page: int | None
    modality: str
    headings: list[str]
    doc_id: str
    chunk_id: str
    # A picture chunk's file, relative to data/artifacts: "<experiment>/parse/<doc_id>.pictures/<n>.png"
    image: str | None = None


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    embed_ms: float
    search_ms: float  # Qdrant, including the BM25 branch and the fusion of a hybrid search
    total_ms: float
    method: str = "dense"
    rerank_ms: float = 0.0


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
    options: SearchConfig | None = None,
    reranker: OllamaReranker | None = None,
) -> SearchResult:
    """Embed the query (with the instruction prefix) and search the experiment's collection.
    `filters` match payload fields exactly; only `doc_id`, `modality` and `source_file` have indexes.
    `embedded` is an already embedded query (same model, dimension and instruction as the experiment),
    so several experiments that share an embedding setup can reuse one call; its time counts as embed_ms.
    `options` picks the search method (dense when not given): `hybrid` needs an experiment made with
    `index.sparse`, and the rerank methods need a `reranker`. A rerank method fetches `options.candidates`
    hits (at least top k) and returns the best top k by the reranker's score."""
    options = options or SearchConfig()
    if options.rerank and reranker is None:
        raise ValueError(f"The search method '{options.method}' needs a reranker.")
    reused = embedded is not None
    first_stage = max(options.candidates, top_k) if options.rerank else top_k
    with timed() as total:
        if embedded is None:
            embedded = embedder.embed([query], config.embed, kind="query")
        with timed() as qdrant_time:
            try:
                points = store.query(
                    config.collection,
                    embedded.vectors[0],
                    first_stage,
                    filters,
                    config.index.hnsw_ef,
                    sparse=query_vector(query) if options.hybrid else None,
                    branch_limit=max(options.candidates, first_stage),
                )
            except Exception as e:  # Qdrant names the missing vector in its error
                if options.hybrid and "bm25" in str(e):
                    raise ValueError(
                        f"Experiment '{config.name}' has no BM25 vector, so it cannot be searched with "
                        f"'{options.method}'. It was made without `index.sparse`."
                    ) from e
                raise
        rerank_ms = 0.0
        scores = [p.score for p in points]
        if options.rerank and points:
            ranked = reranker.score(query, [p.payload["text"] for p in points], options)
            rerank_ms = ranked.wall_ms
            order = sorted(range(len(points)), key=lambda i: -ranked.scores[i])[:top_k]
            points = [points[i] for i in order]
            scores = [ranked.scores[i] for i in order]

    hits = [
        Hit(
            rank=rank,
            similarity=score,
            distance=1 - score,
            text=p.payload["text"],
            source_file=p.payload.get("source_file"),
            page=p.payload.get("page"),
            modality=p.payload["modality"],
            headings=p.payload.get("headings") or [],
            doc_id=p.payload["doc_id"],
            chunk_id=p.payload["chunk_id"],
            image=f"{config.name}/parse/{p.payload['image']}" if p.payload.get("image") else None,
        )
        for rank, (p, score) in enumerate(zip(points, scores), start=1)
    ]
    return SearchResult(
        query=query,
        hits=hits,
        embed_ms=embedded.wall_ms,
        search_ms=qdrant_time.ms,
        total_ms=total.ms + (embedded.wall_ms if reused else 0),
        method=options.method,
        rerank_ms=rerank_ms,
    )
