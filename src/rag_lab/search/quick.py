"""One-call helper for notebooks: `ask("query", "experiment")`. It runs on the host, not in a
container."""

import os
from pathlib import Path

from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.search.engine import SearchResult, load_experiment, search
from rag_lab.storage.qdrant import QdrantStore

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _read_env_file() -> dict[str, str]:
    values = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def _setting(name: str, file_values: dict[str, str]) -> str | None:
    return os.environ.get(name) or file_values.get(name)


def _metrics_store(env: dict[str, str]) -> MetricsStore:
    return MetricsStore(
        _setting("METRICS_DATABASE_URL", env)
        or f"postgresql://{_setting('POSTGRES_USER', env)}:{_setting('POSTGRES_PASSWORD', env)}"
        "@127.0.0.1:5432/rag_metrics"
    )


def ask(
    query: str,
    experiment: str,
    top_k: int = 5,
    modality: str | None = None,
    log: bool = False,
    chars: int | None = 300,
) -> SearchResult:
    """Search an ingested experiment, print each hit (first `chars` characters of its text; None for
    all of it) and the timings, and return the result."""
    env = _read_env_file()
    ollama_url = _setting("OLLAMA_BASE_URL", env)
    if not ollama_url:
        raise RuntimeError("OLLAMA_BASE_URL is not set (environment or .env)")
    # .env holds the address containers use; on the host that name does not resolve.
    ollama_url = ollama_url.replace("host.docker.internal", "localhost")

    metrics = _metrics_store(env)
    config_hash, config = load_experiment(metrics, experiment)
    result = search(
        query,
        config,
        OllamaEmbedder(ollama_url),
        QdrantStore(_setting("QDRANT_URL", env) or "http://127.0.0.1:6333"),
        top_k=top_k,
        filters={"modality": modality} if modality else None,
    )
    if log:
        metrics.add_search_log(
            config_hash,
            query,
            top_k,
            result.embed_ms,
            result.search_ms,
            result.total_ms,
            result.hits[0].similarity if result.hits else None,
        )

    for h in result.hits:
        where = f"{h.source_file or h.doc_id} p.{h.page}" if h.page else (h.source_file or h.doc_id)
        text = " ".join(h.text.split())
        shown = text if chars is None or len(text) <= chars else text[:chars] + "..."
        print(f"{h.rank}. {h.similarity:.3f} [{h.modality}] {where}\n   {shown}")
    if not result.hits:
        print("No hits.")
    print(
        f"embed {result.embed_ms:.0f} ms, search {result.search_ms:.0f} ms, "
        f"total {result.total_ms:.0f} ms"
    )
    return result
