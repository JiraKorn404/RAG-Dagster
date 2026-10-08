"""On-disk cache of document embeddings, one small file per (model, dimension, text).

Used by semantic chunking, which embeds every sentence group: changing only a breakpoint setting
re-uses the vectors instead of calling Ollama again. Shared across experiments.
"""

import hashlib
import time
from array import array
from pathlib import Path

from rag_lab.config import EmbedConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.paths import DATA_DIR


def _key(cfg: EmbedConfig, text: str) -> str:
    # The text as it is sent, so a vector made with another document template is not reused. With
    # Qwen3's template that is the text itself, which keeps the keys of the files already cached.
    sent = cfg.document_template.format(text=text)
    return hashlib.sha256(f"{cfg.model}|{cfg.dimension}|{sent}".encode()).hexdigest()


def _path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.f32"


def embed_cached(
    texts: list[str], embedder: OllamaEmbedder, cfg: EmbedConfig, root: Path | None = None
) -> tuple[list[list[float]], dict]:
    """Document-mode embeddings for `texts`, in order. Returns the vectors and cache statistics."""
    root = root or DATA_DIR / "artifacts" / "_cache" / "embeddings"
    keys = [_key(cfg, t) for t in texts]
    found: dict[str, list[float]] = {}
    missing: dict[str, str] = {}  # key -> text, de-duplicated
    for key, text in zip(keys, texts):
        if key in found or key in missing:
            continue
        path = _path(root, key)
        if path.exists():
            vec = array("f")
            vec.frombytes(path.read_bytes())
            found[key] = vec.tolist()
        else:
            missing[key] = text

    start = time.perf_counter()
    if missing:
        result = embedder.embed(list(missing.values()), cfg, kind="document")
        for key, vec in zip(missing, result.vectors):
            found[key] = vec
            path = _path(root, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(array("f", vec).tobytes())

    stats = {
        "texts": len(texts),
        "cache_hits": len(texts) - sum(1 for k in keys if k in missing),
        "embedded": len(missing),
        "embed_ms": (time.perf_counter() - start) * 1000 if missing else 0.0,
    }
    return [found[k] for k in keys], stats
