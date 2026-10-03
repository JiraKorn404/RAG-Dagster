"""LlamaIndex's view of our Ollama embedder.

Only LlamaIndex's semantic splitter embeds anything inside LlamaIndex. The stock OllamaEmbedding
would lose Ollama's timings, the Matryoshka truncation and our retries, so this wraps OllamaEmbedder
instead. Vectors go through the on-disk cache, so changing a breakpoint setting does not call Ollama
again.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from rag_lab.config import EmbedConfig
from rag_lab.embedding.cache import embed_cached
from rag_lab.embedding.ollama import OllamaEmbedder


class OllamaLabEmbedding(BaseEmbedding):
    _embedder: OllamaEmbedder = PrivateAttr()
    _cfg: EmbedConfig = PrivateAttr()
    _stats: dict = PrivateAttr()

    def __init__(self, embedder: OllamaEmbedder, cfg: EmbedConfig):
        super().__init__(model_name=cfg.model, embed_batch_size=cfg.batch_size)
        self._embedder = embedder
        self._cfg = cfg
        self._stats = {"texts": 0, "cache_hits": 0, "embedded": 0, "embed_ms": 0.0}

    @classmethod
    def class_name(cls) -> str:
        return "OllamaLabEmbedding"

    @property
    def stats(self) -> dict:
        """Texts asked for, cache hits, texts sent to Ollama and their milliseconds, so far."""
        return dict(self._stats)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors, info = embed_cached(texts, self._embedder, self._cfg)
        for key in self._stats:
            self._stats[key] += info[key]
        return vectors

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed([query], self._cfg, kind="query").vectors[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
