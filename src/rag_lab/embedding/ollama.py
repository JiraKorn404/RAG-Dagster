import time
from dataclasses import dataclass
from typing import Literal

import httpx

from rag_lab.config import EmbedConfig
from rag_lab.embedding.vectors import truncate_and_normalise


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    wall_ms: float  # includes network and retries
    ollama_ms: float  # Ollama's own total_duration, summed over batches
    load_ms: float  # model load time; large only on a cold start
    prompt_tokens: int
    batches: int


class OllamaEmbedder:
    def __init__(self, base_url: str, timeout: float = 120.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def embed(
        self,
        texts: list[str],
        cfg: EmbedConfig,
        kind: Literal["document", "query"] = "document",
    ) -> EmbedResult:
        """Qwen3 is asymmetric: queries get an instruction prefix, documents are sent as-is."""
        if kind == "query":
            texts = [f"Instruct: {cfg.query_instruction}\nQuery: {t}" for t in texts]

        start = time.perf_counter()
        vectors: list[list[float]] = []
        ollama_ns = load_ns = tokens = batches = 0
        for i in range(0, len(texts), cfg.batch_size):
            data = self._post_embed(texts[i : i + cfg.batch_size], cfg)
            vectors.extend(data["embeddings"])
            ollama_ns += data.get("total_duration", 0)
            load_ns += data.get("load_duration", 0)
            tokens += data.get("prompt_eval_count", 0)
            batches += 1

        return EmbedResult(
            vectors=truncate_and_normalise(vectors, cfg.dimension),
            wall_ms=(time.perf_counter() - start) * 1000,
            ollama_ms=ollama_ns / 1e6,
            load_ms=load_ns / 1e6,
            prompt_tokens=tokens,
            batches=batches,
        )

    def _post_embed(self, batch: list[str], cfg: EmbedConfig) -> dict:
        body = {"model": cfg.model, "input": batch, "keep_alive": cfg.keep_alive}
        for attempt in range(self.retries):
            try:
                resp = httpx.post(f"{self.base_url}/api/embed", json=body, timeout=self.timeout)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError:
                pass
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama embed failed after {self.retries} attempts ({self.base_url})")
