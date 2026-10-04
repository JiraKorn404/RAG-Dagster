"""BM25 sparse vectors, for the hybrid search methods. Plain Python, no model.

A term is a lower-cased `\\w+` token, hashed to a stable index. A document's weight for a term is
BM25's term-frequency part; the IDF part is applied by Qdrant (the `bm25` vector is created with the
IDF modifier), and a query's weight for a term is 1. Meant for languages that put spaces between
words: other languages need a word segmenter first."""

import re
import zlib
from collections import Counter

K1 = 1.2
B = 0.75

SparseVector = tuple[list[int], list[float]]  # indices, values


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8"))  # stable between runs, unlike hash(); fits Qdrant's u32


def document_vectors(texts: list[str]) -> list[SparseVector]:
    """One vector per text. Length normalisation uses the mean length of these texts."""
    tokenised = [tokens(t) for t in texts]
    avg_len = (sum(len(t) for t in tokenised) / len(tokenised)) if tokenised else 0.0
    vectors = []
    for terms in tokenised:
        weights: dict[int, float] = {}
        norm = K1 * (1 - B + B * len(terms) / avg_len) if avg_len else K1
        for term, tf in Counter(terms).items():
            index = _index(term)
            weights[index] = weights.get(index, 0.0) + tf * (K1 + 1) / (tf + norm)
        vectors.append((list(weights), list(weights.values())))
    return vectors


def query_vector(text: str) -> SparseVector:
    indices = sorted({_index(t) for t in tokens(text)})
    return indices, [1.0] * len(indices)
