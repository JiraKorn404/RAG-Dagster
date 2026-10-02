"""Retrieval quality formulas. `rel` is the binary relevance of the ranked hits (1 = relevant) and
`n` the number of expected items for the query."""

import math


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def _matches(hit: dict, item: dict) -> bool:
    """Every field of the expected item must equal the hit's, except `contains`: a snippet that must
    appear in the hit's text (ignoring case and whitespace). The snippet judges a hit regardless of
    how the document was chunked or which page the chunker assigned."""
    for key, value in item.items():
        if key == "contains":
            if _normalise(value) not in _normalise(hit.get("text", "")):
                return False
        elif hit.get(key) != value:
            return False
    return True


def relevance(hits: list[dict], expected: list[dict]) -> list[int]:
    """1 for each hit that matches an expected item, 0 otherwise. An expected item is a dict such as
    {"chunk_id": ...}, {"source_file": ..., "page": ...} or {"source_file": ..., "contains": ...}.
    Each expected item is matched by at most one hit (the best ranked), so two chunks from the same
    expected page cannot push a score above 1."""
    remaining = list(expected)
    rel = []
    for hit in hits:
        match = next((e for e in remaining if _matches(hit, e)), None)
        if match is None:
            rel.append(0)
        else:
            remaining.remove(match)
            rel.append(1)
    return rel


def hit_rate(rel: list[int], k: int) -> float:
    return float(any(rel[:k]))


def recall(rel: list[int], k: int, n: int) -> float:
    return sum(rel[:k]) / n


def precision(rel: list[int], k: int) -> float:
    return sum(rel[:k]) / k


def reciprocal_rank(rel: list[int]) -> float:
    return next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)


def average_precision(rel: list[int], n: int) -> float:
    found, total = 0, 0.0
    for i, r in enumerate(rel):
        if r:
            found += 1
            total += found / (i + 1)
    return total / n


def ndcg(rel: list[int], k: int, n: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n, k)))
    return dcg / ideal


def query_metrics(rel: list[int], n: int, ks: list[int]) -> dict[tuple[str, int | None], float]:
    """All metrics for one query, keyed by (metric, k). MRR and MAP cover the whole retrieved list,
    so their k is None."""
    out: dict[tuple[str, int | None], float] = {
        ("mrr", None): reciprocal_rank(rel),
        ("map", None): average_precision(rel, n),
    }
    for k in ks:
        out[("hit_rate", k)] = hit_rate(rel, k)
        out[("recall", k)] = recall(rel, k, n)
        out[("precision", k)] = precision(rel, k)
        out[("ndcg", k)] = ndcg(rel, k, n)
    return out
