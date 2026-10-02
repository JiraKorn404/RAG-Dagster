import math

from rag_lab.metrics.retrieval import query_metrics, relevance


def test_retrieval_metrics_against_a_hand_computed_case():
    # Two expected pages. The hits at ranks 1 and 3 are elsewhere; rank 2 is page 4 of a.pdf;
    # rank 4 is page 9 of a.pdf. Rank 5 is a second chunk of page 4, which must not count twice.
    page = lambda p: {"source_file": "a.pdf", "page": p}
    hits = [page(1), page(4), page(2), page(9), page(4)]
    rel = relevance(hits, [page(4), page(9)])
    assert rel == [0, 1, 0, 1, 0]

    m = query_metrics(rel, n=2, ks=[1, 3, 4])
    assert m[("hit_rate", 1)] == 0 and m[("hit_rate", 3)] == 1
    assert m[("recall", 3)] == 0.5 and m[("recall", 4)] == 1.0
    assert math.isclose(m[("precision", 3)], 1 / 3)
    assert m[("mrr", None)] == 0.5
    assert math.isclose(m[("map", None)], (1 / 2 + 2 / 4) / 2)
    # DCG@4 = 1/log2(3) + 1/log2(5); ideal DCG = 1/log2(2) + 1/log2(3)
    assert math.isclose(m[("ndcg", 4)], (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))
    assert math.isclose(m[("ndcg", 4)], 0.6509, abs_tol=1e-4)


def test_contains_matches_text_ignoring_case_and_whitespace():
    hit = {"source_file": "a.pdf", "text": "| Transformer (big)   |   28.4\n| 41.8 |"}
    assert relevance([hit], [{"source_file": "a.pdf", "contains": "TRANSFORMER (big) | 28.4 | 41.8"}]) == [1]
    assert relevance([hit], [{"source_file": "a.pdf", "contains": "ByteNet"}]) == [0]
    assert relevance([hit], [{"source_file": "b.pdf", "contains": "28.4"}]) == [0]
