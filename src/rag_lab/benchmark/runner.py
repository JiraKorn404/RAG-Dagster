"""Search benchmark: a warm-up pass, then `repeats` timed passes over the query set. Plain Python;
the Dagster asset only supplies the clients. Writes benchmark_runs, query_results, benchmark_metrics."""

from statistics import fmean

from rag_lab.config import BenchmarkConfig, ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.retrieval import query_metrics, relevance
from rag_lab.metrics.stats import latency_summary
from rag_lab.metrics.store import MetricsStore
from rag_lab.search import search
from rag_lab.storage.qdrant import QdrantStore

QUALITY_KS = (1, 3, 5, 10)


def _hit_rows(result) -> list[dict]:
    return [
        {
            "rank": h.rank,
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "source_file": h.source_file,
            "page": h.page,
            "modality": h.modality,
            "similarity": h.similarity,
            "distance": h.distance,
        }
        for h in result.hits
    ]


def _quality_rows(queries: list[dict], rels: dict[str, list[int]], ks: list[int]) -> list[dict]:
    """Mean of each quality metric over the queries that have expected results, overall and per
    query modality. Null when no query in the group has any."""
    rows = []
    modalities = sorted({q["modality"] for q in queries if q.get("modality")})
    for modality in [None, *modalities]:
        group = [
            q for q in queries
            if q["expected"] and (modality is None or q.get("modality") == modality)
        ]
        per_query = [query_metrics(rels[q["id"]], len(q["expected"]), ks) for q in group]
        names = [("mrr", None), ("map", None)] + [
            (m, k) for k in ks for m in ("hit_rate", "recall", "precision", "ndcg")
        ]
        for metric, k in names:
            value = fmean(m[(metric, k)] for m in per_query) if per_query else None
            rows.append({"metric": metric, "k": k, "modality": modality, "value": value})
    return rows


def run_benchmark(
    config: ExperimentConfig,
    bench: BenchmarkConfig,
    version: str,
    queries: list[dict],
    embedder: OllamaEmbedder,
    store: QdrantStore,
    metrics: MetricsStore,
    dagster_run_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Returns (benchmark_run_id, the metric rows that were stored)."""
    collection = config.collection
    if not store.client.collection_exists(collection):
        raise ValueError(f"Collection '{collection}' does not exist. Ingest a document first.")
    config_hash = config.config_hash()
    metrics.upsert_experiment(config_hash, config.name, config.model_dump(mode="json"))
    metrics.sync_eval_queries(version, queries)
    run_id = metrics.start_benchmark_run(
        config_hash, bench.top_k, bench.repeats, version, dagster_run_id
    )

    def run_one(q: dict):
        filters = {"modality": q["modality"]} if q.get("modality") else None
        return search(q["query"], config, embedder, store, bench.top_k, filters)

    # Warm-up: the first query pays for the model load and cold caches; it is not stored per query.
    cold_first_ms = None
    for q in queries:
        result = run_one(q)
        cold_first_ms = result.total_ms if cold_first_ms is None else cold_first_ms

    result_rows, embed, search_ms, total = [], [], [], []
    top1, mean_topk, gap = [], [], []
    rels: dict[str, list[int]] = {}
    first_query_warm = []
    for repeat in range(bench.repeats):
        for i, q in enumerate(queries):
            result = run_one(q)
            hits = _hit_rows(result)
            # the stored hit rows carry no text; matching a `contains` snippet needs it
            with_text = [{**row, "text": h.text} for row, h in zip(hits, result.hits)]
            rel = relevance(with_text, q["expected"]) if q["expected"] else None
            result_rows.append({
                "query_id": q["id"],
                "repeat": repeat,
                "embed_ms": result.embed_ms,
                "search_ms": result.search_ms,
                "total_ms": result.total_ms,
                "hits": hits,
                "first_relevant_rank": rel.index(1) + 1 if rel and 1 in rel else None,
            })
            embed.append(result.embed_ms)
            search_ms.append(result.search_ms)
            total.append(result.total_ms)
            if i == 0:
                first_query_warm.append(result.total_ms)
            if repeat == 0:  # the hits are the same every repeat, so score them once
                if rel is not None:
                    rels[q["id"]] = rel
                if hits:
                    sims = [h["similarity"] for h in hits]
                    top1.append(sims[0])
                    mean_topk.append(fmean(sims))
                    gap.append(sims[0] - sims[-1])

    rows: list[dict] = []
    for name, values in (("embed_ms", embed), ("search_ms", search_ms), ("total_ms", total)):
        rows += [{"metric": f"{name}_{stat}", "value": v} for stat, v in latency_summary(values).items()]
    rows += [
        {"metric": "queries_per_second", "value": len(total) / (sum(total) / 1000)},
        {"metric": "first_query_cold_ms", "value": cold_first_ms},
        {"metric": "first_query_warm_ms", "value": fmean(first_query_warm)},
    ]
    if top1:
        rows += [
            {"metric": f"top1_similarity_{stat}", "value": v}
            for stat, v in latency_summary(top1).items()
        ]
        rows += [
            {"metric": "mean_topk_similarity", "k": bench.top_k, "value": fmean(mean_topk)},
            {"metric": "gap_top1_topk", "k": bench.top_k, "value": fmean(gap)},
        ]
    rows += [{"metric": name, "value": value} for name, value in store.stats(collection).items()]
    # Only ks the run actually retrieved: precision@10 of a top-5 list would be misleading.
    rows += _quality_rows(queries, rels, [k for k in QUALITY_KS if k <= bench.top_k])

    metrics.add_query_results(run_id, result_rows)
    metrics.add_benchmark_metrics(run_id, rows)
    metrics.finish_benchmark_run(run_id, store.count(collection))
    return run_id, rows
