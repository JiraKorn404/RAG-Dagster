from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from rag_lab.assets.indexing import qdrant_index
from rag_lab.benchmark import load_queries, run_benchmark
from rag_lab.config import BenchmarkConfig
from rag_lab.metrics.summary import summary_markdown
from rag_lab.resources import (
    ExperimentResource,
    MetricsStoreResource,
    OllamaResource,
    QdrantResource,
)
from rag_lab.search import load_experiment

# Shown in the Dagster UI; the full set is in benchmark_metrics.
_HEADLINE = (
    "total_ms_p50", "total_ms_p95", "queries_per_second", "first_query_cold_ms",
    "first_query_warm_ms", "top1_similarity_mean", "point_count",
)


@asset(deps=[qdrant_index], group_name="benchmark")
def search_benchmark(
    context: AssetExecutionContext,
    config: BenchmarkConfig,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    ollama: OllamaResource,
    qdrant: QdrantResource,
) -> MaterializeResult:
    """Benchmark search on an ingested experiment: a warm-up pass, then timed repeats of the query
    set in `eval/queries.yaml`. Only the experiment `name` is taken from the run config; the settings
    are those it was ingested with (read from the experiments table, as the search CLI does), so the
    results always line up with its ingestion metrics."""
    store = metrics.store()
    try:
        config_hash, exp_config = load_experiment(store, experiment.name)
        version, queries = load_queries(config.queries_file)
        run_id, rows = run_benchmark(
            exp_config, config, version, queries, ollama.embedder(), qdrant.store(), store, context.run_id
        )
    except (ValueError, FileNotFoundError) as e:
        raise Failure(str(e)) from e

    values = {r["metric"]: r["value"] for r in rows if r.get("k") is None and r.get("modality") is None}
    has_quality = any(r["value"] is not None for r in rows if r["metric"] in ("mrr", "map"))
    return MaterializeResult(
        metadata={
            "experiment": exp_config.name,
            "config_hash": config_hash,
            "benchmark_run_id": run_id,
            "query_set_version": version,
            "queries": len(queries),
            "quality_metrics": "computed" if has_quality else "null: no expected results in the query set",
            **{m: round(values[m], 3) for m in _HEADLINE if values.get(m) is not None},
        }
    )


@asset(deps=[search_benchmark], group_name="benchmark")
def experiment_summary(metrics: MetricsStoreResource) -> MaterializeResult:
    """The experiment_summary view as a Markdown table, one column per experiment so that two
    experiments can be read side by side. The same data: SELECT * FROM experiment_summary."""
    columns, rows = metrics.store().experiment_summary()
    if not rows:
        raise Failure("No experiments in rag_metrics yet.")
    return MaterializeResult(
        metadata={"experiments": len(rows), "summary": MetadataValue.md(summary_markdown(columns, rows))}
    )
