from dagster import AssetExecutionContext, Failure, MaterializeResult, asset

from rag_lab.assets.chunking import chunks
from rag_lab.assets.partitions import documents_partitions
from rag_lab.ingest import IngestError, embed_chunks, index_chunks
from rag_lab.resources import ExperimentResource, MetricsStoreResource, OllamaResource, QdrantResource


@asset(partitions_def=documents_partitions, deps=[chunks], group_name="ingestion")
def embeddings(
    context: AssetExecutionContext,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    ollama: OllamaResource,
) -> MaterializeResult:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk."""
    try:
        metadata = embed_chunks(
            metrics.store(), ollama.embedder(), experiment.config(), context.partition_key, context.run_id
        )
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=metadata)


@asset(partitions_def=documents_partitions, deps=[embeddings], group_name="ingestion")
def qdrant_index(
    context: AssetExecutionContext,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    qdrant: QdrantResource,
) -> MaterializeResult:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic."""
    try:
        details = index_chunks(
            metrics.store(), qdrant.store(), experiment.config(), context.partition_key, context.run_id
        )
    except IngestError as e:
        raise Failure(str(e)) from e
    return MaterializeResult(metadata=details)
