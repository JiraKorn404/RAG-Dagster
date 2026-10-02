import json
import time
from datetime import datetime, timezone

import numpy as np
from dagster import AssetExecutionContext, Failure, MaterializeResult, asset

from rag_lab.assets.chunking import chunks
from rag_lab.assets.partitions import documents_partitions
from rag_lab.chunking.models import read_chunks
from rag_lab.paths import artifacts_dir
from rag_lab.resources import ExperimentResource, MetricsStoreResource, OllamaResource, QdrantResource
from rag_lab.storage.qdrant import to_point_id


@asset(partitions_def=documents_partitions, deps=[chunks], group_name="ingestion")
def embeddings(
    context: AssetExecutionContext,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    ollama: OllamaResource,
) -> MaterializeResult:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk."""
    config = experiment.config()
    doc_id = context.partition_key
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    if not chunk_path.exists():
        raise Failure(f"No chunks at {chunk_path}. Materialise `chunks` first (same experiment name).")

    texts = [c.text for c in read_chunks(chunk_path)]
    result = ollama.embedder().embed(texts, config.embed, kind="document")
    vectors = np.array(result.vectors, dtype=np.float32)
    np.save(artifacts_dir(config.name, "embed") / f"{doc_id}.npy", vectors)

    seconds = result.wall_ms / 1000
    details = {
        "model": config.embed.model,
        "dimension": int(vectors.shape[1]),
        "batch_size": config.embed.batch_size,
        "batches": result.batches,
        "prompt_tokens": result.prompt_tokens,
        "tokens_per_second": round(result.prompt_tokens / seconds, 1) if seconds else None,
        "ollama_ms": round(result.ollama_ms, 1),
        "model_load_ms": round(result.load_ms, 1),
    }
    metrics.store().add_stage_metric(
        config.config_hash(),
        doc_id,
        "embed",
        duration_ms=result.wall_ms,
        items=len(texts),
        throughput=len(texts) / seconds if seconds else None,
        details=details,
        dagster_run_id=context.run_id,
    )
    return MaterializeResult(
        metadata={
            "chunks": len(texts),
            "chunks_per_second": round(len(texts) / seconds, 2) if seconds else 0,
            "embed_seconds": round(seconds, 3),
            **details,
        }
    )


@asset(partitions_def=documents_partitions, deps=[embeddings], group_name="ingestion")
def qdrant_index(
    context: AssetExecutionContext,
    experiment: ExperimentResource,
    metrics: MetricsStoreResource,
    qdrant: QdrantResource,
) -> MaterializeResult:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic."""
    config = experiment.config()
    doc_id = context.partition_key
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    vector_path = artifacts_dir(config.name, "embed") / f"{doc_id}.npy"
    if not (chunk_path.exists() and vector_path.exists()):
        raise Failure("Missing chunks or embeddings. Materialise `chunks` and `embeddings` first.")

    rows = read_chunks(chunk_path)
    vectors = np.load(vector_path)
    if len(rows) != len(vectors):
        raise Failure(
            f"{len(rows)} chunks but {len(vectors)} vectors: the embeddings are stale. "
            "Materialise `embeddings` again."
        )

    # source_file comes from the parse step's metadata file
    meta_path = artifacts_dir(config.name, "parse") / f"{doc_id}.meta.json"
    source_file = json.loads(meta_path.read_text(encoding="utf-8")).get("source_file") if meta_path.exists() else None

    config_hash = config.config_hash()
    ingested_at = datetime.now(timezone.utc).isoformat()
    payloads = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "modality": c.modality,
            "strategy": c.strategy,
            "page": c.page,
            "headings": c.headings,
            "bbox": c.bbox,
            "token_count": c.token_count,
            "experiment": config.name,
            "config_hash": config_hash,
            "source_file": source_file,
            "ingested_at": ingested_at,
        }
        for c in rows
    ]

    store = qdrant.store()
    collection = config.collection
    t0 = time.perf_counter()
    store.ensure_collection(collection, int(vectors.shape[1]), config.index)
    store.delete_document(collection, doc_id)
    t1 = time.perf_counter()
    written = store.upsert(
        collection, [to_point_id(c.chunk_id) for c in rows], vectors.tolist(), payloads
    )
    upsert_seconds = time.perf_counter() - t1
    total = store.count(collection)

    # A failed batch raises out of upsert, so there is no partial-failure count to report.
    details = {
        "collection": collection,
        "points_written": written,
        "batches": -(-written // 256),
        "upsert_ms": round(upsert_seconds * 1000, 1),
        "setup_and_delete_ms": round((t1 - t0) * 1000, 1),
        "collection_points": total,
    }
    metrics.store().add_stage_metric(
        config_hash,
        doc_id,
        "index",
        duration_ms=(time.perf_counter() - t0) * 1000,
        items=written,
        throughput=written / upsert_seconds if upsert_seconds else None,
        details=details,
        dagster_run_id=context.run_id,
    )
    return MaterializeResult(metadata=details)
