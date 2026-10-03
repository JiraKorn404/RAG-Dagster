"""The stage bodies that both the Dagster assets and the Streamlit upload page run. Plain Python, no
Dagster: the assets wrap these and keep only what is Dagster's (partition key, Failure, run id,
MaterializeResult)."""

import dataclasses
import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import ValidationError

from rag_lab.chunking.models import Chunk, read_chunks, write_chunks
from rag_lab.config import NAME_PATTERN, ExperimentConfig, ParseConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.metrics.store import MetricsStore
from rag_lab.parsing.parse import ParsedDocument, parse_pdf
from rag_lab.paths import DATA_DIR, artifacts_dir
from rag_lab.storage.qdrant import QdrantStore, to_point_id

# Below this many extracted characters per page, the PDF is almost certainly scanned images.
SCANNED_PDF_CHARS_PER_PAGE = 50


class IngestError(Exception):
    """A stage cannot run, for example because the previous stage's files are missing."""


def register_experiment(store: MetricsStore, config: ExperimentConfig) -> str:
    """Record the experiment's settings under its name; returns the config hash."""
    config_hash = config.config_hash()
    store.upsert_experiment(config_hash, config.name, config.model_dump(mode="json"))
    return config_hash


def check_new_experiment_name(store: MetricsStore, qdrant: QdrantStore, name: str) -> str | None:
    """Why `name` cannot be used for a new experiment, or None if it can. A name only has to be unused:
    the settings may equal another experiment's, because a new experiment made on the Upload page is
    tagged with its name and so has a hash of its own."""
    if not re.fullmatch(NAME_PATTERN, name):
        return "The name may only contain lower-case letters, digits, - and _, and must start with a letter or digit."
    if store.get_experiment(name) is not None:
        return f"'{name}' is already an experiment. Choose another name, or add the document to it."
    if qdrant.client.collection_exists(name):
        return f"A Qdrant collection called '{name}' already exists. Choose another name."
    return None


def experiments_with_settings(store: MetricsStore, qdrant: QdrantStore, settings_hash: str) -> list[dict]:
    """The experiments a document with these settings can be added to: those whose settings are the
    same, whatever their name and tag. Each row has `name`, `config_hash`, `config`, `documents`,
    `benchmarked` and `points` (None when it has no collection yet).

    An untagged row's stored hash is its settings hash, which also covers rows from before `engine`
    existed (their config has no `engine` key, so rebuilding it would give a different hash). A tagged
    row was made by the Upload page, so its config is complete and its settings hash is recomputed."""
    found = []
    for row in store.list_experiments():
        if row["config"].get("tag"):
            try:
                same = ExperimentConfig.model_validate(row["config"]).settings_hash()
            except ValidationError:
                continue
        else:
            same = row["config_hash"]
        if same == settings_hash:
            exists = qdrant.client.collection_exists(row["name"])
            found.append({**row, "points": qdrant.count(row["name"]) if exists else None})
    return found


def save_upload(name: str, content: bytes) -> tuple[str, Path]:
    """Save an uploaded file as data/uploads/<doc_id>/<name> (not data/raw, so the sensor does not
    register it). The document id is the content hash, as everywhere else."""
    doc_id = hashlib.sha256(content).hexdigest()[:16]
    path = DATA_DIR / "uploads" / doc_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return doc_id, path


def ensure_parsed(path: Path, doc_id: str) -> tuple[Path, dict]:
    """The Docling parse of an uploaded document with default settings, made once and kept in
    data/artifacts/_uploads/<doc_id>/. Returns that folder and the parse metadata."""
    parse_dir = artifacts_dir("_uploads", doc_id)
    meta_path = parse_dir / f"{doc_id}.meta.json"
    if not (parse_dir / f"{doc_id}.json").exists() or not meta_path.exists():
        parse_pdf(path, doc_id, ParseConfig(), parse_dir)
    return parse_dir, json.loads(meta_path.read_text(encoding="utf-8"))


def ingest_document(
    metrics: MetricsStore,
    embedder: OllamaEmbedder,
    qdrant: QdrantStore,
    config: ExperimentConfig,
    doc_id: str,
    source_name: str,
    parse_dir: Path,
    meta: dict,
    chunks: list[Chunk],
    summary: dict,
    stats: dict,
    warnings: list[str],
    seconds: float,
    register: bool = True,
    on_step: Callable[[str], None] = lambda step: None,
) -> tuple[dict, dict]:
    """Put an already chunked, already parsed document into an experiment: the parse and chunk files
    under data/artifacts/<experiment>/, the experiment row (when `register`) and the parse and chunk
    metric rows, then embed and index. Returns the embed and index numbers. `on_step` is told what
    is about to happen, so a caller can show it or say which step failed."""
    name = config.name
    on_step("copying the parse files")
    target = artifacts_dir(name, "parse")
    for suffix in (".json", ".md"):
        shutil.copy(parse_dir / f"{doc_id}{suffix}", target / f"{doc_id}{suffix}")
    stored = {**meta, "source_file": source_name}
    (target / f"{doc_id}.meta.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")

    on_step("recording the experiment, the parse and the chunks")
    write_chunks(chunks, artifacts_dir(name, "chunk") / f"{doc_id}.chunks.jsonl")
    if register:
        register_experiment(metrics, config)
    fields = {f.name for f in dataclasses.fields(ParsedDocument)}
    record_parse(metrics, config, doc_id, ParsedDocument(**{k: v for k, v in stored.items() if k in fields}))
    record_chunk(metrics, config, doc_id, summary, stats, warnings, seconds)

    on_step("embedding the chunks")
    embedded = embed_chunks(metrics, embedder, config, doc_id)
    on_step("writing the points to Qdrant")
    indexed = index_chunks(metrics, qdrant, config, doc_id)
    return embedded, indexed


def record_parse(
    store: MetricsStore,
    config: ExperimentConfig,
    doc_id: str,
    parsed: ParsedDocument,
    run_id: str | None = None,
) -> None:
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "parse",
        duration_ms=parsed.parse_seconds * 1000,
        items=parsed.pages,
        throughput=parsed.pages_per_second,
        details={
            "source_file": parsed.source_file,
            "status": parsed.status,
            "tables": parsed.tables,
            "table_cells": parsed.table_cells,
            "text_items": parsed.text_items,
            "chars_per_page": parsed.chars_per_page,
            "model_load_seconds": parsed.model_load_seconds,
        },
        dagster_run_id=run_id,
    )


def record_chunk(
    store: MetricsStore,
    config: ExperimentConfig,
    doc_id: str,
    summary: dict,
    stats: dict,
    warnings: list[str],
    seconds: float,
    run_id: str | None = None,
) -> None:
    """`summary` is chunking.summarise(chunks); `stats` and `warnings` come from the chunk context."""
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "chunk",
        duration_ms=seconds * 1000,
        items=summary["chunks"],
        throughput=summary["chunks"] / seconds if seconds else None,
        details={
            "strategy": config.chunk.strategy,
            "engine": config.chunk.engine,
            **summary,
            **stats,
            "warnings": warnings,
        },
        dagster_run_id=run_id,
    )


def embed_chunks(
    store: MetricsStore,
    embedder: OllamaEmbedder,
    config: ExperimentConfig,
    doc_id: str,
    run_id: str | None = None,
) -> dict:
    """Embed every chunk of the document (document mode, no query prefix) with Ollama.
    Reads chunk/<doc_id>.chunks.jsonl and writes embed/<doc_id>.npy: one float32 row per chunk.
    Returns the numbers the asset shows as metadata."""
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    if not chunk_path.exists():
        raise IngestError(f"No chunks at {chunk_path}. Materialise `chunks` first (same experiment name).")

    texts = [c.text for c in read_chunks(chunk_path)]
    result = embedder.embed(texts, config.embed, kind="document")
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
    store.add_stage_metric(
        config.config_hash(),
        doc_id,
        "embed",
        duration_ms=result.wall_ms,
        items=len(texts),
        throughput=len(texts) / seconds if seconds else None,
        details=details,
        dagster_run_id=run_id,
    )
    return {
        "chunks": len(texts),
        "chunks_per_second": round(len(texts) / seconds, 2) if seconds else 0,
        "embed_seconds": round(seconds, 3),
        **details,
    }


def index_chunks(
    store: MetricsStore,
    qdrant: QdrantStore,
    config: ExperimentConfig,
    doc_id: str,
    run_id: str | None = None,
) -> dict:
    """Write the document's chunks and vectors to the experiment's Qdrant collection.
    Idempotent: the document's existing points are deleted first and point ids are deterministic.
    Returns the details the asset shows as metadata."""
    chunk_path = artifacts_dir(config.name, "chunk") / f"{doc_id}.chunks.jsonl"
    vector_path = artifacts_dir(config.name, "embed") / f"{doc_id}.npy"
    if not (chunk_path.exists() and vector_path.exists()):
        raise IngestError("Missing chunks or embeddings. Materialise `chunks` and `embeddings` first.")

    rows = read_chunks(chunk_path)
    vectors = np.load(vector_path)
    if len(rows) != len(vectors):
        raise IngestError(
            f"{len(rows)} chunks but {len(vectors)} vectors: the embeddings are stale. "
            "Materialise `embeddings` again."
        )

    # source_file comes from the parse step's metadata file
    meta_path = artifacts_dir(config.name, "parse") / f"{doc_id}.meta.json"
    source_file = (
        json.loads(meta_path.read_text(encoding="utf-8")).get("source_file") if meta_path.exists() else None
    )

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

    collection = config.collection
    t0 = time.perf_counter()
    qdrant.ensure_collection(collection, int(vectors.shape[1]), config.index)
    qdrant.delete_document(collection, doc_id)
    # The collection exists now, so LlamaIndex's store reuses its (unnamed) vector instead of creating
    # its own. It writes each node's metadata flat into the payload and the Document id as `doc_id`.
    vector_store = QdrantVectorStore(collection_name=collection, client=qdrant.client, batch_size=256)
    nodes = []
    for chunk, vector, payload in zip(rows, vectors, payloads):
        metadata = {k: v for k, v in payload.items() if k != "doc_id"}
        nodes.append(
            TextNode(
                id_=to_point_id(chunk.chunk_id),
                text=chunk.text,
                embedding=vector.tolist(),
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk.doc_id)},
            )
        )
    t1 = time.perf_counter()
    written = len(vector_store.add(nodes))
    upsert_seconds = time.perf_counter() - t1
    total = qdrant.count(collection)

    # A failed batch raises out of add, so there is no partial-failure count to report.
    details = {
        "collection": collection,
        "points_written": written,
        "batches": -(-written // 256),
        "upsert_ms": round(upsert_seconds * 1000, 1),
        "setup_and_delete_ms": round((t1 - t0) * 1000, 1),
        "collection_points": total,
    }
    store.add_stage_metric(
        config_hash,
        doc_id,
        "index",
        duration_ms=(time.perf_counter() - t0) * 1000,
        items=written,
        throughput=written / upsert_seconds if upsert_seconds else None,
        details=details,
        dagster_run_id=run_id,
    )
    return details


def clean_experiment_name(text: str) -> str:
    """A string turned into something that matches the experiment name pattern."""
    name = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-_")
    return name[:60] or "upload"
