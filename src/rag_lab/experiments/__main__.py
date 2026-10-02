"""Run a matrix of experiments over the same documents:

    python -m rag_lab.experiments run experiments/matrix.yaml [--only PATTERN] [--dry-run] [--force]

Run it inside the dagster-code container. Each experiment is ingested (parse, chunk, embed, index) for
every document and then benchmarked, through Dagster, so every run shows up in the Dagster UI.
An experiment that already has a finished benchmark on the current query set is skipped.
"""

import argparse
import fnmatch
import os
import sys
import time

from dagster import DagsterInstance, materialize

from rag_lab.assets.benchmark import search_benchmark
from rag_lab.assets.chunking import chunks
from rag_lab.assets.indexing import embeddings, qdrant_index
from rag_lab.assets.parsing import parsed_document
from rag_lab.benchmark import load_queries
from rag_lab.documents import scan_raw
from rag_lab.experiments import Matrix, expand, load_matrix
from rag_lab.metrics.store import MetricsStore
from rag_lab.resources import (
    ExperimentResource,
    MetricsStoreResource,
    OllamaResource,
    QdrantResource,
)
from rag_lab.storage.qdrant import QdrantStore


def _document_ids(matrix: Matrix) -> dict[str, str]:
    """doc_id -> file name for the matrix's documents."""
    found = scan_raw()
    if matrix.documents == "all":
        return {doc_id: path.name for doc_id, path in found.items()}
    by_name = {path.name: doc_id for doc_id, path in found.items()}
    missing = [name for name in matrix.documents if name not in by_name]
    if missing:
        sys.exit(f"Not in data/raw: {', '.join(missing)}")
    return {by_name[name]: name for name in matrix.documents}


def _run(label: str, assets: list, resources: dict, **kwargs) -> bool:
    result = materialize(
        assets, resources=resources, instance=DagsterInstance.get(), raise_on_error=False, **kwargs
    )
    if not result.success:
        print(f"    FAILED: {label}", flush=True)
    return result.success


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag_lab.experiments")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("matrix")
    parser.add_argument("--only", help="shell-style pattern on experiment names, e.g. 'q3-4b-*'")
    parser.add_argument("--dry-run", action="store_true", help="list the experiments and stop")
    parser.add_argument("--force", action="store_true", help="redo experiments that are already done")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    configs = [c for c in expand(matrix) if not args.only or fnmatch.fnmatch(c.name, args.only)]
    if args.dry_run:
        for c in configs:
            print(f"{c.name:<24} {c.config_hash()}  {c.embed.model}  {c.chunk.strategy}")
        print(f"{len(configs)} experiments")
        return

    docs = _document_ids(matrix)
    version, _ = load_queries(matrix.benchmark.get("queries_file", "eval/queries.yaml"))
    metrics = MetricsStore(os.environ["METRICS_DATABASE_URL"])
    qdrant = QdrantStore(os.environ["QDRANT_URL"])
    shared_resources = {
        "ollama": OllamaResource(base_url=os.environ["OLLAMA_BASE_URL"]),
        "qdrant": QdrantResource(url=os.environ["QDRANT_URL"]),
        "metrics": MetricsStoreResource(database_url=os.environ["METRICS_DATABASE_URL"]),
    }

    failed, started = [], time.perf_counter()
    for n, config in enumerate(configs, start=1):
        print(f"[{n}/{len(configs)}] {config.name}", flush=True)
        config_hash = config.config_hash()
        if not args.force and metrics.has_finished_benchmark(config_hash, version):
            print("    already benchmarked on this query set, skipped", flush=True)
            continue
        resources = {
            **shared_resources,
            "experiment": ExperimentResource(
                name=config.name,
                parse=config.parse,
                chunk=config.chunk,
                embed=config.embed,
                index=config.index,
            ),
        }
        ok = True
        for doc_id, file_name in docs.items():
            # Checked in Qdrant, not in the metrics: the same settings under another experiment
            # name share a config hash, but not a collection.
            if not args.force and qdrant.has_document(config.collection, doc_id):
                continue
            print(f"    ingesting {file_name}", flush=True)
            ok = _run(
                f"{config.name} ingest {file_name}",
                [parsed_document, chunks, embeddings, qdrant_index],
                resources,
                partition_key=doc_id,
            )
            if not ok:
                break
        if ok:
            print("    benchmarking", flush=True)
            ok = _run(
                f"{config.name} benchmark",
                [search_benchmark],
                resources,
                run_config={"ops": {"search_benchmark": {"config": matrix.benchmark}}},
            )
        if not ok:
            failed.append(config.name)

    minutes = (time.perf_counter() - started) / 60
    print(f"Done in {minutes:.1f} min. Failed: {', '.join(failed) if failed else 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
