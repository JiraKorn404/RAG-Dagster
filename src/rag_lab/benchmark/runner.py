"""A benchmark report: one document, every chosen embedding model x chunking strategy.

Plain Python (no Dagster, no Streamlit). Each combination becomes an experiment of its own, built
the way the Upload page's Embed button builds one, and is then searched with probe queries (latency
and score spread) and, when test queries were given, scored for retrieval quality. Results are saved
to the database as each experiment finishes, so a report can be read while it runs and survives a
closed browser tab.
"""

import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from docling_core.types.doc import DoclingDocument

from rag_lab.chunking import chunk_document, summarise
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import reference_text, segment
from rag_lab.config import ChunkConfig, EmbedConfig, ExperimentConfig
from rag_lab.embedding.ollama import OllamaEmbedder
from rag_lab.ingest import clean_experiment_name, ingest_document
from rag_lab.library import delete_experiment
from rag_lab.metrics.retrieval import query_metrics, relevance
from rag_lab.metrics.stats import latency_summary
from rag_lab.metrics.store import MetricsStore
from rag_lab.search import search
from rag_lab.storage.qdrant import QdrantStore

QUALITY_KS = (1, 3, 5, 10)


@dataclass
class BenchmarkSettings:
    """What a report varies (models and strategies) and what it holds fixed. Stored with the report."""

    models: list[str]
    strategies: list[str]
    engine: str = "llamaindex"
    max_tokens: int = 512
    overlap: int = 64  # tokens; only the `fixed` strategy uses it
    table_handling: str = "markdown"
    include_headings: bool = True
    top_k: int = 10
    repeats: int = 3  # timed passes over the probe queries, after one warm-up pass
    probe_count: int = 20

    def asdict(self) -> dict:
        return asdict(self)

    @property
    def combinations(self) -> list[tuple[str, str]]:
        """Model by model, so each model is loaded in Ollama once."""
        return [(model, strategy) for model in self.models for strategy in self.strategies]


@dataclass
class TestQuery:
    query: str
    snippet: str  # text that must appear in a chunk for that chunk to count as the right answer


@dataclass
class Document:
    doc_id: str
    name: str
    parse_dir: Path
    meta: dict


@dataclass
class Services:
    metrics: MetricsStore
    embedder: OllamaEmbedder
    qdrant: QdrantStore


def model_slug(model: str) -> str:
    """'qwen3-embedding:0.6b' -> '0-6b'."""
    return re.sub(r"[^a-z0-9]+", "-", model.split(":")[-1].lower()).strip("-")


def experiment_name(document_name: str, report_id: str, model: str, strategy: str) -> str:
    stem = clean_experiment_name(Path(document_name).stem)[:20].strip("-_") or "doc"
    return f"bench-{stem}-{report_id}-{model_slug(model)}-{strategy}"


def experiment_config(settings: BenchmarkSettings, name: str, model: str, strategy: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        tag=name,  # its own hash even when another experiment has the same settings
        embed=EmbedConfig(model=model),
        chunk=ChunkConfig(
            engine=settings.engine,
            strategy=strategy,
            max_tokens=settings.max_tokens,
            overlap=settings.overlap if strategy == "fixed" else 0,
            table_handling=settings.table_handling,
            include_headings_in_text=settings.include_headings,
        ),
    )


def probe_queries(text: str, count: int) -> list[str]:
    """Sentences taken evenly from the document, to measure search speed and score spread. They are
    the same for every experiment, so the numbers compare. Only whole sentences of 8 to 40 words that
    look like prose (no table or markup lines) are used."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    usable = [
        s
        for s in sentences
        if 8 <= len(s.split()) <= 40
        and s[-1] in ".!?"
        and not re.search(r"[|<>]|-->", s)
        and sum(c.isalpha() or c.isspace() for c in s) / len(s) > 0.85
    ]
    if len(usable) <= count:
        return usable
    step = (len(usable) - 1) / (count - 1) if count > 1 else 0
    return [usable[round(i * step)] for i in range(count)]


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def snippet_coverage(chunks: list[Chunk], snippets: list[str]) -> float:
    """The share of snippets that appear whole inside some chunk. A snippet cut by a chunk boundary
    cannot be found by any model, so this caps the quality an experiment can show."""
    texts = [_normalise(c.text) for c in chunks]
    found = sum(any(_normalise(s) in t for t in texts) for s in snippets)
    return found / len(snippets)


def snippets_missing_from(text: str, snippets: list[str]) -> list[str]:
    """Snippets that are not in the document at all (they would silently score zero)."""
    whole = _normalise(text)
    return [s for s in snippets if _normalise(s) not in whole]


def _hit_dicts(result) -> list[dict]:
    return [{"text": h.text, "chunk_id": h.chunk_id, "page": h.page} for h in result.hits]


def _search_metrics(config, settings, probes, services) -> dict:
    """A warm-up pass, then `repeats` timed passes over the probe queries."""
    cold_first_ms = None
    for q in probes:
        r = search(q, config, services.embedder, services.qdrant, settings.top_k)
        cold_first_ms = r.total_ms if cold_first_ms is None else cold_first_ms
    embed, qdrant_ms, total, top1, gap = [], [], [], [], []
    warm_first = []
    for repeat in range(settings.repeats):
        for i, q in enumerate(probes):
            r = search(q, config, services.embedder, services.qdrant, settings.top_k)
            embed.append(r.embed_ms)
            qdrant_ms.append(r.search_ms)
            total.append(r.total_ms)
            if i == 0:
                warm_first.append(r.total_ms)
            if repeat == 0 and r.hits:
                top1.append(r.hits[0].similarity)
                gap.append(r.hits[0].similarity - r.hits[-1].similarity)
    return {
        "queries": len(probes),
        "embed_ms": latency_summary(embed),
        "search_ms": latency_summary(qdrant_ms),
        "total_ms": latency_summary(total),
        "queries_per_second": len(total) / (sum(total) / 1000),
        "first_query_cold_ms": cold_first_ms,
        "first_query_warm_ms": fmean(warm_first),
        "top1_similarity": latency_summary(top1) if top1 else None,
        "gap_top1_topk": fmean(gap) if gap else None,
    }


def _quality(config, settings, test_queries: list[TestQuery], services) -> tuple[dict, list[dict]]:
    """Mean of each quality metric over the test queries, and each query's first relevant rank."""
    ks = [k for k in QUALITY_KS if k <= settings.top_k]
    per_query_metrics, per_query = [], []
    for tq in test_queries:
        result = search(tq.query, config, services.embedder, services.qdrant, settings.top_k)
        rel = relevance(_hit_dicts(result), [{"contains": tq.snippet}])
        per_query_metrics.append(query_metrics(rel, 1, ks))
        per_query.append(
            {
                "query": tq.query,
                "snippet": tq.snippet,
                "first_relevant_rank": rel.index(1) + 1 if 1 in rel else None,
            }
        )
    quality = {
        (f"{metric}@{k}" if k else metric): fmean(m[(metric, k)] for m in per_query_metrics)
        for (metric, k) in per_query_metrics[0]
    }
    return quality, per_query


def run_experiment(
    report_id: str,
    document: Document,
    docling: DoclingDocument,
    settings: BenchmarkSettings,
    model: str,
    strategy: str,
    test_queries: list[TestQuery],
    probes: list[str],
    services: Services,
    on_step=lambda step: None,
) -> tuple[str, dict, list[dict] | None]:
    """Build one experiment and measure it. Returns (experiment name, metrics, per-query results)."""
    name = experiment_name(document.name, report_id, model, strategy)
    config = experiment_config(settings, name, model, strategy)

    on_step("chunking")
    started = time.perf_counter()
    chunks, ctx = chunk_document(docling, document.doc_id, config, services.embedder)
    chunk_seconds = time.perf_counter() - started
    if not chunks:
        raise ValueError("The strategy produced no chunks for this document.")
    summary = summarise(chunks)

    embedded, indexed = ingest_document(
        services.metrics, services.embedder, services.qdrant, config, document.doc_id, document.name,
        document.parse_dir, document.meta, chunks, summary, ctx.stats, ctx.warnings, chunk_seconds,
        on_step=on_step,
    )

    on_step("searching with the probe queries")
    metrics = {
        "chunking": {
            **summary,
            "seconds": chunk_seconds,
            "warnings": ctx.warnings,
            "stats": ctx.stats,
        },
        "embedding": {
            "seconds": embedded["embed_seconds"],
            "chunks_per_second": embedded["chunks_per_second"],
            "tokens_per_second": embedded["tokens_per_second"],
            "prompt_tokens": embedded["prompt_tokens"],
            "model_load_ms": embedded["model_load_ms"],
            "dimension": embedded["dimension"],
        },
        "index": {"upsert_ms": indexed["upsert_ms"], "points": indexed["collection_points"]},
        "search": _search_metrics(config, settings, probes, services),
        "quality": None,
        "snippet_coverage": None,
    }
    per_query = None
    if test_queries:
        on_step("scoring the test queries")
        metrics["snippet_coverage"] = snippet_coverage(chunks, [t.snippet for t in test_queries])
        metrics["quality"], per_query = _quality(config, settings, test_queries, services)
    return name, metrics, per_query


def start_report(
    document: Document,
    settings: BenchmarkSettings,
    test_queries: list[TestQuery],
    metrics: MetricsStore,
) -> tuple[str, list[str]]:
    """Create the report row. Returns its id and the probe queries (taken from the document)."""
    docling = DoclingDocument.load_from_json(document.parse_dir / f"{document.doc_id}.json")
    probes = probe_queries(reference_text(segment(docling)), settings.probe_count)
    report_id = uuid.uuid4().hex[:6]
    info = {
        key: document.meta.get(key)
        for key in ("pages", "tables", "table_cells", "parse_seconds", "chars_per_page", "pages_per_second")
    }
    metrics.create_report(
        report_id,
        document.doc_id,
        document.name,
        settings.asdict(),
        [asdict(t) for t in test_queries] or None,
        probes,
        info,
        len(settings.combinations),
    )
    return report_id, probes


def run_report(
    report_id: str,
    document: Document,
    settings: BenchmarkSettings,
    test_queries: list[TestQuery],
    probes: list[str],
    services: Services,
) -> None:
    """Run every combination, saving each result as it is done. Ends the report as done, stopped
    (a stop was requested between experiments) or failed. Meant to run in a thread."""
    metrics = services.metrics
    try:
        docling = DoclingDocument.load_from_json(document.parse_dir / f"{document.doc_id}.json")
        for model, strategy in settings.combinations:
            if metrics.stop_requested(report_id):
                metrics.finish_report(report_id, "stopped")
                return
            name = experiment_name(document.name, report_id, model, strategy)
            try:
                _, result, per_query = run_experiment(
                    report_id, document, docling, settings, model, strategy, test_queries, probes,
                    services, on_step=lambda step, name=name: _beat(metrics, report_id, name, step),
                )
                metrics.save_result(report_id, name, model, strategy, "done", result, per_query)
            except Exception as e:  # noqa: BLE001  (one experiment failing does not end the report)
                metrics.save_result(
                    report_id, name, model, strategy, "failed", {}, None,
                    error=f"{type(e).__name__}: {e}",
                )
        metrics.finish_report(report_id, "done")
    except Exception:  # noqa: BLE001
        metrics.finish_report(report_id, "failed", error=traceback.format_exc(limit=3))


# What each running report is doing now, for the page to show (this process only).
CURRENT_STEP: dict[str, str] = {}


def _beat(metrics: MetricsStore, report_id: str, name: str, step: str) -> None:
    CURRENT_STEP[report_id] = f"{name}: {step}"
    metrics.touch_report(report_id)



def delete_report_experiments(report_id: str, metrics: MetricsStore, qdrant: QdrantStore) -> list[str]:
    """Remove the experiments a report created: collection, files and database rows. The report and
    its numbers stay. Returns the names removed."""
    removed = []
    for result in metrics.get_results(report_id):
        delete_experiment(result["experiment"], metrics, qdrant)
        removed.append(result["experiment"])
    return removed
