"""A benchmark report as plain data, built from the saved rows. The Streamlit page and the PDF both
draw this one structure, so the two cannot disagree."""

from dataclasses import dataclass, field

from rag_lab.benchmark.runner import QUALITY_KS


def model_label(model: str) -> str:
    """'qwen3-embedding:4b' -> '4b'."""
    return model.split(":")[-1]


@dataclass
class Chart:
    title: str
    unit: str
    fmt: str  # a format spec such as ".1f"
    values: list[dict]  # experiment, model (label), strategy, value


@dataclass
class Report:
    report_id: str
    document: dict  # name, id, pages, tables, parse_seconds
    status: str
    created: str
    settings: list[tuple[str, str]]  # label, value
    columns: list[str]
    rows: list[list[str]]
    best: list[tuple[int, int]]  # (row, column) of the best value of each column that has a best
    charts: list[Chart]
    queries: dict | None  # per-query ranks: queries, experiments, ranks
    failed: list[tuple[str, str]]
    notes: list[str]
    has_quality: bool = False
    errors: list[str] = field(default_factory=list)


def _get(metrics: dict, *path):
    for key in path:
        if not isinstance(metrics, dict) or metrics.get(key) is None:
            return None
        metrics = metrics[key]
    return metrics


def _fmt(value, spec: str) -> str:
    return "–" if value is None else format(value, spec)


def build_report(report: dict, results: list[dict]) -> Report:
    settings = report["settings"]
    models, strategies = settings["models"], settings["strategies"]
    done = sorted(
        (r for r in results if r["status"] == "done"),
        key=lambda r: (models.index(r["model"]), strategies.index(r["strategy"])),
    )
    top_k = settings["top_k"]
    quality_ks = [k for k in QUALITY_KS if k <= top_k]
    cutoff = 5 if 5 in quality_ks else (quality_ks[-1] if quality_ks else 1)
    has_quality = bool(report["test_queries"]) and any(r["metrics"].get("quality") for r in done)

    # (header, getter, format, better: "high" | "low" | None)
    spec = [
        ("Chunks", lambda m: _get(m, "chunking", "chunks"), "d", None),
        ("Mean tokens", lambda m: _get(m, "chunking", "tokens_mean"), ".0f", None),
        ("Embed tokens/s", lambda m: _get(m, "embedding", "tokens_per_second"), ".0f", "high"),
        ("Search p50 (ms)", lambda m: _get(m, "search", "total_ms", "p50"), ".0f", "low"),
    ]
    if has_quality:
        spec += [
            ("Snippet coverage", lambda m: m.get("snippet_coverage"), ".2f", "high"),
            (f"Recall@{cutoff}", lambda m: _get(m, "quality", f"recall@{cutoff}"), ".3f", "high"),
            ("MRR", lambda m: _get(m, "quality", "mrr"), ".3f", "high"),
            (f"nDCG@{cutoff}", lambda m: _get(m, "quality", f"ndcg@{cutoff}"), ".3f", "high"),
        ]
    columns = ["Model", "Strategy"] + [s[0] for s in spec]
    rows = [
        [model_label(r["model"]), r["strategy"]] + [_fmt(get(r["metrics"]), fmt) for _, get, fmt, _ in spec]
        for r in done
    ]
    best = []
    for c, (_, get, _, better) in enumerate(spec):
        values = [get(r["metrics"]) for r in done]
        present = [v for v in values if v is not None]
        if better and len(present) > 1:
            target = max(present) if better == "high" else min(present)
            best += [(i, c + 2) for i, v in enumerate(values) if v == target]

    def series(title: str, unit: str, fmt: str, get) -> Chart:
        return Chart(
            title,
            unit,
            fmt,
            [
                {
                    "experiment": r["experiment"],
                    "model": model_label(r["model"]),
                    "strategy": r["strategy"],
                    "value": get(r["metrics"]),
                }
                for r in done
                if get(r["metrics"]) is not None
            ],
        )

    charts = [
        series("Embedding speed", "tokens per second", ".0f", lambda m: _get(m, "embedding", "tokens_per_second")),
        series("Search latency, median", "ms", ".1f", lambda m: _get(m, "search", "total_ms", "p50")),
        series("Chunks produced", "chunks", ".0f", lambda m: _get(m, "chunking", "chunks")),
    ]
    if has_quality:
        charts.append(series(f"nDCG@{cutoff}", f"nDCG@{cutoff}", ".3f", lambda m: _get(m, "quality", f"ndcg@{cutoff}")))
        charts.append(series("MRR", "MRR", ".3f", lambda m: _get(m, "quality", "mrr")))

    queries = None
    with_ranks = [r for r in done if r["per_query"]]
    if with_ranks:
        base = with_ranks[0]["per_query"]
        queries = {
            "queries": [{"query": q["query"], "snippet": q["snippet"]} for q in base],
            "experiments": [
                {"name": r["experiment"], "label": f"{model_label(r['model'])} {r['strategy']}"}
                for r in with_ranks
            ],
            "ranks": [[r["per_query"][i]["first_relevant_rank"] for r in with_ranks] for i in range(len(base))],
        }

    info = report["document_info"]
    shown = [
        ("Models", ", ".join(models)),
        ("Strategies", ", ".join(strategies)),
        ("Chunk engine", settings["engine"]),
        ("Max tokens", str(settings["max_tokens"])),
        ("Overlap (fixed only)", str(settings["overlap"])),
        ("Tables", settings["table_handling"]),
        ("Headings in chunk text", "yes" if settings["include_headings"] else "no"),
        ("Top k", str(top_k)),
        ("Probe queries", f"{len(report['probes'])} sentences from the document, {settings['repeats']} timed passes"),
        ("Test queries", str(len(report["test_queries"] or []))),
    ]
    notes = [
        "Similarity scores are on a different scale for each embedding model, so models are compared on speed "
        "and, when test queries were given, on quality only; the report never ranks models by similarity.",
        "Probe queries are sentences taken from the document itself. They measure search speed and how scores "
        "are spread, not how well the right chunk is found.",
    ]
    if has_quality:
        notes += [
            "Quality comes from the test queries: a retrieved chunk is right when its text contains the query's "
            "snippet. Snippet coverage is the share of snippets that appear whole inside some chunk; a snippet "
            "cut by a chunk boundary cannot be found by any model, so coverage caps what a strategy can score.",
            f"Best values are marked for speed and quality columns. {len(report['test_queries'])} test queries "
            "is a small sample: a difference of one query moves recall by "
            f"{1 / len(report['test_queries']):.2f}.",
        ]
    else:
        notes.append("No test queries were given, so there are no retrieval quality metrics in this report.")
    return Report(
        report_id=report["report_id"],
        document={
            "name": report["document_name"],
            "id": report["document_id"],
            "pages": info.get("pages"),
            "tables": info.get("tables"),
            "parse_seconds": info.get("parse_seconds"),
        },
        status=report["status"],
        created=report["created_at"].strftime("%Y-%m-%d %H:%M"),
        settings=shown,
        columns=columns,
        rows=rows,
        best=best,
        charts=charts,
        queries=queries,
        failed=[(r["experiment"], r["error"] or "") for r in results if r["status"] == "failed"],
        notes=notes,
        has_quality=has_quality,
    )
