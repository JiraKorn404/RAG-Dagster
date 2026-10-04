"""A benchmark report as plain data, built from the saved rows. The Streamlit page and the PDF both
draw this one structure, so the two cannot disagree."""

from dataclasses import dataclass, field
from statistics import fmean

from rag_lab.benchmark.runner import QUALITY_KS, result_searches
from rag_lab.config import SEARCH_METHODS


def model_label(model: str) -> str:
    """'qwen3-embedding:4b' -> '4b'."""
    return model.split(":")[-1]


@dataclass
class Chart:
    title: str
    unit: str
    fmt: str  # a format spec such as ".1f"
    values: list[dict]  # experiment, model (label), strategy (the x label), value
    # None: one value per experiment. "compare": the search strategies side by side (`strategy` then
    # holds the search strategy). Otherwise the search strategy whose results the chart shows.
    group: str | None = None


@dataclass
class Report:
    report_id: str
    document: dict  # name, id, pages, tables, parse_seconds
    status: str
    created: str
    settings: list[tuple[str, str]]  # label, value
    columns: list[str]
    rows: list[list[str]]  # model, chunking strategy, search strategy, then the numbers
    best: list[tuple[int, int]]  # (row, column) of the best value of each column that has a best
    charts: list[Chart]
    queries: dict | None  # per-query ranks: {"queries": [...], "by_method": {method: {experiments, ranks}}}
    failed: list[tuple[str, str]]
    notes: list[str]
    methods: list[str]  # the search strategies in the report, in display order
    experiments: int  # experiments that finished (a row is one experiment under one search strategy)
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
    wanted = settings.get("search_methods") or ["dense"]  # a report from before search strategies is dense
    methods = [m for m in SEARCH_METHODS if m in wanted]
    done = sorted(
        (r for r in results if r["status"] == "done"),
        key=lambda r: (models.index(r["model"]), strategies.index(r["strategy"])),
    )
    # One item per experiment and search strategy: (result, method, that method's results).
    items = []
    failed = [(r["experiment"], r["error"] or "") for r in results if r["status"] == "failed"]
    for r in done:
        found = result_searches(r["metrics"], r["per_query"])
        for m in methods:
            entry = found.get(m)
            if entry and entry.get("error"):
                failed.append((f"{r['experiment']} ({m})", entry["error"]))
            if entry and entry.get("search"):
                items.append((r, m, entry))
    top_k = settings["top_k"]
    quality_ks = [k for k in QUALITY_KS if k <= top_k]
    cutoff = 5 if 5 in quality_ks else (quality_ks[-1] if quality_ks else 1)
    has_quality = bool(report["test_queries"]) and any(e.get("quality") for _, _, e in items)
    has_rerank = any(m.endswith("+rerank") for _, m, _ in items)

    # (header, level, getter(experiment metrics, search entry), format, better: "high" | "low" | None)
    spec = [
        ("Chunks", "experiment", lambda m, e: _get(m, "chunking", "chunks"), "d", None),
        ("Mean tokens", "experiment", lambda m, e: _get(m, "chunking", "tokens_mean"), ".0f", None),
        ("Embed tokens/s", "experiment", lambda m, e: _get(m, "embedding", "tokens_per_second"), ".0f", "high"),
        ("Search p50 (ms)", "search", lambda m, e: _get(e, "search", "total_ms", "p50"), ".0f", "low"),
    ]
    if has_rerank:
        spec.append(("Rerank p50 (ms)", "search", lambda m, e: _get(e, "search", "rerank_ms", "p50"), ".0f", None))
    if has_quality:
        spec += [
            ("Snippet coverage", "experiment", lambda m, e: m.get("snippet_coverage"), ".2f", "high"),
            (f"Recall@{cutoff}", "search", lambda m, e: _get(e, "quality", f"recall@{cutoff}"), ".3f", "high"),
            ("MRR", "search", lambda m, e: _get(e, "quality", "mrr"), ".3f", "high"),
            (f"nDCG@{cutoff}", "search", lambda m, e: _get(e, "quality", f"ndcg@{cutoff}"), ".3f", "high"),
        ]
    columns = ["Model", "Chunking", "Search"] + [c[0] for c in spec]
    rows = [
        [model_label(r["model"]), r["strategy"], m]
        + [_fmt(get(r["metrics"], e), fmt) for _, _, get, fmt, _ in spec]
        for r, m, e in items
    ]
    # Experiment-level columns repeat on an experiment's rows, so their best is taken over experiments
    # and marked on the experiment's first row only.
    first_row: dict[str, int] = {}
    for i, (r, _, _) in enumerate(items):
        first_row.setdefault(r["experiment"], i)
    best = []
    for c, (_, level, get, _, better) in enumerate(spec):
        indexes = list(first_row.values()) if level == "experiment" else list(range(len(items)))
        values = {i: get(items[i][0]["metrics"], items[i][2]) for i in indexes}
        present = [v for v in values.values() if v is not None]
        if better and len(present) > 1:
            target = max(present) if better == "high" else min(present)
            best += [(i, c + 3) for i, v in values.items() if v == target]

    def chart(title, unit, fmt, get, group=None, source=None):
        source = items if source is None else source
        return Chart(
            title,
            unit,
            fmt,
            [
                {
                    "experiment": r["experiment"],
                    "model": model_label(r["model"]),
                    "strategy": r["strategy"],
                    "value": get(r["metrics"], e),
                }
                for r, m, e in source
                if get(r["metrics"], e) is not None
            ],
            group,
        )

    def compared(title, unit, fmt, get):
        """For each search strategy and model, the mean over the chunking strategies."""
        values = []
        for method in methods:
            for model in models:
                found = [
                    get(r["metrics"], e)
                    for r, m, e in items
                    if m == method and r["model"] == model and get(r["metrics"], e) is not None
                ]
                if found:
                    values.append(
                        {
                            "experiment": f"{model_label(model)}, {method}: mean over the chunking strategies",
                            "model": model_label(model),
                            "strategy": method,
                            "value": fmean(found),
                        }
                    )
        return Chart(title, unit, fmt, values, "compare")

    ndcg = lambda m, e: _get(e, "quality", f"ndcg@{cutoff}")
    mrr = lambda m, e: _get(e, "quality", "mrr")
    latency = lambda m, e: _get(e, "search", "total_ms", "p50")
    one_per_experiment = [items[i] for i in first_row.values()]
    charts = [
        chart("Embedding speed", "tokens per second", ".0f", lambda m, e: _get(m, "embedding", "tokens_per_second"), source=one_per_experiment),
        chart("Chunks produced", "chunks", ".0f", lambda m, e: _get(m, "chunking", "chunks"), source=one_per_experiment),
    ]
    if len(methods) > 1:
        charts.append(compared("Median search latency by search strategy", "ms", ".1f", latency))
        if has_quality:
            charts.append(compared(f"nDCG@{cutoff} by search strategy", f"nDCG@{cutoff}", ".3f", ndcg))
    for method in methods:
        source = [item for item in items if item[1] == method]
        charts.append(chart("Search latency, median", "ms", ".1f", latency, method, source))
        if has_quality:
            charts.append(chart(f"nDCG@{cutoff}", f"nDCG@{cutoff}", ".3f", ndcg, method, source))
            charts.append(chart("MRR", "MRR", ".3f", mrr, method, source))

    queries = None
    by_method = {}
    for method in methods:
        with_ranks = [(r, e) for r, m, e in items if m == method and e.get("per_query")]
        if with_ranks:
            by_method[method] = {
                "experiments": [
                    {"name": r["experiment"], "label": f"{model_label(r['model'])} {r['strategy']}"}
                    for r, _ in with_ranks
                ],
                "ranks": [
                    [e["per_query"][i]["first_relevant_rank"] for _, e in with_ranks]
                    for i in range(len(with_ranks[0][1]["per_query"]))
                ],
            }
    if by_method:
        base = next(e["per_query"] for _, m, e in items if m in by_method and e.get("per_query"))
        queries = {"queries": [{"query": q["query"], "snippet": q["snippet"]} for q in base], "by_method": by_method}

    info = report["document_info"]
    shown = [
        ("Models", ", ".join(models)),
        ("Strategies", ", ".join(strategies)),
        ("Search strategies", ", ".join(methods)),
        ("Chunk engine", settings["engine"]),
        ("Max tokens", str(settings["max_tokens"])),
        ("Overlap (fixed only)", str(settings["overlap"])),
        ("Tables", settings["table_handling"]),
        ("Headings in chunk text", "yes" if settings["include_headings"] else "no"),
        ("Top k", str(top_k)),
        ("Probe queries", f"{len(report['probes'])} sentences from the document, {settings['repeats']} timed passes"),
        ("Test queries", str(len(report["test_queries"] or []))),
    ]
    candidates = settings.get("candidates", 20)
    rerank_probes = settings.get("rerank_probe_count", 5)
    if any(m.startswith("hybrid") for m in methods):
        shown.append(("Hybrid", f"dense plus BM25, fused by reciprocal rank fusion, {candidates} hits from each"))
    if any(m.endswith("+rerank") for m in methods):
        shown += [
            ("Reranker", str(settings.get("reranker"))),
            ("Rerank candidates", str(candidates)),
            ("Rerank probe queries", f"the first {rerank_probes}, one timed pass"),
        ]
    notes = [
        "Similarity scores are on a different scale for each embedding model, so models are compared on speed "
        "and, when test queries were given, on quality only; the report never ranks models by similarity.",
        "Probe queries are sentences taken from the document itself. They measure search speed and how scores "
        "are spread, not how well the right chunk is found.",
    ]
    if methods != ["dense"]:
        notes.append(
            "Search strategies are compared on speed and quality only. Hybrid scores are rank-fusion scores and "
            "reranked scores are the reranker's probability of \"yes\", not cosine similarities, so the spread of "
            "similarities is measured for dense search only. Every search strategy of an experiment searches the "
            "same collection."
        )
    if has_rerank:
        notes.append(
            f"Rerank latency is measured on the first {rerank_probes} probe queries in one pass (each query scores "
            f"{candidates} chunks, one call each), so it rests on fewer measurements than the other strategies."
        )
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
        failed=failed,
        notes=notes,
        methods=methods,
        experiments=len(done),
        has_quality=has_quality,
    )

