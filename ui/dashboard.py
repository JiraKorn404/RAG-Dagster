import charts
import data
import pandas as pd
import streamlit as st
import style

style.hero("Experiment dashboard", "Embedding model × chunking strategy, read live from rag_metrics")

versions = data.versions()
if versions.empty:
    st.info(
        "No finished benchmarks yet. Run the matrix "
        "(`python -m rag_lab.experiments run experiments/matrix.yaml`) or the `search_benchmark` asset."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Filters")
    version = st.selectbox(
        "Query set",
        versions["query_set_version"],
        format_func=lambda v: (
            f"{v} · {int(versions.set_index('query_set_version').loc[v, 'experiments'])} experiments"
        ),
        help="Experiments are only comparable when they ran the same query set.",
    )
    scope = st.radio("Queries", ["All queries", "Table queries"], help="Quality metrics are averaged over these.")
    k = st.select_slider("Cut-off k", options=[1, 3, 5, 10], value=5)
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

modality = "table" if scope == "Table queries" else None
metrics = data.benchmark_metrics(version)
quality = data.pick(metrics, "ndcg", k, modality)

# --- headline cards ---------------------------------------------------------------------------
cards = st.columns(4)
if quality.empty:
    for col, label in zip(cards[:3], ["Best model", "Best strategy", "Best experiment"]):
        col.markdown(style.card(label, "–", "no expected results in this query set"), unsafe_allow_html=True)
else:
    by_model = quality.groupby("model")["value"].mean()
    by_strategy = quality.groupby("strategy")["value"].mean()
    best = quality.loc[quality["value"].idxmax()]
    cards[0].markdown(
        style.card("Best model", by_model.idxmax(), f"mean nDCG@{k} {by_model.max():.3f}", accent=True),
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        style.card("Best strategy", by_strategy.idxmax(), f"mean nDCG@{k} {by_strategy.max():.3f}", accent=True),
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        style.card("Best experiment", best["name"], f"nDCG@{k} {best['value']:.3f}"), unsafe_allow_html=True
    )
speed = data.pick(metrics, "total_ms_p50")
if not speed.empty:
    fastest = speed.loc[speed["value"].idxmin()]
    cards[3].markdown(
        style.card("Fastest search", f"{fastest['value']:.0f} ms", f"median · {fastest['name']}"),
        unsafe_allow_html=True,
    )
st.write("")

tab_quality, tab_speed, tab_ingest, tab_queries, tab_data = st.tabs(
    ["Quality", "Speed", "Ingestion", "Queries", "Data"]
)

# --- quality ----------------------------------------------------------------------------------
QUALITY_METRICS = {
    f"nDCG@{k}": ("ndcg", k),
    f"Recall@{k}": ("recall", k),
    f"Precision@{k}": ("precision", k),
    f"Hit rate@{k}": ("hit_rate", k),
    "MRR": ("mrr", None),
    "MAP": ("map", None),
}
with tab_quality:
    label = st.selectbox("Metric", list(QUALITY_METRICS), label_visibility="collapsed")
    metric, metric_k = QUALITY_METRICS[label]
    values = data.pick(metrics, metric, metric_k, modality)
    if values.empty:
        st.info("This query set has no expected results, so quality metrics are empty. Fill in `expected` in `eval/queries.yaml`.")
    else:
        left, right = st.columns(2)
        with left, st.container(border=True):
            style.section(label, "Strategy by model. Brighter is higher.")
            charts.heatmap(values)
        with right, st.container(border=True):
            style.section(label, "The same values by strategy.")
            charts.grouped_bars(values, label)
        st.caption(
            "Models are compared on quality metrics only: similarity scores are on a different scale for each model."
        )

# --- speed ------------------------------------------------------------------------------------
with tab_speed:
    panels = [
        ("Search latency, median", "total_ms_p50", "ms"),
        ("Search latency, 95th percentile", "total_ms_p95", "ms"),
        ("Query embedding time, median", "embed_ms_p50", "ms"),
    ]
    for col, (title, metric, unit) in zip(st.columns(3), panels):
        values = data.pick(metrics, metric)
        with col, st.container(border=True):
            style.section(title, "Embedding the query plus the Qdrant search.")
            if values.empty:
                st.caption("No data.")
            else:
                charts.grouped_bars(values, unit, fmt=".1f", height=260)

# --- ingestion --------------------------------------------------------------------------------
with tab_ingest:
    ingest = data.summary()
    panels = [
        ("Seconds per document", "seconds_per_document", "seconds"),
        ("Embedding speed", "embed_tokens_per_second", "tokens / second"),
        ("Chunks per document", "chunks_per_document", "chunks"),
    ]
    for col, (title, column, unit) in zip(st.columns(3), panels):
        frame = ingest.dropna(subset=[column]).rename(columns={column: "value"})
        frame["value"] = frame["value"].astype(float)
        with col, st.container(border=True):
            style.section(title, "Latest run of each stage.")
            if frame.empty:
                st.caption("No data.")
            else:
                charts.grouped_bars(frame, unit, fmt=".1f", height=260)

# --- per-query --------------------------------------------------------------------------------
with tab_queries:
    ranks = data.first_ranks(version)
    if ranks.empty:
        st.info("No queries with expected results in this query set.")
    else:
        order_df = ranks.drop_duplicates("name")
        experiments = [
            n
            for model in data.model_order(order_df["model"])
            for strategy in data.strategy_order(order_df["strategy"])
            for n in order_df[(order_df["model"] == model) & (order_df["strategy"] == strategy)]["name"]
        ]
        recip = ranks.assign(s=ranks["rank"].map(lambda r: 0.0 if pd.isna(r) else 1.0 / r))
        queries = list(recip.groupby("query_id")["s"].mean().sort_values().index)  # hardest first
        with st.container(border=True):
            style.section(
                "Where each experiment finds the answer",
                "Brighter: the first relevant hit is near rank 1. Grey: not found in the top results. Hardest queries on top.",
            )
            charts.query_grid(ranks, experiments, queries, height=max(260, 22 * len(queries)))

        style.section("Compare two experiments")
        a_col, b_col = st.columns(2)
        a = a_col.selectbox("Experiment A", experiments, index=0)
        b = b_col.selectbox("Experiment B", experiments, index=min(1, len(experiments) - 1))
        wide = (
            ranks[ranks["name"].isin([a, b])]
            .pivot(index=["query_id", "query_text"], columns="name", values="rank")
            .reset_index()
        )
        if a != b:
            rank_a = wide[a].fillna(99)
            rank_b = wide[b].fillna(99)
            wide["better"] = ["A" if x < y else "B" if y < x else "=" for x, y in zip(rank_a, rank_b)]
            counts = wide["better"].value_counts()
            st.caption(
                f"{a} is ahead on {counts.get('A', 0)} queries, {b} on {counts.get('B', 0)}, "
                f"level on {counts.get('=', 0)}."
            )
        st.dataframe(wide, width="stretch", hide_index=True)

# --- data -------------------------------------------------------------------------------------
with tab_data:
    style.section("experiment_summary", "The same view you get from SQL: SELECT * FROM experiment_summary")
    st.dataframe(data.summary(), width="stretch", hide_index=True)
