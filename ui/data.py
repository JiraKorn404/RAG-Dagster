"""Reads from the rag_metrics database. Postgres is the source of truth; nothing is cached for long."""

import os

import pandas as pd
import psycopg
import streamlit as st

STRATEGIES = ["hybrid", "hierarchical", "fixed", "recursive", "semantic"]
_LATEST_RUN = """
    SELECT DISTINCT ON (config_hash) benchmark_run_id, config_hash
    FROM benchmark_runs
    WHERE finished_at IS NOT NULL AND query_set_version = %s
    ORDER BY config_hash, started_at DESC
"""
_EXPERIMENT_COLUMNS = """
    e.name, e.config -> 'embed' ->> 'model' AS model, e.config -> 'chunk' ->> 'strategy' AS strategy
"""


def frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(os.environ["METRICS_DATABASE_URL"]) as conn:
        cur = conn.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def model_label(model: str) -> str:
    """'qwen3-embedding:4b' -> '4b'."""
    return model.split(":")[-1]


def model_order(labels) -> list[str]:
    """Models by size (0.6b, 4b, 8b), so each keeps the same colour on every chart."""

    def size(label: str) -> float:
        try:
            return float(label.rstrip("bB"))
        except ValueError:
            return float("inf")

    return sorted(set(labels), key=lambda label: (size(label), label))


def strategy_order(names) -> list[str]:
    present = set(names)
    return [s for s in STRATEGIES if s in present]


def with_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["model"].map(model_label)
    return df


@st.cache_data(ttl=15)
def summary() -> pd.DataFrame:
    return with_label(frame("SELECT * FROM experiment_summary"))


@st.cache_data(ttl=15)
def versions() -> pd.DataFrame:
    """Query-set versions that have finished benchmarks, newest first."""
    return frame(
        """SELECT query_set_version, count(DISTINCT config_hash) AS experiments,
                  max(started_at) AS latest
           FROM benchmark_runs WHERE finished_at IS NOT NULL AND query_set_version IS NOT NULL
           GROUP BY query_set_version ORDER BY max(started_at) DESC"""
    )


@st.cache_data(ttl=15)
def benchmark_metrics(version: str) -> pd.DataFrame:
    """Every metric of each experiment's latest finished benchmark on this query-set version."""
    return with_label(
        frame(
            f"""WITH last AS ({_LATEST_RUN})
                SELECT {_EXPERIMENT_COLUMNS}, m.metric, m.k, m.modality, m.value
                FROM last JOIN benchmark_metrics m USING (benchmark_run_id)
                JOIN experiments e USING (config_hash)""",
            (version,),
        )
    )


@st.cache_data(ttl=15)
def first_ranks(version: str) -> pd.DataFrame:
    """Rank of the first relevant hit for each query with expected results (null: not retrieved)."""
    return with_label(
        frame(
            f"""WITH last AS ({_LATEST_RUN})
                SELECT {_EXPERIMENT_COLUMNS}, r.query_id, q.query_text, r.first_relevant_rank AS rank
                FROM last JOIN query_results r USING (benchmark_run_id)
                JOIN experiments e USING (config_hash)
                JOIN eval_queries q ON q.query_id = r.query_id AND q.query_set_version = %s
                WHERE r.repeat = 0 AND q.expected IS NOT NULL""",
            (version, version),
        )
    )


def pick(
    metrics: pd.DataFrame, metric: str, k: int | None = None, modality: str | None = None
) -> pd.DataFrame:
    """One value per experiment for a metric. `modality` None means all queries."""
    rows = metrics[metrics["metric"] == metric]
    rows = rows[rows["k"].isna()] if k is None else rows[rows["k"] == k]
    rows = rows[rows["modality"].isna()] if modality is None else rows[rows["modality"] == modality]
    return rows.dropna(subset=["value"])[["name", "model", "strategy", "value"]]
