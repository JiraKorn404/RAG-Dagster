"""Small helpers shared by the pages: reading rag_metrics into a frame, and model and strategy labels
and ordering (so each model keeps the same colour on every chart)."""

import os

import httpx
import pandas as pd
import psycopg
import streamlit as st

STRATEGIES = ["hybrid", "hierarchical", "fixed", "recursive", "semantic"]


def frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(os.environ["METRICS_DATABASE_URL"]) as conn:
        cur = conn.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def model_label(model: str) -> str:
    """'qwen3-embedding:4b' -> '4b'."""
    return model.split(":")[-1]


def model_order(labels) -> list[str]:
    """Models by size (0.6b, 4b, 8b)."""

    def size(label: str) -> float:
        try:
            return float(label.rstrip("bB"))
        except ValueError:
            return float("inf")

    return sorted(set(labels), key=lambda label: (size(label), label))


def strategy_order(names) -> list[str]:
    present = set(names)
    return [s for s in STRATEGIES if s in present]


@st.cache_data(ttl=60)
def embedding_models() -> list[str]:
    """The Qwen3 embedding models Ollama has: the tokenizer used for chunk sizes is Qwen3's."""
    try:
        reply = httpx.get(f"{os.environ['OLLAMA_BASE_URL'].rstrip('/')}/api/tags", timeout=5)
        names = [m["name"] for m in reply.json()["models"]]
    except Exception:  # noqa: BLE001  (Ollama not reachable: callers fall back to the default model)
        return []
    return sorted(n for n in names if n.startswith("qwen3-embedding"))
