"""Small helpers shared by the pages: reading rag_metrics into a frame, and model and strategy labels
and ordering (so each model keeps the same colour on every chart)."""

import os

import httpx
import pandas as pd
import psycopg
import streamlit as st

from rag_lab.config import embed_family, embed_model_label

STRATEGIES = ["hybrid", "hierarchical", "fixed", "recursive", "semantic"]


def frame(sql: str, params: tuple = ()) -> pd.DataFrame:
    with psycopg.connect(os.environ["METRICS_DATABASE_URL"]) as conn:
        cur = conn.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


model_label = embed_model_label  # 'qwen3-embedding:4b' -> '4b'; another family keeps its whole name


def model_order(labels) -> list[str]:
    """Models by size (0.6b, embeddinggemma-2:740m, 4b, 8b)."""

    def size(label: str) -> float:
        tag = label.split(":")[-1].lower()
        try:
            return float(tag[:-1]) / (1000 if tag.endswith("m") else 1)
        except ValueError:
            return float("inf")

    return sorted(set(labels), key=lambda label: (size(label), label))


def strategy_order(names) -> list[str]:
    present = set(names)
    return [s for s in STRATEGIES if s in present]


def _ollama_models() -> list[dict]:
    try:
        reply = httpx.get(f"{os.environ['OLLAMA_BASE_URL'].rstrip('/')}/api/tags", timeout=5)
        return reply.json()["models"]
    except Exception:  # noqa: BLE001  (Ollama not reachable: callers fall back to the default model)
        return []


@st.cache_data(ttl=60)
def embedding_models() -> list[str]:
    """The embedding models Ollama has that are of a known family (config.EMBED_FAMILIES), since a
    model needs its family's tokenizer and prompt templates."""
    return sorted(m["name"] for m in _ollama_models() if embed_family(m["name"]))


@st.cache_data(ttl=60)
def has_model(name: str) -> bool:
    """Whether Ollama has this model (the OCR model, for example)."""
    return any(m["name"] == name for m in _ollama_models())


@st.cache_data(ttl=60)
def reranker_models() -> list[str]:
    """The reranker models Ollama has that can generate (a name with `reranker` in it; a build that only
    embeds cannot answer yes or no). Whether the answer is sharp enough: see reranking/ollama.py."""
    return sorted(
        m["name"]
        for m in _ollama_models()
        if "reranker" in m["name"].lower() and "completion" in m.get("capabilities", ["completion"])
    )


@st.cache_data(ttl=60)
def chat_models() -> dict[str, list[str]]:
    """The models Ollama has that can chat with tools (the chatbot's model), each with its capabilities
    (`thinking` and `vision` are what the page asks about). Rerankers are left out: they also report
    these capabilities but only answer yes or no."""
    return {
        m["name"]: m["capabilities"]
        for m in _ollama_models()
        if {"completion", "tools"} <= set(m.get("capabilities", []))
        and "reranker" not in m["name"].lower()
    }
