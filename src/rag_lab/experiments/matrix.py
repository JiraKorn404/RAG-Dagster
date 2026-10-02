"""Expands a matrix file (models x strategies) into experiment configs. Plain Python."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_lab.config import ChunkConfig, EmbedConfig, ExperimentConfig, IndexConfig, ParseConfig


@dataclass
class Matrix:
    documents: list[str] | str  # file names in data/raw, or "all"
    models: dict[str, str]  # label -> Ollama model
    strategies: list[str]
    shared: dict
    benchmark: dict


def load_matrix(path: str | Path) -> Matrix:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("documents", "models", "strategies"):
        if not raw.get(key):
            raise ValueError(f"{path}: '{key}' is required")
    return Matrix(
        documents=raw["documents"],
        models=raw["models"],
        strategies=raw["strategies"],
        shared=raw.get("shared") or {},
        benchmark=raw.get("benchmark") or {},
    )


def expand(matrix: Matrix) -> list[ExperimentConfig]:
    """One config per model and strategy, model by model so that each model is loaded on the Mac once."""
    configs = []
    for label, model in matrix.models.items():
        for strategy in matrix.strategies:
            chunk = {**matrix.shared.get("chunk", {}), "strategy": strategy}
            if strategy != "fixed":
                chunk["overlap"] = 0  # only `fixed` uses overlap; the others warn if it is set
            configs.append(
                ExperimentConfig(
                    name=f"{label}-{strategy}",
                    parse=ParseConfig(**matrix.shared.get("parse", {})),
                    chunk=ChunkConfig(**chunk),
                    embed=EmbedConfig(**{**matrix.shared.get("embed", {}), "model": model}),
                    index=IndexConfig(**matrix.shared.get("index", {})),
                )
            )
    return configs
