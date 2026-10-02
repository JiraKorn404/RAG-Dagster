import math


def truncate_and_normalise(vectors: list[list[float]], dimension: int | None) -> list[list[float]]:
    """Matryoshka truncation: keep the first `dimension` values, then L2-normalise again."""
    if dimension is None:
        return vectors
    out = []
    for v in vectors:
        cut = v[:dimension]
        norm = math.sqrt(sum(x * x for x in cut)) or 1.0
        out.append([x / norm for x in cut])
    return out
