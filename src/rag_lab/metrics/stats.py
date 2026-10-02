def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in 0..100."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
