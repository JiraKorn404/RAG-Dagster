import hashlib
from pathlib import Path

import yaml


def load_queries(path: str | Path) -> tuple[str, list[dict]]:
    """(query_set_version, queries) from a YAML list. The version is a hash of the file, so editing
    a query or its expected results starts a new version without any manual bookkeeping."""
    raw = Path(path).read_bytes()
    queries = yaml.safe_load(raw) or []
    seen = set()
    for q in queries:
        if not q.get("id") or not q.get("query"):
            raise ValueError(f"{path}: every query needs an id and a query: {q}")
        if q["id"] in seen:
            raise ValueError(f"{path}: duplicate query id '{q['id']}'")
        seen.add(q["id"])
        if q.get("modality") not in (None, "text", "table"):
            raise ValueError(f"{path}: query '{q['id']}' has modality {q['modality']!r}")
        q["expected"] = q.get("expected") or None  # an empty list means "not filled in yet"
    if not queries:
        raise ValueError(f"{path} has no queries")
    return hashlib.sha256(raw).hexdigest()[:12], queries
