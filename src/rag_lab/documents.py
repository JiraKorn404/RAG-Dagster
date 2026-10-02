"""Finding PDFs in data/raw. A document's id is a hash of its content, so renaming a file does not
make it a new document, and the same bytes under two names count once."""

import hashlib
from pathlib import Path

from rag_lab.paths import RAW_DIR

# (path, mtime_ns, size) -> doc_id, so the sensor does not re-hash every file on every tick
_hash_cache: dict[tuple[str, int, int], str] = {}


def doc_id_for(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hash_cache:
        _hash_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return _hash_cache[key]


def scan_raw() -> dict[str, Path]:
    """doc_id -> path for every PDF in data/raw."""
    return {
        doc_id_for(p): p
        for p in sorted(RAW_DIR.iterdir())
        if p.is_file() and p.suffix.lower() == ".pdf"
    }
