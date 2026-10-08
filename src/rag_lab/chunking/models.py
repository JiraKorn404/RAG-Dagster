import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    text: str
    modality: str  # "text", "table" or "picture"
    strategy: str
    page: int | None = None
    headings: list[str] = field(default_factory=list)
    bbox: list[float] | None = None  # [left, top, right, bottom] of the first element, PDF points
    token_count: int = 0
    chunk_id: str = ""  # f"{doc_id}:{index}", assigned once the document's chunks are final
    # [start, end] in the reference text (chunking/segment.py: reference_text), for the upload page's
    # preview. An empty span (start == end) means the chunk has no source text. Not stored in Qdrant.
    span: list[int] | None = None
    span_approx: bool = False  # the span is the region the chunk came from, wider than the chunk
    # A picture chunk's file, relative to the parse folder: "<doc_id>.pictures/<n>.png"
    image: str | None = None


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
