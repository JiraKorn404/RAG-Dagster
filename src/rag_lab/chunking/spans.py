"""Where chunks sit in the reference text (segment.reference_text), for the upload page's preview.

The section strategies know their spans from their own cuts. Docling's chunkers do not, so their
chunks are placed by the Docling items they were built from.
"""

import re
from collections import Counter

from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import Section, TableBlock, reference_text


def find_normalised(text: str, lo: int, hi: int, needle: str) -> tuple[int, int] | None:
    """(start, end) of `needle` inside text[lo:hi], ignoring differences in whitespace. The span
    is in `text` coordinates and covers exactly the matched characters."""
    pieces = [m for m in re.finditer(r"\S+", text[lo:hi])]
    words = needle.split()
    if not words:
        return None
    normalised = " ".join(m.group() for m in pieces)
    starts, pos = [], 0  # character offset in `normalised` where each word starts
    for m in pieces:
        starts.append(pos)
        pos += len(m.group()) + 1
    at = normalised.find(" ".join(words))
    if at < 0:
        return None
    first = max(i for i, s in enumerate(starts) if s <= at)
    last = first + len(words) - 1
    return lo + pieces[first].start(), lo + pieces[last].end()


def assign_docling_spans(
    chunks: list[Chunk],
    item_refs: list[list[str]],
    bodies: list[str],
    blocks: list[Section | TableBlock],
) -> None:
    """Set `span` and `span_approx` on chunks built by a Docling chunker.

    `item_refs[i]` are the `self_ref`s of chunk i's Docling items and `bodies[i]` its text without
    the headings. A chunk spans from its first to its last item that the segmenter kept. When several
    chunks come from the same items (Docling split a long paragraph, or a table), each is narrowed
    to its own text if that can be found, and otherwise keeps the whole region as an approximation.
    A chunk none of whose items was kept (a formula Docling could not decode) gets an empty span
    where the previous chunk ends.
    """
    item_spans: dict[str, tuple[int, int]] = {}
    for block in blocks:
        if isinstance(block, TableBlock):
            item_spans[block.item.self_ref] = (block.offset, block.offset + len(block.markdown))
        else:
            for paragraph, start, end in block.paragraph_spans():
                if paragraph.ref:
                    item_spans[paragraph.ref] = (start, end)

    text = reference_text(blocks)
    keys = [tuple(r for r in refs if r in item_spans) for refs in item_refs]
    shared = Counter(keys)
    cursor: dict[tuple[str, ...], int] = {}
    last_end = 0
    for chunk, key, body in zip(chunks, keys, bodies):
        if not key:
            chunk.span, chunk.span_approx = [last_end, last_end], False
            continue
        start = min(item_spans[r][0] for r in key)
        end = max(item_spans[r][1] for r in key)
        approx = False
        if shared[key] > 1:
            found = find_normalised(text, cursor.get(key, start), end, body)
            if found:
                start, end = found
                cursor[key] = end
            else:
                approx = True
        chunk.span, chunk.span_approx = [start, end], approx
        last_end = end
