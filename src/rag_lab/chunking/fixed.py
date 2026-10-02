from rag_lab.chunking.base import ChunkContext, register
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.sectioned import chunk_sections
from rag_lab.chunking.tokens import Tokens


def token_windows(
    text: str, start: int, end: int, budget: int, overlap: int, tokens: Tokens
) -> list[tuple[int, int]]:
    """Windows of `budget` tokens over text[start:end], stepping by budget - overlap."""
    offsets = tokens.offsets(text[start:end])
    spans = []
    i = 0
    while i < len(offsets):
        j = min(i + budget, len(offsets))
        spans.append((start + offsets[i][0], start + offsets[j - 1][1]))
        if j == len(offsets):
            break
        i += budget - overlap
    return spans


def _split(text: str, budget: int, ctx: ChunkContext) -> list[tuple[int, int]]:
    overlap = ctx.cfg.chunk.overlap
    if overlap >= budget:
        raise ValueError(f"overlap ({overlap}) must be smaller than the token budget ({budget})")
    return token_windows(text, 0, len(text), budget, overlap, ctx.tokens)


@register("fixed")
def fixed(ctx: ChunkContext) -> list[Chunk]:
    return chunk_sections(ctx, _split)
