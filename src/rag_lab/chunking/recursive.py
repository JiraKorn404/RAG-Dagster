from rag_lab.chunking.base import ChunkContext, register
from rag_lab.chunking.fixed import token_windows
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.sectioned import chunk_sections
from rag_lab.chunking.tokens import Tokens


def recursive_spans(
    text: str, start: int, end: int, budget: int, separators: list[str], tokens: Tokens
) -> list[tuple[int, int]]:
    """Split text[start:end] into spans of at most `budget` tokens.

    Try the first separator; any piece still over budget is split with the next one, and so on.
    Adjacent small pieces are then merged back up to the budget. If the separators run out, the
    piece is cut into token windows.
    """
    if tokens.count(text[start:end]) <= budget:
        return [(start, end)]
    if not separators:
        return token_windows(text, start, end, budget, 0, tokens)

    pieces: list[tuple[int, int]] = []
    for s, e in _split_on(text, start, end, separators[0]):
        pieces.extend(recursive_spans(text, s, e, budget, separators[1:], tokens))

    merged = []
    cur_s, cur_e = pieces[0]
    for s, e in pieces[1:]:
        if tokens.count(text[cur_s:e]) <= budget:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _split_on(text: str, start: int, end: int, sep: str) -> list[tuple[int, int]]:
    """Cut at each occurrence of `sep`, keeping the separator with the piece before it."""
    parts = []
    pos = start
    while True:
        i = text.find(sep, pos, end)
        if i == -1 or i + len(sep) >= end:
            break
        parts.append((pos, i + len(sep)))
        pos = i + len(sep)
    parts.append((pos, end))
    return parts


def _split(text: str, budget: int, ctx: ChunkContext) -> list[tuple[int, int]]:
    return recursive_spans(text, 0, len(text), budget, ctx.cfg.chunk.recursive.separators, ctx.tokens)


@register("recursive")
def recursive(ctx: ChunkContext) -> list[Chunk]:
    return chunk_sections(ctx, _split)
