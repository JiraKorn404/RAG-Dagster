"""Step 2 for the fixed, recursive and semantic strategies: split each Section with a strategy's
splitter, and turn tables into chunks according to `table_handling`."""

from collections.abc import Callable

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import Section, TableBlock, segment

# (section text, token budget, context) -> character spans of the text, one per chunk
Splitter = Callable[[str, int, ChunkContext], list[tuple[int, int]]]


def chunk_sections(ctx: ChunkContext, splitter: Splitter) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    strategy = cfg.strategy
    chunks: list[Chunk] = []
    for block in segment(ctx.doc):
        prefix = "\n".join(block.headings) + "\n" if cfg.include_headings_in_text and block.headings else ""
        # Headings are part of the embedded text, so they use up part of the budget (but never more
        # than half of it, so a very long heading cannot starve the body).
        budget = max(cfg.max_tokens - ctx.tokens.count(prefix), cfg.max_tokens // 2) if prefix else cfg.max_tokens

        if isinstance(block, TableBlock):
            if cfg.table_handling == "skip":
                continue
            for text in _table_texts(ctx, block, budget):
                chunks.append(
                    Chunk(ctx.doc_id, prefix + text, "table", strategy, block.page, block.headings, block.bbox)
                )
            continue

        for start, end in splitter(block.text, budget, ctx):
            body = block.text[start:end].strip()
            if body:
                para = block.locate(start)
                chunks.append(
                    Chunk(ctx.doc_id, prefix + body, "text", strategy, para.page, block.headings, para.bbox)
                )
    return chunks


def _table_texts(ctx: ChunkContext, block: TableBlock, budget: int) -> list[str]:
    if ctx.cfg.chunk.table_handling == "row-wise":
        df = block.item.export_to_dataframe(doc=ctx.doc)
        rows = [
            "; ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
            for _, row in df.iterrows()
        ]
        rows = [r for r in rows if r]
        if rows:
            return rows
    markdown = block.item.export_to_markdown(doc=ctx.doc)
    if ctx.tokens.count(markdown) <= budget:
        return [markdown]
    return _split_markdown_table(markdown, budget, ctx)


def _split_markdown_table(markdown: str, budget: int, ctx: ChunkContext) -> list[str]:
    """Split an oversize table by rows, repeating the header in every piece. A single row that is
    over budget stays whole: it cannot be split without cutting a cell."""
    lines = markdown.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and ctx.tokens.count("\n".join(header + current + [row])) > budget:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces
