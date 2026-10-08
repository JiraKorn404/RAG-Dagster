"""One chunk per picture, added after the chosen strategy's chunks when `parse.pictures` is on.

A picture chunk's `image` is its file (parsing/pictures.py) and its text is the picture's caption, or
"Figure on page N" when Docling linked none, so the text is never empty: BM25 and the rerankers read
it, and the embedding does too unless `embed.picture_input` is `image`. It has no text of its own in
the document, so its span is empty, at the end of the last text or table before it."""

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.segment import TableBlock, segment
from rag_lab.parsing.pictures import kept_pictures, pictures_dir


def picture_chunks(ctx: ChunkContext) -> list[Chunk]:
    doc, cfg = ctx.doc, ctx.cfg
    # where every text and table item ends in the reference text, and the headings it is under
    where: dict[str, tuple[int, list[str]]] = {}
    for block in segment(doc):
        if isinstance(block, TableBlock):
            where[block.item.self_ref] = (block.offset + len(block.markdown), block.headings)
        else:
            for paragraph, _, end in block.paragraph_spans():
                where[paragraph.ref] = (end, block.headings)

    kept = {picture.self_ref: n for n, picture in kept_pictures(doc, cfg.parse)}
    at, headings = 0, []
    chunks = []
    for item, _ in doc.iterate_items():
        if item.self_ref in where:
            at, headings = where[item.self_ref]
        elif item.self_ref in kept:
            prov = item.prov[0]
            text = " ".join(item.caption_text(doc).split()) or f"Figure on page {prov.page_no}"
            if cfg.chunk.include_headings_in_text and headings:
                text = "\n".join(headings) + "\n" + text
            chunks.append(
                Chunk(
                    doc_id=ctx.doc_id,
                    text=text,
                    modality="picture",
                    strategy=cfg.chunk.strategy,
                    page=prov.page_no,
                    headings=list(headings),
                    bbox=list(prov.bbox.as_tuple()),
                    span=[at, at],
                    image=f"{pictures_dir(ctx.doc_id)}/{kept[item.self_ref]}.png",
                )
            )
    return chunks
