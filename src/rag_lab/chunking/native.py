"""The two strategies that Docling implements itself (it does the segmenting and the splitting).

Docling's default table serialisation is a "row, column = value" triplet text, which embeds poorly,
so tables are serialised as Markdown instead.
"""

from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel

from rag_lab.chunking.base import ChunkContext, register
from rag_lab.chunking.models import Chunk


class _MarkdownTables(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def _convert(ctx: ChunkContext, chunker) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    if cfg.table_handling == "row-wise":
        ctx.warnings.append(
            f"table_handling 'row-wise' is not supported by '{cfg.strategy}'; tables use markdown"
        )
    chunks = []
    for c in chunker.chunk(dl_doc=ctx.doc):
        items = c.meta.doc_items
        labels = [i.label for i in items]
        is_table = DocItemLabel.TABLE in labels
        if cfg.table_handling == "skip" and labels and all(lab == DocItemLabel.TABLE for lab in labels):
            continue
        prov = items[0].prov[0] if items and items[0].prov else None
        chunks.append(
            Chunk(
                doc_id=ctx.doc_id,
                text=chunker.contextualize(c) if cfg.include_headings_in_text else c.text,
                modality="table" if is_table else "text",
                strategy=cfg.strategy,
                page=prov.page_no if prov else None,
                headings=list(c.meta.headings or []),
                bbox=list(prov.bbox.as_tuple()) if prov else None,
            )
        )
    return chunks


@register("hybrid")
def hybrid(ctx: ChunkContext) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=ctx.tokens.hf, max_tokens=cfg.max_tokens),
        merge_peers=cfg.hybrid.merge_peers,
        serializer_provider=_MarkdownTables(),
    )
    return _convert(ctx, chunker)


@register("hierarchical")
def hierarchical(ctx: ChunkContext) -> list[Chunk]:
    return _convert(ctx, HierarchicalChunker(serializer_provider=_MarkdownTables()))
