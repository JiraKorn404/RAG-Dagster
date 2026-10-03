"""The chunk stage on LlamaIndex: the same strategies as the native engine, with LlamaIndex's
splitters doing the splitting.

- hybrid, hierarchical: DoclingNodeParser around the same Docling chunker the native engine builds.
- fixed: TokenTextSplitter. recursive: SentenceSplitter. semantic: SemanticSplitterNodeParser, then
  SentenceSplitter for nodes that are still over the limit.

Docling-aware segmenting (headings, pages, tables) is still ours: LlamaIndex splitters only see text.
Each text section becomes a Document whose one embed-visible metadata value is its headings, so the
splitters reserve room for them and `node.get_content(MetadataMode.EMBED)` is "headings, newline,
body", as in the native engine. Tables never go through a splitter.

What differs from the native engine (see PLAN.md, Phase 8): `recursive.separators`, the `stddev` and
`absolute` breakpoint types and `semantic.min_tokens` do not exist here; the semantic threshold is
taken per section, not per document; token windows of `fixed` break on word boundaries.
"""

import json
import re

from docling_core.types.doc import BoundingBox
from llama_index.core import Document
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.node_parser.docling import DoclingNodeParser

from rag_lab.chunking.base import ChunkContext
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.native import build_chunker, warn_row_wise
from rag_lab.chunking.sectioned import table_chunks
from rag_lab.chunking.segment import Section, TableBlock, segment
from rag_lab.chunking.spans import assign_docling_spans
from rag_lab.config import RecursiveSettings
from rag_lab.embedding.llamaindex import OllamaLabEmbedding


def chunk_llamaindex(ctx: ChunkContext) -> list[Chunk]:
    if ctx.cfg.chunk.strategy in ("hybrid", "hierarchical"):
        return _docling_chunks(ctx)
    return _section_chunks(ctx)


def _docling_chunks(ctx: ChunkContext) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    warn_row_wise(ctx)
    parser = DoclingNodeParser(chunker=build_chunker(ctx))
    document = Document(id_=ctx.doc_id, text=json.dumps(ctx.doc.export_to_dict()))

    chunks, item_refs, bodies = [], [], []
    for node in parser.get_nodes_from_documents([document]):
        items = node.metadata.get("doc_items") or []
        labels = [i["label"] for i in items]
        if cfg.table_handling == "skip" and labels and all(lab == "table" for lab in labels):
            continue
        headings = list(node.metadata.get("headings") or [])
        prov = items[0]["prov"][0] if items and items[0].get("prov") else None
        # Docling's contextualize(): the headings, one per line, then the chunk text
        text = "\n".join([*headings, node.text]) if cfg.include_headings_in_text else node.text
        chunks.append(
            Chunk(
                doc_id=ctx.doc_id,
                text=text,
                modality="table" if "table" in labels else "text",
                strategy=cfg.strategy,
                page=prov["page_no"] if prov else None,
                headings=headings,
                bbox=list(BoundingBox.model_validate(prov["bbox"]).as_tuple()) if prov else None,
            )
        )
        item_refs.append([i["self_ref"] for i in items])
        bodies.append(node.text)
    assign_docling_spans(chunks, item_refs, bodies, segment(ctx.doc))
    return chunks


def _section_chunks(ctx: ChunkContext) -> list[Chunk]:
    cfg = ctx.cfg.chunk
    blocks = segment(ctx.doc)
    sections = [b for b in blocks if isinstance(b, Section)]
    documents = [_document(ctx, f"{ctx.doc_id}:{i}", sec) for i, sec in enumerate(sections)]
    nodes_by_section = _split(ctx, documents)

    chunks: list[Chunk] = []
    next_section = 0
    for block in blocks:
        if isinstance(block, TableBlock):
            if cfg.table_handling != "skip":
                chunks.extend(table_chunks(ctx, block))
            continue
        for text, offset, length, exact in nodes_by_section[documents[next_section].id_]:
            para = block.locate(offset)
            chunks.append(
                Chunk(
                    ctx.doc_id, text, "text", cfg.strategy, para.page, block.headings, para.bbox,
                    span=[block.offset + offset, block.offset + offset + length],
                    span_approx=not exact,
                )
            )
        next_section += 1
    return chunks


def _document(ctx: ChunkContext, doc_id: str, section: Section) -> Document:
    """A section as a Document. Its headings are the one metadata value the embedder sees."""
    headings = "\n".join(section.headings) if ctx.cfg.chunk.include_headings_in_text else ""
    return Document(
        id_=doc_id,
        text=section.text,
        metadata={"headings": headings} if headings else {},
        metadata_template="{value}",
        metadata_seperator="\n",  # (sic) LlamaIndex's alias for metadata_separator
        text_template="{metadata_str}\n{content}",
    )


def _split(
    ctx: ChunkContext, documents: list[Document]
) -> dict[str, list[tuple[str, int, int, bool]]]:
    """Run the strategy's LlamaIndex splitters. Returns, per document id, each chunk's embedded
    text, the character offset and length of its body in the section (for its page, bounding box and
    span), and whether that body was found verbatim in the section text."""
    cfg = ctx.cfg.chunk
    tokenizer = lambda text: ctx.tokens.hf.encode(text, add_special_tokens=False)  # noqa: E731
    text_of = {d.id_: d.text for d in documents}

    if cfg.strategy == "fixed":
        if cfg.overlap >= cfg.max_tokens:
            raise ValueError(f"overlap ({cfg.overlap}) must be smaller than max_tokens ({cfg.max_tokens})")
        nodes = TokenTextSplitter(
            chunk_size=cfg.max_tokens, chunk_overlap=cfg.overlap, tokenizer=tokenizer
        ).get_nodes_from_documents(documents)
    else:
        # chunk_overlap is explicit because SentenceSplitter's own default is 200 tokens
        sentence_splitter = SentenceSplitter(
            chunk_size=cfg.max_tokens,
            chunk_overlap=0,
            paragraph_separator="\n\n",
            tokenizer=tokenizer,
        )
        if cfg.strategy == "recursive":
            if cfg.recursive.separators != RecursiveSettings().separators:
                raise ValueError("recursive.separators is not supported with engine 'llamaindex'")
            nodes = sentence_splitter.get_nodes_from_documents(documents)
        else:
            nodes = sentence_splitter(_semantic_nodes(ctx, documents))

    out: dict[str, list[tuple[str, int, int, bool]]] = {d.id_: [] for d in documents}
    cursor: dict[str, int] = {}
    for node in nodes:
        body = node.get_content(MetadataMode.NONE).strip()
        if not body:
            continue
        section_id = node.ref_doc_id
        # LlamaIndex splitters copy the section's metadata to every node, so the offset is found
        # by searching for the body from the previous chunk's start (chunks can overlap).
        start = cursor.get(section_id, 0)
        found = text_of[section_id].find(body, start)
        offset = found if found >= 0 else start
        cursor[section_id] = offset
        out[section_id].append(
            (node.get_content(MetadataMode.EMBED), offset, len(body), found >= 0)
        )
    return out


def _semantic_nodes(ctx: ChunkContext, documents: list[Document]) -> list[TextNode]:
    s = ctx.cfg.chunk.semantic
    if ctx.embedder is None:
        raise ValueError("The semantic strategy needs an embedder (Ollama)")
    if s.breakpoint_type != "percentile":
        raise ValueError(
            f"semantic.breakpoint_type '{s.breakpoint_type}' is not supported with engine "
            "'llamaindex' (percentile only)"
        )
    ctx.warnings.append("semantic.min_tokens is not applied with engine 'llamaindex'")

    embed_model = OllamaLabEmbedding(ctx.embedder, ctx.cfg.embed)
    parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=s.buffer_size,
        breakpoint_percentile_threshold=round(s.breakpoint_threshold),
        sentence_splitter=lambda text: _sentence_pieces(text, s.sentence_pattern),
    )
    nodes = parser.get_nodes_from_documents(documents)
    stats = embed_model.stats
    ctx.stats.update(
        sentences=stats["texts"],  # one embedded group per sentence
        group_texts=stats["texts"],
        embed_cache_hits=stats["cache_hits"],
        embedded=stats["embedded"],
        embed_ms=round(stats["embed_ms"], 1),
    )
    return nodes


def _sentence_pieces(text: str, pattern: str) -> list[str]:
    """Cut `text` after each sentence boundary, keeping the boundary with the sentence before it,
    so the pieces join back into `text` (LlamaIndex's semantic splitter concatenates them)."""
    pieces, pos = [], 0
    for m in re.finditer(pattern, text):
        if text[pos : m.end()].strip():
            pieces.append(text[pos : m.end()])
            pos = m.end()
    if text[pos:].strip():
        pieces.append(text[pos:])
    return pieces
