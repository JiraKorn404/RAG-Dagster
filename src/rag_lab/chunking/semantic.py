"""Semantic chunking: break a section where the meaning shifts.

Each sentence is embedded together with `buffer_size` neighbours on each side. The cosine distance
between consecutive sentences' embeddings is large where the topic changes; sentences are cut at the
distances above a threshold. The threshold is taken over the whole document, so a section with only
a few sentences is not forced to split. Chunks under `min_tokens` are merged into a neighbour, and
chunks over the token budget are split again with the recursive strategy.
"""

import math
import re
import statistics

from rag_lab.chunking.base import ChunkContext, register
from rag_lab.chunking.models import Chunk
from rag_lab.chunking.recursive import recursive_spans
from rag_lab.chunking.sectioned import chunk_sections
from rag_lab.chunking.segment import Section, segment
from rag_lab.config import SemanticSettings
from rag_lab.embedding.cache import embed_cached
from rag_lab.metrics.stats import percentile


def _sentences(text: str, pattern: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in re.finditer(pattern, text):
        if text[pos : m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1 - dot / norm if norm else 1.0


def _threshold(distances: list[float], s: SemanticSettings) -> float:
    if not distances:
        return math.inf
    if s.breakpoint_type == "percentile":
        return percentile(distances, s.breakpoint_threshold)
    if s.breakpoint_type == "stddev":
        return statistics.fmean(distances) + s.breakpoint_threshold * statistics.pstdev(distances)
    return s.breakpoint_threshold


@register("semantic")
def semantic(ctx: ChunkContext) -> list[Chunk]:
    if ctx.embedder is None:
        raise ValueError("The semantic strategy needs an embedder (Ollama)")
    s = ctx.cfg.chunk.semantic
    sections = [b for b in segment(ctx.doc) if isinstance(b, Section)]

    # Pass 1: embed every sentence group and collect the distances, to pick one threshold.
    analysed: dict[str, tuple[list[tuple[int, int]], list[float]]] = {}
    totals = {"texts": 0, "cache_hits": 0, "embedded": 0, "embed_ms": 0.0}
    for sec in sections:
        text = sec.text
        sents = _sentences(text, s.sentence_pattern)
        if len(sents) < 2:
            analysed[text] = (sents, [])
            continue
        n, b = len(sents), s.buffer_size
        groups = [text[sents[max(0, i - b)][0] : sents[min(n - 1, i + b)][1]] for i in range(n)]
        vectors, info = embed_cached(groups, ctx.embedder, ctx.cfg.embed)
        for k in totals:
            totals[k] += info[k]
        analysed[text] = (sents, [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(n - 1)])

    all_distances = [d for _, dists in analysed.values() for d in dists]
    threshold = _threshold(all_distances, s)
    ctx.stats.update(
        sentences=sum(len(sents) for sents, _ in analysed.values()),
        group_texts=totals["texts"],
        embed_cache_hits=totals["cache_hits"],
        embedded=totals["embedded"],
        embed_ms=round(totals["embed_ms"], 1),
        threshold=None if math.isinf(threshold) else round(threshold, 4),
        breakpoints=0,
    )

    # Pass 2: cut each section at the breakpoints, merge the small chunks, split the oversize ones.
    def split(text: str, budget: int, _ctx: ChunkContext) -> list[tuple[int, int]]:
        sents, dists = analysed[text]
        if not sents:
            return [(0, len(text))]
        cuts = [i for i, d in enumerate(dists) if d > threshold]
        ctx.stats["breakpoints"] += len(cuts)
        spans, first = [], 0
        for i in cuts:
            spans.append((sents[first][0], sents[i][1]))
            first = i + 1
        spans.append((sents[first][0], sents[-1][1]))

        merged: list[tuple[int, int]] = []
        for span in spans:
            if merged and ctx.tokens.count(text[span[0] : span[1]]) < s.min_tokens:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        if len(merged) > 1 and ctx.tokens.count(text[merged[0][0] : merged[0][1]]) < s.min_tokens:
            merged[1] = (merged[0][0], merged[1][1])
            merged.pop(0)

        out: list[tuple[int, int]] = []
        for start, end in merged:
            out.extend(
                recursive_spans(text, start, end, budget, ctx.cfg.chunk.recursive.separators, ctx.tokens)
            )
        return out

    return chunk_sections(ctx, split)
