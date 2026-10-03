"""HTML for the upload page's two chunk previews. Pure functions over chunks, so they need no Streamlit.

A chunk's `span` is [start, end] in the reference text (the parsed document read in order). An exact
span is drawn as the chunk's own stretch of text; where two exact spans overlap the shared text gets
its own colour. An approximate span (a table split into several chunks, a paragraph Docling split) is
drawn as one outlined region labelled with the chunks that came from it. An empty span marks a chunk
with no source text (a formula Docling could not decode)."""

import html


def _label(i: int, chunk) -> str:
    parts = [f"#{i + 1}", f"{chunk.token_count} tok"]
    if chunk.page:
        parts.append(f"p. {chunk.page}")
    if chunk.modality == "table":
        parts.append("table")
    return " · ".join(parts)


def _numbers(indexes: list[int]) -> str:
    return " + ".join(f"#{i + 1}" for i in indexes)


def document_html(text: str, chunks: list, lo: int, hi: int) -> str:
    """The reference text from `lo` to `hi`, cut at every chunk boundary and marked up."""
    exact = [
        (i, c.span[0], c.span[1])
        for i, c in enumerate(chunks)
        if c.span and not c.span_approx and c.span[1] > c.span[0]
    ]
    regions: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(chunks):
        if c.span and c.span_approx and c.span[1] > c.span[0]:
            regions.setdefault((c.span[0], c.span[1]), []).append(i)
    empty = [(i, c.span[0]) for i, c in enumerate(chunks) if c.span and c.span[0] == c.span[1]]

    points = {lo, hi}
    for _, s, e in exact:
        points.update(p for p in (s, e) if lo < p < hi)
    for s, e in regions:
        points.update(p for p in (s, e) if lo < p < hi)
    ordered = sorted(points)

    out: list[str] = []
    for n, a in enumerate(ordered):
        marks: list[str] = []
        for i, s, _ in exact:
            if s == a:
                marks.append(f'<span class="cut cut-{i % 2}">{html.escape(_label(i, chunks[i]))}</span>')
        for (s, _), members in regions.items():
            if s == a:
                label = f"table · {_numbers(members)}" if all(
                    chunks[i].modality == "table" for i in members
                ) else f"split text · {_numbers(members)}"
                marks.append(f'<span class="cut cut-ap">{html.escape(label)} (approximate)</span>')
        for i, at in empty:
            if at == a and (a < hi or hi == len(text)):
                marks.append(
                    f'<span class="cut cut-none">#{i + 1} · no source text ({html.escape(chunks[i].text.strip()[:40])})</span>'
                )
        if n == len(ordered) - 1:
            out.extend(marks)  # marks at the very end of the text
            break
        b = ordered[n + 1]
        covering = [i for i, s, e in exact if s <= a and e >= b]
        if len(covering) >= 2:
            css = "ov"
            marks.append(f'<span class="cut cut-ov">overlap {html.escape(_numbers(covering))}</span>')
        elif covering:
            css = f"c{covering[0] % 2}"
        elif any(s <= a and e >= b for s, e in regions):
            css = "ap"
        else:
            css = "gap"
        out.append("".join(marks) + f'<span class="{css}">{html.escape(text[a:b])}</span>')
    return f'<div class="doc">{"".join(out)}</div>'


def embedded_html(chunks: list, first: int) -> str:
    """One block per chunk, with the stored text (its headings prefix dimmed)."""
    blocks = []
    for offset, c in enumerate(chunks):
        i = first + offset
        prefix = "\n".join(c.headings) + "\n" if c.headings else ""
        if prefix and c.text.startswith(prefix):
            body = f'<span class="dim">{html.escape(prefix)}</span>{html.escape(c.text[len(prefix):])}'
        else:
            body = html.escape(c.text)
        where = f"p. {c.page}" if c.page else "page ?"
        blocks.append(
            '<div class="hit"><div class="hit-top">'
            f'<span class="rank">{i + 1}</span><span class="sim">{c.token_count} tokens</span></div>'
            f'<div class="badges"><span class="badge {c.modality}">{c.modality}</span>'
            f'<span class="badge page">{html.escape(where)}</span></div>'
            + (f'<div class="heading">{html.escape(" › ".join(c.headings))}</div>' if c.headings else "")
            + f'<pre class="chunktext">{body}</pre></div>'
        )
    return "".join(blocks)


def key_html() -> str:
    swatch = lambda css, text: f'<span class="doc-key"><span class="{css}">&nbsp;&nbsp;&nbsp;</span> {text}</span>'  # noqa: E731
    return (
        '<div class="doc-keys">'
        + swatch("c0", "chunk")
        + swatch("c1", "next chunk")
        + swatch("ov", "text in two chunks (overlap)")
        + swatch("ap", "region the chunks came from (approximate)")
        + swatch("gap", "in no chunk, or a heading (headings are added to each chunk when embedding)")
        + "</div>"
    )

