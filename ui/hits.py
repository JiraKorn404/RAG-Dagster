"""The card that shows one retrieved chunk, shared by Try a query and the Chatbot."""

import html


def hit_card(hit, color: str, cited: bool = False) -> str:
    body = " ".join(hit.text.split())
    snippet = html.escape(body[:330] + ("…" if len(body) > 330 else ""))
    where = f"{hit.source_file}, p. {hit.page}" if hit.page and hit.source_file else (
        f"p. {hit.page}" if hit.page else (hit.source_file or "page ?")
    )
    return (
        '<div class="hit"><div class="hit-top">'
        f'<span class="rank">{hit.rank}</span><span class="sim">{hit.similarity:.3f}</span>'
        f'<div class="bar"><span style="width:{max(0.0, min(1.0, hit.similarity)) * 100:.0f}%;background:{color}"></span></div>'
        "</div>"
        f'<div class="badges"><span class="badge {hit.modality}">{hit.modality}</span>'
        f'<span class="badge page">{html.escape(where)}</span>'
        + (f'<span class="badge agree">cited [{hit.rank}]</span>' if cited else "")
        + "</div>"
        + (f'<div class="heading">{html.escape(" › ".join(hit.headings))}</div>' if hit.headings else "")
        + f'<div class="snippet">{snippet}</div>'
        f"<details><summary>Full text</summary><pre>{html.escape(hit.text)}</pre></details></div>"
    )
