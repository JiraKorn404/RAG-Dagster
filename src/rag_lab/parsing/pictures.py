"""The pictures of a parsed document as files: <doc_id>.pictures/<n>.png next to the parse files.

Docling already finds every picture and its box, with or without a text layer. The picture itself is
cropped here from the PDF page rendered with pypdfium2, so a document that is already parsed does not
have to go through Docling again to get its pictures. `n` is the picture's index in `doc.pictures`."""

from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import PictureItem

from rag_lab.config import ParseConfig


def pictures_dir(doc_id: str) -> str:
    return f"{doc_id}.pictures"


def kept_pictures(doc: DoclingDocument, cfg: ParseConfig) -> list[tuple[int, PictureItem]]:
    """The pictures that are kept: those with a place on a page and no side shorter than
    `picture_min_side`. The parse stage saves exactly these and the chunk stage makes a chunk of each."""
    kept = []
    for n, picture in enumerate(doc.pictures):
        if not picture.prov:
            continue
        box = picture.prov[0].bbox
        if min(abs(box.r - box.l), abs(box.t - box.b)) >= cfg.picture_min_side:
            kept.append((n, picture))
    return kept


def save_pictures(pdf_path: Path, doc: DoclingDocument, out_dir: Path, doc_id: str, cfg: ParseConfig) -> int:
    """Write the kept pictures of `doc` under out_dir/<doc_id>.pictures/. Returns how many."""
    folder = out_dir / pictures_dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    kept = kept_pictures(doc, cfg)
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pages: dict[int, object] = {}  # page number -> rendered page, rendered once
        for n, picture in kept:
            prov = picture.prov[0]
            if prov.page_no not in pages:
                pages[prov.page_no] = pdf[prov.page_no - 1].render(scale=cfg.picture_scale).to_pil()
            image = pages[prov.page_no]
            page = doc.pages[prov.page_no]
            scale = image.width / page.size.width
            box = prov.bbox.to_top_left_origin(page.size.height)
            image.crop(
                (
                    max(int(box.l * scale), 0),
                    max(int(box.t * scale), 0),
                    min(int(box.r * scale) + 1, image.width),
                    min(int(box.b * scale) + 1, image.height),
                )
            ).save(folder / f"{n}.png")
    finally:
        pdf.close()
    return len(kept)
