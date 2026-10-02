import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from rag_lab.config import ParseConfig


@dataclass
class ParsedDocument:
    doc_id: str
    source_file: str
    status: str  # success | partial_success | failure
    pages: int = 0
    tables: int = 0
    table_cells: int = 0
    text_items: int = 0
    text_chars: int = 0
    chars_per_page: float = 0.0  # near zero means a scanned PDF: there is no text layer and OCR is off
    markdown_chars: int = 0
    model_load_seconds: float = 0.0
    parse_seconds: float = 0.0
    pages_per_second: float = 0.0
    errors: list[str] = field(default_factory=list)
    markdown_preview: str = ""


def build_converter(cfg: ParseConfig) -> DocumentConverter:
    """OCR and picture features are fixed off: this lab ingests text and tables from text-layer PDFs."""
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=cfg.do_table_structure,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode(cfg.table_mode),
            do_cell_matching=cfg.table_cell_matching,
        ),
        do_formula_enrichment=cfg.do_formula_enrichment,
        do_code_enrichment=cfg.do_code_enrichment,
        do_picture_classification=False,
        do_picture_description=False,
        generate_picture_images=False,
        document_timeout=cfg.document_timeout,
        accelerator_options=AcceleratorOptions(
            num_threads=cfg.num_threads, device=AcceleratorDevice.CPU
        ),
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(path: Path, doc_id: str, cfg: ParseConfig, out_dir: Path) -> ParsedDocument:
    """Parse one PDF and write <doc_id>.json (Docling document), <doc_id>.md and <doc_id>.meta.json."""
    t0 = time.perf_counter()
    converter = build_converter(cfg)
    converter.initialize_pipeline(InputFormat.PDF)  # loads the models, so parse time excludes it
    model_load_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = converter.convert(path, raises_on_error=False)
    parse_seconds = time.perf_counter() - t1

    parsed = ParsedDocument(
        doc_id=doc_id,
        source_file=path.name,
        status=result.status.value,
        model_load_seconds=round(model_load_seconds, 3),
        parse_seconds=round(parse_seconds, 3),
        errors=[str(e.error_message) for e in result.errors],
    )
    if parsed.status == "failure":
        return parsed

    doc = result.document
    markdown = doc.export_to_markdown()
    text_chars = sum(len(t.text) for t in doc.texts)
    parsed.pages = len(doc.pages)
    parsed.tables = len(doc.tables)
    parsed.table_cells = sum(len(t.data.table_cells) for t in doc.tables)
    parsed.text_items = len(doc.texts)
    parsed.text_chars = text_chars
    parsed.chars_per_page = round(text_chars / parsed.pages, 1) if parsed.pages else 0.0
    parsed.markdown_chars = len(markdown)
    parsed.pages_per_second = round(parsed.pages / parse_seconds, 4) if parse_seconds else 0.0
    parsed.markdown_preview = markdown[:1500]

    doc.save_as_json(out_dir / f"{doc_id}.json")
    (out_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    meta = {**asdict(parsed), "parse_config": cfg.model_dump(mode="json")}
    meta.pop("markdown_preview")
    (out_dir / f"{doc_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return parsed
