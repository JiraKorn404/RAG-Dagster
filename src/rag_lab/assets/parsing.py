from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from rag_lab.assets.partitions import documents_partitions
from rag_lab.documents import scan_raw
from rag_lab.parsing.parse import parse_pdf
from rag_lab.paths import artifacts_dir
from rag_lab.resources import ExperimentResource, MetricsStoreResource

# Below this many extracted characters per page, the PDF is almost certainly scanned images.
SCANNED_PDF_CHARS_PER_PAGE = 50


@asset(partitions_def=documents_partitions, group_name="ingestion")
def parsed_document(
    context: AssetExecutionContext, experiment: ExperimentResource, metrics: MetricsStoreResource
) -> MaterializeResult:
    """Docling parse of one PDF. Output goes to data/artifacts/<experiment>/parse/."""
    config = experiment.config()
    doc_id = context.partition_key
    path = scan_raw().get(doc_id)
    if path is None:
        raise Failure(f"No PDF in data/raw with content hash {doc_id} (changed or removed?)")

    store = metrics.store()
    config_hash = config.config_hash()
    store.upsert_experiment(config_hash, config.name, config.model_dump(mode="json"))

    parsed = parse_pdf(path, doc_id, config.parse, artifacts_dir(config.name, "parse"))
    if parsed.status == "failure":
        raise Failure(f"Docling failed on {path.name}: {parsed.errors}")
    if parsed.chars_per_page < SCANNED_PDF_CHARS_PER_PAGE:
        context.log.warning(
            f"{path.name}: only {parsed.chars_per_page} characters per page. This looks like a "
            "scanned PDF; OCR is off, so there is little text to ingest."
        )

    store.add_stage_metric(
        config_hash,
        doc_id,
        "parse",
        duration_ms=parsed.parse_seconds * 1000,
        items=parsed.pages,
        throughput=parsed.pages_per_second,
        details={
            "source_file": parsed.source_file,
            "status": parsed.status,
            "tables": parsed.tables,
            "table_cells": parsed.table_cells,
            "text_items": parsed.text_items,
            "chars_per_page": parsed.chars_per_page,
            "model_load_seconds": parsed.model_load_seconds,
            "errors": parsed.errors,
        },
        dagster_run_id=context.run_id,
    )

    return MaterializeResult(
        metadata={
            "experiment": config.name,
            "config_hash": config_hash,
            "source_file": parsed.source_file,
            "status": parsed.status,
            "pages": parsed.pages,
            "tables": parsed.tables,
            "text_items": parsed.text_items,
            "chars_per_page": parsed.chars_per_page,
            "parse_seconds": parsed.parse_seconds,
            "pages_per_second": parsed.pages_per_second,
            "model_load_seconds": parsed.model_load_seconds,
            "markdown_preview": MetadataValue.md(parsed.markdown_preview),
        }
    )
