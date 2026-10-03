from dagster import AssetExecutionContext, Failure, MaterializeResult, MetadataValue, asset

from rag_lab.assets.partitions import documents_partitions
from rag_lab.documents import scan_raw
from rag_lab.ingest import SCANNED_PDF_CHARS_PER_PAGE, record_parse, register_experiment
from rag_lab.parsing.parse import parse_pdf
from rag_lab.paths import artifacts_dir
from rag_lab.resources import ExperimentResource, MetricsStoreResource


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
    config_hash = register_experiment(store, config)

    parsed = parse_pdf(path, doc_id, config.parse, artifacts_dir(config.name, "parse"))
    if parsed.chars_per_page < SCANNED_PDF_CHARS_PER_PAGE:
        context.log.warning(
            f"{path.name}: only {parsed.chars_per_page} characters per page. This looks like a "
            "scanned PDF; OCR is off, so there is little text to ingest."
        )

    record_parse(store, config, doc_id, parsed, context.run_id)

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
