from dagster import (
    DefaultSensorStatus,
    DynamicPartitionsDefinition,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    sensor,
)

from rag_lab.documents import scan_raw

# One partition per document, keyed by content hash.
documents_partitions = DynamicPartitionsDefinition(name="documents")


@sensor(minimum_interval_seconds=30, default_status=DefaultSensorStatus.RUNNING)
def new_pdf_sensor(context: SensorEvaluationContext):
    """Registers a partition for each PDF in data/raw that has not been seen. It starts no runs."""
    known = set(context.instance.get_dynamic_partitions(documents_partitions.name))
    new = sorted(set(scan_raw()) - known)
    if not new:
        return SkipReason("No new PDFs")
    return SensorResult(dynamic_partitions_requests=[documents_partitions.build_add_request(new)])
