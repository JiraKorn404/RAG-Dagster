from dagster import (
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    sensor,
)

from rag_lab.assets.jobs import ingest_job
from rag_lab.assets.partitions import documents_partitions
from rag_lab.config import load_experiment_file
from rag_lab.documents import doc_id_for, is_complete_pdf, raw_pdfs
from rag_lab.ingest import experiment_conflict
from rag_lab.paths import INGEST_CONFIG
from rag_lab.resources import MetricsStoreResource


@sensor(job=ingest_job, minimum_interval_seconds=30, default_status=DefaultSensorStatus.RUNNING)
def new_pdf_sensor(context: SensorEvaluationContext, metrics: MetricsStoreResource):
    """Starts `ingest_job` for each PDF that is new in data/raw, with the experiment written in
    config/ingest.yaml as the run's config. New means not yet a partition: a PDF that was already in
    the folder, or the same bytes under another file name, starts nothing. A file that is still being
    written waits for a later tick. The YAML is read at every tick that finds a new PDF; when it does
    not load, or names an experiment that exists with other settings, the tick fails with the reason
    and the PDF stays new, so it runs once the file is put right."""
    known = set(context.instance.get_dynamic_partitions(documents_partitions.name))
    waiting, new = [], {}
    for path in raw_pdfs():
        if not is_complete_pdf(path):
            waiting.append(path.name)
        elif (doc_id := doc_id_for(path)) not in known:
            new.setdefault(doc_id, path)
    note = ""
    if waiting:
        note = f" Waiting for {len(waiting)} file(s) that are not complete yet: {', '.join(waiting)}."
    if not new:
        return SkipReason("No new PDFs." + note)

    config = load_experiment_file(INGEST_CONFIG)
    problem = experiment_conflict(metrics.store(), config)
    if problem:
        names = ", ".join(path.name for path in new.values())
        raise ValueError(f"{INGEST_CONFIG}: {problem} Not started: {names}.")
    settings = config.model_dump(mode="json", exclude_none=True)
    context.log.info(f"{len(new)} new PDF(s) for the experiment '{config.name}'." + note)
    return SensorResult(
        dynamic_partitions_requests=[documents_partitions.build_add_request(list(new))],
        run_requests=[
            RunRequest(
                run_key=f"{config.name}:{doc_id}",
                partition_key=doc_id,
                run_config={"resources": {"experiment": {"config": settings}}},
                tags={"experiment": config.name, "source_file": path.name},
            )
            for doc_id, path in new.items()
        ],
    )
