from dagster import Definitions, EnvVar

from rag_lab.assets.chunking import chunks
from rag_lab.assets.indexing import embeddings, qdrant_index
from rag_lab.assets.jobs import ingest_job
from rag_lab.assets.parsing import parsed_document
from rag_lab.assets.partitions import new_pdf_sensor
from rag_lab.resources import (
    ExperimentResource,
    MetricsStoreResource,
    OllamaResource,
    QdrantResource,
)

defs = Definitions(
    assets=[parsed_document, chunks, embeddings, qdrant_index],
    jobs=[ingest_job],
    sensors=[new_pdf_sensor],
    resources={
        # Set per run in the launchpad: resources.experiment.config (name, parse, chunk, ...)
        "experiment": ExperimentResource.configure_at_launch(),
        "ollama": OllamaResource(base_url=EnvVar("OLLAMA_BASE_URL")),
        "qdrant": QdrantResource(url=EnvVar("QDRANT_URL")),
        "metrics": MetricsStoreResource(database_url=EnvVar("METRICS_DATABASE_URL")),
    },
)
