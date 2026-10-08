from dagster import define_asset_job, in_process_executor

from rag_lab.assets.chunking import chunks
from rag_lab.assets.indexing import embeddings, qdrant_index
from rag_lab.assets.parsing import parsed_document
from rag_lab.assets.partitions import documents_partitions

# Parse -> chunk -> embed -> index for the selected document partitions. Run config is the shared
# `experiment` resource, set once for the whole run.
# The four steps run in the run's own process. With the default executor each step starts a process
# of its own, which imports Docling again: about 45 s a step, 170 s of a 204 s run for a two-page PDF.
ingest_job = define_asset_job(
    "ingest_job",
    selection=[parsed_document, chunks, embeddings, qdrant_index],
    partitions_def=documents_partitions,
    executor_def=in_process_executor,
)
