from dagster import DynamicPartitionsDefinition

# One partition per document, keyed by content hash. assets/sensors.py registers them.
documents_partitions = DynamicPartitionsDefinition(name="documents")
