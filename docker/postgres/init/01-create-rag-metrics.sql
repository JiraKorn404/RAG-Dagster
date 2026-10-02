-- Runs once, when the Postgres data volume is first created.
-- The `dagster` database is created by POSTGRES_DB; this adds ours beside it.
CREATE DATABASE rag_metrics;
