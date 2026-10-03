-- The Phase 11 clean-out: the benchmark machinery keyed by experiment and query-set file goes. The
-- experiments, ingestion_stage_metrics and search_log tables stay; reports have their own tables (0006).
DROP VIEW IF EXISTS experiment_summary;
DROP TABLE IF EXISTS benchmark_metrics;
DROP TABLE IF EXISTS query_results;
DROP TABLE IF EXISTS benchmark_runs;
DROP TABLE IF EXISTS eval_queries;
