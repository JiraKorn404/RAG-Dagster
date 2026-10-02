-- One row per experiment: ingestion performance (latest run of each stage per document) and the
-- latest finished benchmark. Compare experiments with: SELECT * FROM experiment_summary;
CREATE VIEW experiment_summary AS
WITH latest AS (
    SELECT DISTINCT ON (config_hash, doc_id, stage) *
    FROM ingestion_stage_metrics
    ORDER BY config_hash, doc_id, stage, created_at DESC
),
ingest AS (
    SELECT
        config_hash,
        count(DISTINCT doc_id) AS documents,
        sum(items) FILTER (WHERE stage = 'parse') AS pages,
        sum(duration_ms) FILTER (WHERE stage = 'parse') / 1000 AS parse_seconds,
        sum(items) FILTER (WHERE stage = 'chunk') AS chunks,
        sum(items) FILTER (WHERE stage = 'embed') AS embedded_chunks,
        sum(duration_ms) FILTER (WHERE stage = 'embed') / 1000 AS embed_seconds,
        sum((details ->> 'prompt_tokens')::double precision) FILTER (WHERE stage = 'embed') AS embed_tokens,
        max((details ->> 'model_load_ms')::double precision) FILTER (WHERE stage = 'embed') AS model_load_ms,
        sum(items) FILTER (WHERE stage = 'index') AS points,
        sum((details ->> 'upsert_ms')::double precision) FILTER (WHERE stage = 'index') / 1000 AS upsert_seconds,
        sum(duration_ms) / 1000 AS total_seconds
    FROM latest
    GROUP BY config_hash
),
last_run AS (
    SELECT DISTINCT ON (config_hash) benchmark_run_id, config_hash, started_at, top_k
    FROM benchmark_runs
    WHERE finished_at IS NOT NULL
    ORDER BY config_hash, started_at DESC
),
bench AS (
    SELECT
        r.config_hash,
        r.started_at,
        r.top_k,
        max(m.value) FILTER (WHERE m.metric = 'total_ms_p50') AS search_p50_ms,
        max(m.value) FILTER (WHERE m.metric = 'total_ms_p95') AS search_p95_ms,
        max(m.value) FILTER (WHERE m.metric = 'queries_per_second') AS queries_per_second,
        max(m.value) FILTER (WHERE m.metric = 'top1_similarity_mean') AS top1_similarity,
        max(m.value) FILTER (WHERE m.metric = 'point_count') AS point_count,
        max(m.value) FILTER (WHERE m.metric = 'recall' AND m.k = 5 AND m.modality IS NULL) AS recall_at_5,
        max(m.value) FILTER (WHERE m.metric = 'mrr' AND m.modality IS NULL) AS mrr,
        max(m.value) FILTER (WHERE m.metric = 'ndcg' AND m.k = 5 AND m.modality IS NULL) AS ndcg_at_5
    FROM last_run r
    JOIN benchmark_metrics m USING (benchmark_run_id)
    GROUP BY r.config_hash, r.started_at, r.top_k
)
SELECT
    e.name,
    e.config_hash,
    i.documents,
    i.pages,
    round((i.parse_seconds)::numeric, 2) AS parse_seconds,
    round((i.pages / nullif(i.parse_seconds, 0))::numeric, 2) AS parse_pages_per_second,
    i.chunks,
    round((i.chunks::numeric / nullif(i.documents, 0)), 1) AS chunks_per_document,
    round((i.embedded_chunks / nullif(i.embed_seconds, 0))::numeric, 2) AS embed_chunks_per_second,
    round((i.embed_tokens / nullif(i.embed_seconds, 0))::numeric, 1) AS embed_tokens_per_second,
    round(i.model_load_ms::numeric, 1) AS model_load_ms,
    round((i.points / nullif(i.upsert_seconds, 0))::numeric, 1) AS index_points_per_second,
    round((i.total_seconds / nullif(i.documents, 0))::numeric, 2) AS seconds_per_document,
    b.started_at AS benchmarked_at,
    b.top_k,
    b.point_count,
    round(b.search_p50_ms::numeric, 1) AS search_p50_ms,
    round(b.search_p95_ms::numeric, 1) AS search_p95_ms,
    round(b.queries_per_second::numeric, 2) AS queries_per_second,
    round(b.top1_similarity::numeric, 3) AS top1_similarity,
    round(b.recall_at_5::numeric, 3) AS recall_at_5,
    round(b.mrr::numeric, 3) AS mrr,
    round(b.ndcg_at_5::numeric, 3) AS ndcg_at_5
FROM experiments e
LEFT JOIN ingest i USING (config_hash)
LEFT JOIN bench b USING (config_hash)
ORDER BY e.created_at;
