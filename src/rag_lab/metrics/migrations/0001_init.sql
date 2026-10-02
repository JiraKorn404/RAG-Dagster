CREATE TABLE experiments (
    config_hash text PRIMARY KEY,
    name        text NOT NULL,
    config      jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE ingestion_stage_metrics (
    id             bigserial PRIMARY KEY,
    config_hash    text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    doc_id         text NOT NULL,
    stage          text NOT NULL CHECK (stage IN ('parse', 'chunk', 'embed', 'index')),
    dagster_run_id text,
    duration_ms    double precision NOT NULL,
    items          integer,
    throughput     double precision,
    details        jsonb NOT NULL DEFAULT '{}',
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eval_queries (
    query_set_version text NOT NULL,
    query_id          text NOT NULL,
    query_text        text NOT NULL,
    expected          jsonb,
    PRIMARY KEY (query_set_version, query_id)
);

CREATE TABLE benchmark_runs (
    benchmark_run_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    config_hash            text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    dagster_run_id         text,
    top_k                  integer NOT NULL,
    repeats                integer NOT NULL,
    query_set_version      text,
    collection_point_count bigint,
    started_at             timestamptz NOT NULL DEFAULT now(),
    finished_at            timestamptz
);

CREATE TABLE query_results (
    id                  bigserial PRIMARY KEY,
    benchmark_run_id    uuid NOT NULL REFERENCES benchmark_runs ON DELETE CASCADE,
    query_id            text NOT NULL,
    repeat              integer NOT NULL,
    embed_ms            double precision NOT NULL,
    search_ms           double precision NOT NULL,
    total_ms            double precision NOT NULL,
    hits                jsonb NOT NULL,
    first_relevant_rank integer
);

-- Long format, so a new metric needs no schema change. value is null when it cannot be computed
-- (quality metrics before eval/queries.yaml has expected results).
CREATE TABLE benchmark_metrics (
    id               bigserial PRIMARY KEY,
    benchmark_run_id uuid NOT NULL REFERENCES benchmark_runs ON DELETE CASCADE,
    metric           text NOT NULL,
    k                integer,
    modality         text,
    value            double precision
);

CREATE TABLE search_log (
    id             bigserial PRIMARY KEY,
    config_hash    text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    query_text     text NOT NULL,
    top_k          integer NOT NULL,
    embed_ms       double precision NOT NULL,
    search_ms      double precision NOT NULL,
    total_ms       double precision NOT NULL,
    top_similarity double precision,
    created_at     timestamptz NOT NULL DEFAULT now()
);
