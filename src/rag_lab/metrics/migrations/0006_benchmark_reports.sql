-- One row per benchmark report (a run over one document) and one per experiment in it. The results of
-- an experiment are read whole into the report, so its numbers are one jsonb value.
CREATE TABLE benchmark_reports (
    report_id         text PRIMARY KEY,
    document_id       text NOT NULL,
    document_name     text NOT NULL,
    status            text NOT NULL CHECK (status IN ('running', 'done', 'failed', 'stopped', 'interrupted')),
    settings          jsonb NOT NULL,
    test_queries      jsonb,
    probes            jsonb NOT NULL,
    document_info     jsonb NOT NULL,
    total_experiments integer NOT NULL,
    stop_requested    boolean NOT NULL DEFAULT false,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);

CREATE TABLE benchmark_results (
    report_id  text NOT NULL REFERENCES benchmark_reports ON DELETE CASCADE,
    experiment text NOT NULL,
    model      text NOT NULL,
    strategy   text NOT NULL,
    status     text NOT NULL CHECK (status IN ('done', 'failed')),
    metrics    jsonb NOT NULL DEFAULT '{}',
    per_query  jsonb,
    error      text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (report_id, experiment)
);
