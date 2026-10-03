"""Writes to the rag_metrics database, one method per table. Plain Python, no Dagster."""

import psycopg
from psycopg.types.json import Jsonb


class MetricsStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _execute(self, sql: str, params: tuple | list, many: bool = False):
        """Runs one statement in its own transaction; returns the first row if it has RETURNING."""
        with psycopg.connect(self.database_url) as conn:
            if many:
                with conn.cursor() as cur:
                    cur.executemany(sql, params)
                return None
            cur = conn.execute(sql, params)
            return cur.fetchone() if "RETURNING" in sql else None

    def upsert_experiment(self, config_hash: str, name: str, config: dict) -> None:
        self._execute(
            """INSERT INTO experiments (config_hash, name, config) VALUES (%s, %s, %s)
               ON CONFLICT (config_hash) DO UPDATE SET name = EXCLUDED.name, config = EXCLUDED.config""",
            (config_hash, name, Jsonb(config)),
        )

    def get_experiment(self, name: str) -> tuple[str, dict] | None:
        """(config_hash, config) of the newest config recorded under this experiment name."""
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT config_hash, config FROM experiments WHERE name = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def add_stage_metric(
        self,
        config_hash: str,
        doc_id: str,
        stage: str,
        duration_ms: float,
        items: int | None = None,
        throughput: float | None = None,
        details: dict | None = None,
        dagster_run_id: str | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO ingestion_stage_metrics
                   (config_hash, doc_id, stage, dagster_run_id, duration_ms, items, throughput, details)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (config_hash, doc_id, stage, dagster_run_id, duration_ms, items, throughput,
             Jsonb(details or {})),
        )

    def list_experiments(self) -> list[dict]:
        """The newest row of each experiment name, with how many documents it holds (distinct documents
        in its stage metrics) and whether it belongs to a benchmark report."""
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (e.name) e.name, e.config_hash, e.config,
                          (SELECT count(DISTINCT m.doc_id) FROM ingestion_stage_metrics m
                           WHERE m.config_hash = e.config_hash) AS documents,
                          EXISTS (SELECT 1 FROM benchmark_results r
                                  WHERE r.experiment = e.name) AS benchmarked
                   FROM experiments e ORDER BY e.name, e.created_at DESC"""
            ).fetchall()
        return [
            {"name": r[0], "config_hash": r[1], "config": r[2], "documents": r[3], "benchmarked": r[4]}
            for r in rows
        ]

    def add_search_log(
        self,
        config_hash: str,
        query_text: str,
        top_k: int,
        embed_ms: float,
        search_ms: float,
        total_ms: float,
        top_similarity: float | None,
    ) -> None:
        self._execute(
            """INSERT INTO search_log (config_hash, query_text, top_k, embed_ms, search_ms,
                                       total_ms, top_similarity)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (config_hash, query_text, top_k, embed_ms, search_ms, total_ms, top_similarity),
        )

    # --- benchmark reports -----------------------------------------------------------------------

    def create_report(
        self,
        report_id: str,
        document_id: str,
        document_name: str,
        settings: dict,
        test_queries: list[dict] | None,
        probes: list[str],
        document_info: dict,
        total_experiments: int,
    ) -> None:
        self._execute(
            """INSERT INTO benchmark_reports (report_id, document_id, document_name, status, settings,
                                              test_queries, probes, document_info, total_experiments)
               VALUES (%s, %s, %s, 'running', %s, %s, %s, %s, %s)""",
            (report_id, document_id, document_name, Jsonb(settings),
             Jsonb(test_queries) if test_queries else None, Jsonb(probes), Jsonb(document_info),
             total_experiments),
        )

    def touch_report(self, report_id: str) -> None:
        """Heartbeat: a running report whose updated_at stops moving is shown as interrupted."""
        self._execute("UPDATE benchmark_reports SET updated_at = now() WHERE report_id = %s", (report_id,))

    def finish_report(self, report_id: str, status: str, error: str | None = None) -> None:
        self._execute(
            """UPDATE benchmark_reports SET status = %s, error = %s, updated_at = now(), finished_at = now()
               WHERE report_id = %s""",
            (status, error, report_id),
        )

    def request_stop(self, report_id: str) -> None:
        self._execute("UPDATE benchmark_reports SET stop_requested = true WHERE report_id = %s", (report_id,))

    def stop_requested(self, report_id: str) -> bool:
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT stop_requested FROM benchmark_reports WHERE report_id = %s", (report_id,)
            ).fetchone()
        return bool(row and row[0])

    def save_result(
        self,
        report_id: str,
        experiment: str,
        model: str,
        strategy: str,
        status: str,
        metrics: dict,
        per_query: list[dict] | None = None,
        error: str | None = None,
    ) -> None:
        self._execute(
            """INSERT INTO benchmark_results (report_id, experiment, model, strategy, status, metrics,
                                              per_query, error)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (report_id, experiment) DO UPDATE SET status = EXCLUDED.status,
                   metrics = EXCLUDED.metrics, per_query = EXCLUDED.per_query, error = EXCLUDED.error""",
            (report_id, experiment, model, strategy, status, Jsonb(metrics),
             Jsonb(per_query) if per_query is not None else None, error),
        )
        self.touch_report(report_id)

    def mark_stale_reports_interrupted(self, minutes: int = 15) -> None:
        """A report still 'running' whose heartbeat is old belongs to a process that no longer exists."""
        self._execute(
            """UPDATE benchmark_reports SET status = 'interrupted', finished_at = now()
               WHERE status = 'running' AND updated_at < now() - make_interval(mins => %s)""",
            (minutes,),
        )

    def running_report(self) -> dict | None:
        self.mark_stale_reports_interrupted()
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT report_id FROM benchmark_reports WHERE status = 'running' LIMIT 1"
            ).fetchone()
        return self.get_report(row[0]) if row else None

    _REPORT_COLUMNS = (
        "report_id, document_id, document_name, status, settings, test_queries, probes, document_info, "
        "total_experiments, stop_requested, error, created_at, updated_at, finished_at"
    )

    def _report_row(self, row) -> dict:
        keys = [c.strip() for c in self._REPORT_COLUMNS.split(",")]
        return dict(zip(keys, row))

    def get_report(self, report_id: str) -> dict | None:
        self.mark_stale_reports_interrupted()
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                f"SELECT {self._REPORT_COLUMNS} FROM benchmark_reports WHERE report_id = %s", (report_id,)
            ).fetchone()
        return self._report_row(row) if row else None

    def list_reports(self) -> list[dict]:
        self.mark_stale_reports_interrupted()
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT {self._REPORT_COLUMNS} FROM benchmark_reports ORDER BY created_at DESC"
            ).fetchall()
        return [self._report_row(r) for r in rows]

    def get_results(self, report_id: str) -> list[dict]:
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """SELECT experiment, model, strategy, status, metrics, per_query, error, created_at
                   FROM benchmark_results WHERE report_id = %s ORDER BY created_at""",
                (report_id,),
            ).fetchall()
        keys = ["experiment", "model", "strategy", "status", "metrics", "per_query", "error", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def delete_experiment(self, name: str) -> None:
        """Remove an experiment's rows (its stage metrics go with it). The collection and files are
        removed by the caller."""
        self._execute("DELETE FROM experiments WHERE name = %s", (name,))
