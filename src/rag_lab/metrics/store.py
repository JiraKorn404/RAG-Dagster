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
               ON CONFLICT (config_hash) DO UPDATE SET name = EXCLUDED.name""",
            (config_hash, name, Jsonb(config)),
        )

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

    def sync_eval_queries(self, version: str, queries: list[dict]) -> None:
        """`queries`: dicts with id, query, and optional expected."""
        self._execute(
            """INSERT INTO eval_queries (query_set_version, query_id, query_text, expected)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (query_set_version, query_id)
               DO UPDATE SET query_text = EXCLUDED.query_text, expected = EXCLUDED.expected""",
            [
                (version, q["id"], q["query"],
                 Jsonb(q["expected"]) if q.get("expected") is not None else None)
                for q in queries
            ],
            many=True,
        )

    def start_benchmark_run(
        self,
        config_hash: str,
        top_k: int,
        repeats: int,
        query_set_version: str | None = None,
        dagster_run_id: str | None = None,
    ) -> str:
        row = self._execute(
            """INSERT INTO benchmark_runs
                   (config_hash, dagster_run_id, top_k, repeats, query_set_version)
               VALUES (%s, %s, %s, %s, %s) RETURNING benchmark_run_id""",
            (config_hash, dagster_run_id, top_k, repeats, query_set_version),
        )
        return str(row[0])

    def finish_benchmark_run(self, benchmark_run_id: str, collection_point_count: int) -> None:
        self._execute(
            """UPDATE benchmark_runs SET finished_at = now(), collection_point_count = %s
               WHERE benchmark_run_id = %s""",
            (collection_point_count, benchmark_run_id),
        )

    def add_query_results(self, benchmark_run_id: str, rows: list[dict]) -> None:
        """`rows`: dicts with query_id, repeat, embed_ms, search_ms, total_ms, hits,
        and optional first_relevant_rank."""
        self._execute(
            """INSERT INTO query_results (benchmark_run_id, query_id, repeat, embed_ms, search_ms,
                                          total_ms, hits, first_relevant_rank)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (benchmark_run_id, r["query_id"], r["repeat"], r["embed_ms"], r["search_ms"],
                 r["total_ms"], Jsonb(r["hits"]), r.get("first_relevant_rank"))
                for r in rows
            ],
            many=True,
        )

    def add_benchmark_metrics(self, benchmark_run_id: str, rows: list[dict]) -> None:
        """`rows`: dicts with metric, value (may be None), and optional k and modality."""
        self._execute(
            """INSERT INTO benchmark_metrics (benchmark_run_id, metric, k, modality, value)
               VALUES (%s, %s, %s, %s, %s)""",
            [
                (benchmark_run_id, r["metric"], r.get("k"), r.get("modality"), r.get("value"))
                for r in rows
            ],
            many=True,
        )

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
