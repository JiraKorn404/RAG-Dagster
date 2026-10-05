-- Database answers a user marked as good: the question (as typed and made standalone) and the SQL that
-- produced the answer. They are shown to the text-to-SQL agent when a similar question comes later, and
-- indexed in Qdrant (rag_lab/sql/examples.py). An example belongs to its schema and goes with it. It
-- keeps the turn it came from for provenance, but outlives that turn's chat.
CREATE TABLE sql_examples (
    id          bigserial PRIMARY KEY,
    schema_name text NOT NULL REFERENCES db_schemas ON DELETE CASCADE,
    question    text NOT NULL,
    standalone  text NOT NULL,
    sql         text NOT NULL,
    turn_id     bigint REFERENCES chat_turns ON DELETE SET NULL,
    enabled     boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The same question and SQL is kept once (hashed: a long query does not fit in an index entry).
CREATE UNIQUE INDEX sql_examples_once ON sql_examples (schema_name, md5(standalone), md5(sql));
