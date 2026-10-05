-- What the app has imported into the database for tables (`rag_data`, which is a separate database: a
-- foreign key cannot cross it, so these rows are kept in step with it by rag_lab/sql/database.py).
-- A schema is one dataset; a table's columns are described here: type, description, and a profile of
-- its values (null count, distinct count, example values, range).
CREATE TABLE db_schemas (
    schema_name text PRIMARY KEY,
    description text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE db_tables (
    schema_name text NOT NULL REFERENCES db_schemas ON DELETE CASCADE,
    table_name  text NOT NULL,
    source_file text,
    imported_at timestamptz NOT NULL DEFAULT now(),
    row_count   bigint NOT NULL DEFAULT 0,
    description text NOT NULL DEFAULT '',
    columns     jsonb NOT NULL DEFAULT '[]',
    PRIMARY KEY (schema_name, table_name)
);

-- A database chat searches one schema and has no experiment; a documents chat searches one experiment
-- and has no schema. Deleting a schema deletes its chats.
ALTER TABLE chat_sessions
    ADD COLUMN schema_name text REFERENCES db_schemas ON DELETE CASCADE,
    DROP CONSTRAINT chat_sessions_documents_have_experiment,
    ADD CONSTRAINT chat_sessions_one_target CHECK (
        (kind = 'documents' AND config_hash IS NOT NULL AND schema_name IS NULL)
        OR (kind = 'database' AND schema_name IS NOT NULL AND config_hash IS NULL)
    );
