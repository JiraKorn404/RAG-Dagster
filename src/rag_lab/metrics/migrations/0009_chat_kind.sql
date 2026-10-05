-- A chat searches one kind of thing for its whole life: the documents of an experiment, or (later) a
-- database schema. The kind is on the chat, and every turn repeats it under a composite foreign key, so
-- the database refuses a turn of the other kind. A database chat has no experiment, so config_hash may
-- be null; a documents chat still needs one.
ALTER TABLE chat_sessions
    ADD COLUMN kind text NOT NULL DEFAULT 'documents' CHECK (kind IN ('documents', 'database')),
    ALTER COLUMN config_hash DROP NOT NULL,
    ADD CONSTRAINT chat_sessions_documents_have_experiment CHECK (kind <> 'documents' OR config_hash IS NOT NULL),
    ADD CONSTRAINT chat_sessions_session_kind_key UNIQUE (session_id, kind);

-- Every turn saved so far belongs to a documents chat.
ALTER TABLE chat_turns
    ADD COLUMN kind text NOT NULL DEFAULT 'documents',
    ALTER COLUMN config_hash DROP NOT NULL,
    DROP CONSTRAINT chat_turns_session_fk,
    ADD CONSTRAINT chat_turns_session_kind_fk
        FOREIGN KEY (session_id, kind) REFERENCES chat_sessions (session_id, kind) ON DELETE CASCADE;

-- From here on a writer says which kind it means.
ALTER TABLE chat_sessions ALTER COLUMN kind DROP DEFAULT;
ALTER TABLE chat_turns ALTER COLUMN kind DROP DEFAULT;
