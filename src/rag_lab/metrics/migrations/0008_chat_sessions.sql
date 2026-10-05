-- A chat is a row here; its turns are the chat_turns rows that point at it. It is made with its first
-- turn and goes with its experiment. Deleting a chat is one delete: its turns go with it.
CREATE TABLE chat_sessions (
    session_id  text PRIMARY KEY,
    config_hash text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- The turns saved before this migration: one session each, from its first and last turn.
INSERT INTO chat_sessions (session_id, config_hash, title, created_at, updated_at)
SELECT first_turn.session_id, first_turn.config_hash, left(first_turn.question, 80),
       first_turn.created_at, last_turn.created_at
FROM (SELECT DISTINCT ON (session_id) session_id, config_hash, question, created_at
      FROM chat_turns ORDER BY session_id, created_at, id) AS first_turn
JOIN (SELECT session_id, max(created_at) AS created_at FROM chat_turns GROUP BY session_id) AS last_turn
  USING (session_id);

-- What a turn needs to be shown again and counted: its events in order (without the token-by-token
-- ones), whether it abstained, which passages it cited, how long it took, the settings it ran with and,
-- for a turn that failed, the error. Rows from before this migration have nulls here.
ALTER TABLE chat_turns
    ADD COLUMN events    jsonb,
    ADD COLUMN abstained boolean NOT NULL DEFAULT false,
    ADD COLUMN cited     jsonb,
    ADD COLUMN total_ms  double precision,
    ADD COLUMN settings  jsonb,
    ADD COLUMN error     text,
    ADD CONSTRAINT chat_turns_session_fk
        FOREIGN KEY (session_id) REFERENCES chat_sessions ON DELETE CASCADE;
