-- One row per chatbot turn: the question, the query it was made into, what was retrieved and said, and
-- how long each step took. It goes with its experiment, like the search log.
CREATE TABLE chat_turns (
    id           bigserial PRIMARY KEY,
    session_id   text NOT NULL,
    config_hash  text NOT NULL REFERENCES experiments ON DELETE CASCADE,
    question     text NOT NULL,
    query        text NOT NULL,
    model        text NOT NULL,
    think        boolean NOT NULL,
    answer       text NOT NULL,
    thinking     text NOT NULL,
    hits         jsonb NOT NULL,
    timings      jsonb NOT NULL,
    model_states jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chat_turns_session ON chat_turns (session_id, created_at);
