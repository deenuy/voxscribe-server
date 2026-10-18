-- Tool calls log: one row per tool invocation during a reply turn.
-- turn_id references turns(id); meeting_id is denormalised for fast meeting-scoped queries.

CREATE TABLE IF NOT EXISTS tool_calls (
    id          TEXT PRIMARY KEY,
    turn_id     TEXT NOT NULL,
    meeting_id  TEXT NOT NULL,
    tool        TEXT NOT NULL,
    query       TEXT NOT NULL,
    engine      TEXT,
    result_count INTEGER,
    duration_ms INTEGER,
    error       TEXT,
    called_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS tool_calls_meeting_idx ON tool_calls(meeting_id);
CREATE INDEX IF NOT EXISTS tool_calls_turn_idx ON tool_calls(turn_id);
