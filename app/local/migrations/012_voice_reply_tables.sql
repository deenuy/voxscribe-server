-- Voice reply pipeline tables: turns, latency_ledger, replies, gate_decisions.
-- Part of the R6 streaming chain.

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT NOT NULL PRIMARY KEY,
    meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    turn_n      INTEGER NOT NULL,
    user_text   TEXT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    UNIQUE (meeting_id, turn_n)
);

CREATE INDEX IF NOT EXISTS turns_meeting ON turns (meeting_id);

CREATE TABLE IF NOT EXISTS latency_ledger (
    id                  TEXT NOT NULL PRIMARY KEY,
    meeting_id          TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    turn_n              INTEGER NOT NULL,
    wake_ms             INTEGER,
    turn_end_ms         INTEGER,
    llm_first_token_ms  INTEGER,
    tts_first_audio_ms  INTEGER,
    playback_end_ms     INTEGER,
    recorded_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS latency_meeting ON latency_ledger (meeting_id);

CREATE TABLE IF NOT EXISTS replies (
    id            TEXT NOT NULL PRIMARY KEY,
    meeting_id    TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    turn_n        INTEGER NOT NULL,
    text          TEXT NOT NULL,
    model         TEXT,
    gate_category TEXT NOT NULL DEFAULT 'pass',
    played_at     TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS replies_meeting ON replies (meeting_id);

CREATE TABLE IF NOT EXISTS gate_decisions (
    id           TEXT NOT NULL PRIMARY KEY,
    reply_id     TEXT NOT NULL REFERENCES replies(id) ON DELETE CASCADE,
    meeting_id   TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    mode         TEXT NOT NULL DEFAULT 'default',
    input_text   TEXT,
    decision     TEXT NOT NULL,
    category     TEXT,
    decided_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS gate_decisions_meeting ON gate_decisions (meeting_id);
