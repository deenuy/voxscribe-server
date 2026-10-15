-- wake_events: one row per wake-word trigger, linked to the session it started.
-- started_meeting_id is NULL when the trigger fired but session creation failed.
-- consent columns added to meetings so every wake-originated session carries them (B1.1, D6).

CREATE TABLE wake_events (
    id               TEXT PRIMARY KEY,
    phrase           TEXT NOT NULL,
    confidence       REAL NOT NULL,
    triggered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL
);

CREATE INDEX ix_wake_events_triggered ON wake_events(triggered_at DESC);
CREATE INDEX ix_wake_events_meeting   ON wake_events(started_meeting_id)
    WHERE started_meeting_id IS NOT NULL;

-- Consent columns (B1.1, D6): present on every meeting after this migration.
ALTER TABLE meetings
    ADD COLUMN IF NOT EXISTS consent_version TEXT,
    ADD COLUMN IF NOT EXISTS consent_at      TIMESTAMPTZ;
