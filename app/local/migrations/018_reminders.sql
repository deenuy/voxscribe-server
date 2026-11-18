-- Reminders (T16): native reminder and timer support backed by the local DB.
--   text      what the user asked to be reminded about
--   fire_at   when to fire (UTC)
--   status    pending, fired, cancelled

CREATE TABLE IF NOT EXISTS reminders (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    fire_at     TIMESTAMPTZ NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reminders_fire_at ON reminders(fire_at) WHERE status = 'pending';
