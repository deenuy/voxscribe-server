-- Assistant profile per tenant (PL07): one mutable row per tenant schema,
-- persisted across restarts. Each tenant schema gets its own table instance so
-- tenant A's profile is physically separate from tenant B's.
-- revision increments on every PATCH so a session's profile_version changes.
-- The table is a singleton: at most one row per schema (enforced by the unique index).

CREATE TABLE IF NOT EXISTS assistant_profiles (
    singleton     BOOLEAN NOT NULL DEFAULT TRUE,
    name          TEXT NOT NULL DEFAULT 'vox',
    wake_phrase   TEXT NOT NULL DEFAULT 'hey vox',
    end_phrases   TEXT NOT NULL DEFAULT 'goodbye,stop listening',
    voice_id      TEXT NOT NULL DEFAULT '',
    speed         REAL NOT NULL DEFAULT 1.0,
    system_prompt TEXT NOT NULL DEFAULT '',
    persona_label TEXT NOT NULL DEFAULT 'assistant',
    revision      INTEGER NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT assistant_profiles_one_row UNIQUE (singleton)
);
