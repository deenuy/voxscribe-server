-- Revisions on everything a person can edit (so two editors cannot silently overwrite each other), a flag for a summary
-- a person wrote or changed, and a record of committed edit sessions so a retried commit is applied once.
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS revision integer NOT NULL DEFAULT 0;
ALTER TABLE summaries ADD COLUMN IF NOT EXISTS revision integer NOT NULL DEFAULT 0;
ALTER TABLE summaries ADD COLUMN IF NOT EXISTS edited boolean NOT NULL DEFAULT false;
ALTER TABLE summaries ADD COLUMN IF NOT EXISTS edited_at timestamptz;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS revision integer NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS edit_commits (
    idempotency_key text PRIMARY KEY,
    meeting_id      text NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    response        jsonb NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_edit_commits_created ON edit_commits (created_at);
