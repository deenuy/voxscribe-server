-- Tasks: a to-do list per scribe, written by hand or drafted by the AI from the transcript.
-- A task belongs to one scribe; its project is the scribe's project (read through the scribe, so moving a scribe
-- to another project moves its tasks with it). Deleting a scribe deletes its tasks.
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    title       TEXT NOT NULL CHECK (length(btrim(title)) BETWEEN 1 AND 300),
    detail      TEXT CHECK (detail IS NULL OR length(detail) <= 2000),
    status      TEXT NOT NULL DEFAULT 'todo' CHECK (status IN ('todo', 'done')),
    source      TEXT NOT NULL CHECK (source IN ('manual', 'ai')),
    -- The title in lower case with single spaces: what "the same task" means, so a second Generate (or two at once)
    -- cannot add it twice. The database enforces it; the code only reports it.
    title_key   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_tasks_meeting_title ON tasks (meeting_id, title_key);
CREATE INDEX IF NOT EXISTS ix_tasks_meeting ON tasks (meeting_id, created_at);
CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (status, created_at DESC);
