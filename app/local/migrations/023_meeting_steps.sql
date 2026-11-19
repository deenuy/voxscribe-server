-- The steps a scribe goes through after Stop (PL36): transcript, summary, tasks. One row per meeting and step,
-- rewritten on every change, so GET /v1/meetings/{id}/pipeline can say what happened after the event went by.
-- `reason` is a short code (app/local/pipeline.py REASONS), never text from the model or the transcript: the check
-- keeps it to lower-case letters and underscores. The summary's state itself stays in summaries.status; its row
-- here only carries the reason. Adds a table only.

CREATE TABLE IF NOT EXISTS meeting_steps (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    step       TEXT NOT NULL CHECK (step IN ('transcript', 'summary', 'tasks')),
    state      TEXT NOT NULL CHECK (state IN ('working', 'ready', 'failed', 'skipped')),
    reason     TEXT CHECK (reason IS NULL OR reason ~ '^[a-z_]{1,40}$'),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (meeting_id, step)
);
