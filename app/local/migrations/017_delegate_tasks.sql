-- Delegate lane (T14): one row per background task, and the task id on every tool call.
--   status        queued, running, done, cancelled, delivered (app/local/voice/delegate_lane.py)
--   result        the text to deliver, set when the task is done; the failure line when the
--                 thinker failed or timed out
--   delivered_at  when the result, or the offer to run a stale one again, was spoken
--   task_id       the lane's task id, the value tool_calls.task_id carries
-- session_id is the wake session's meeting id. meetings.id is TEXT, so the reference is TEXT.

CREATE TABLE IF NOT EXISTS delegate_tasks (
    id            UUID PRIMARY KEY,
    session_id    TEXT REFERENCES meetings(id) ON DELETE CASCADE,
    request       TEXT NOT NULL,
    status        TEXT NOT NULL,
    result        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at  TIMESTAMPTZ,
    task_id       TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS delegate_tasks_session_idx ON delegate_tasks(session_id);

-- Every tool call is made by a delegate task (A5). Rows written before this migration came
-- from the reply path and keep NULL.
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS task_id TEXT;

CREATE INDEX IF NOT EXISTS tool_calls_task_idx ON tool_calls(task_id);
