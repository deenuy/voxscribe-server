-- Barge-in (T13): what the user heard of a reply that was interrupted.
--   barge_in     true when the user interrupted the reply while it played
--   played_text  the clauses that finished playing; replies.text holds the same text, since the
--                stored turn and the next turn's context are what was heard
--   full_text    every clause the reply stage released for speech, played or not
-- Both texts are NULL for a reply that played to its end: text is then the whole reply.
-- 018, not 017: t14-delegation adds 017_delegate_tasks.sql, and the runner applies only versions
-- above the newest one recorded, so the two must reach main in number order.

ALTER TABLE replies
    ADD COLUMN IF NOT EXISTS barge_in    BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS played_text TEXT,
    ADD COLUMN IF NOT EXISTS full_text   TEXT;
