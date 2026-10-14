-- Allow meetings to be marked 'interrupted' when a server crash leaves them in 'recording'.
-- recover_unfinished() calls mark_interrupted() at startup; without this value the UPDATE
-- is rejected by the CHECK constraint and the meeting stays stuck in 'recording' forever.

ALTER TABLE meetings DROP CONSTRAINT IF EXISTS meetings_status_check;
ALTER TABLE meetings ADD CONSTRAINT meetings_status_check
    CHECK (status IN ('scheduled', 'recording', 'processing', 'done', 'interrupted'));
