-- Let the database refuse values the code never writes, and make "one recording at a time" a database fact.

ALTER TABLE meetings ADD CONSTRAINT meetings_status_check
    CHECK (status IN ('scheduled', 'recording', 'processing', 'done'));
ALTER TABLE transcript_segments ADD CONSTRAINT transcript_segments_state_check
    CHECK (state IN ('provisional', 'stable'));
ALTER TABLE transcript_segments ADD CONSTRAINT transcript_segments_source_check
    CHECK (source IN ('mic', 'system'));

-- At most one meeting may be recording. Starting is also serialised in code; this is the backstop.
CREATE UNIQUE INDEX IF NOT EXISTS ux_meetings_single_recording ON meetings ((true)) WHERE status = 'recording';
