-- Transcript editing, a revision history, full-text search over transcript text, and the metadata a
-- meeting record carries once it is published.
--
-- The transcript is stored as one row per segment, never as one long text: a four-hour meeting is a few
-- thousand short rows that can be streamed, edited one at a time (a single row lock, a small write),
-- searched, cited by id and re-indexed individually. The long text is derived on read (export, summary).

-- ============================================================================
-- Meeting metadata
-- ============================================================================
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS project_id text REFERENCES projects(id) ON DELETE SET NULL;
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS origin text NOT NULL DEFAULT 'recorded'
    CHECK (origin IN ('recorded', 'text', 'file', 'youtube'));
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS language text;
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS published_at timestamptz;
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS word_count integer NOT NULL DEFAULT 0;
-- Bumped by every transcript edit; lets a client see that what it holds is out of date.
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS transcript_revision integer NOT NULL DEFAULT 0;
-- True once the transcript changed after the summary was generated.
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS summary_stale boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS ix_meetings_project ON meetings (project_id) WHERE project_id IS NOT NULL;

-- ============================================================================
-- Transcript segments
-- ============================================================================
ALTER TABLE transcript_segments ADD COLUMN IF NOT EXISTS revision integer NOT NULL DEFAULT 0;
ALTER TABLE transcript_segments ADD COLUMN IF NOT EXISTS edited_at timestamptz;
ALTER TABLE transcript_segments ADD COLUMN IF NOT EXISTS search tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED;

CREATE INDEX IF NOT EXISTS ix_transcript_meeting_start ON transcript_segments (meeting_id, start_ms);
CREATE INDEX IF NOT EXISTS ix_transcript_search ON transcript_segments USING gin (search);

-- Every edit keeps what was there before it, so nothing the speech recogniser produced is ever lost.
CREATE TABLE IF NOT EXISTS transcript_edits (
    id             bigserial PRIMARY KEY,
    meeting_id     text NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    segment_id     text NOT NULL,
    revision       integer NOT NULL,
    before_text    text NOT NULL,
    after_text     text NOT NULL,
    before_speaker text,
    after_speaker  text,
    edited_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_transcript_edits_segment ON transcript_edits (meeting_id, segment_id, revision);

-- Existing meetings: word counts, and published time for those already done.
UPDATE meetings m SET word_count = COALESCE((
    SELECT sum(cardinality(regexp_split_to_array(btrim(s.text), '\s+')))
    FROM transcript_segments s WHERE s.meeting_id = m.id AND btrim(s.text) <> ''
), 0);
UPDATE meetings SET published_at = updated_at WHERE status = 'done' AND published_at IS NULL;
