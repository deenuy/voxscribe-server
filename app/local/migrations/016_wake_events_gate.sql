-- wake_events records every wake gate candidate (T02), accepted or not.
--   accepted       false when the gate refused the candidate
--   reject_reason  why: patience, no_voice, no_text, no_match, confirm_error (voxscribe.wake.gate);
--                  quiet_hours is added by T06
--   detector       stage 1 adapter, e.g. openvocab:hey_vox, oww:hey_vox, trigger
--   speech_ratio   stage 2 voice check result, NULL when the candidate stopped before it
-- confidence keeps the stage 1 score. Rows written before this migration were all accepted.
-- accepted and reject_reason are also added by 015 (T06); IF NOT EXISTS makes either merge order
-- safe, and this file is 016 so the two never share a version number.

ALTER TABLE wake_events
    ADD COLUMN IF NOT EXISTS accepted      BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS reject_reason TEXT,
    ADD COLUMN IF NOT EXISTS detector      TEXT,
    ADD COLUMN IF NOT EXISTS speech_ratio  REAL;
