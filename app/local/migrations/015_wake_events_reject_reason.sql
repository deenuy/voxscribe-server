-- wake_events records refused wakes too: accepted is false and reject_reason says why
-- (first reason: quiet_hours, T06). Rows written before this migration were all accepted.
-- IF NOT EXISTS so a later migration that adds the same columns (wake gate, T02) is harmless.

ALTER TABLE wake_events
    ADD COLUMN IF NOT EXISTS accepted      BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS reject_reason TEXT;
