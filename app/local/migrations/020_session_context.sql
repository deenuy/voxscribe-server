-- Session context (PL05): a voice session belongs to a tenant, and its row says who and what it ran for.
-- A voice session is a meeting with origin 'wake', so the four facts are columns of meetings.
--   tenant           the tenant key the session ran in. NULL when tenant isolation is off (one shared scope)
--   product          the product the session served (the loaded profile's mode)
--   device           where the audio came from
--   profile_version  the profile in force when the session opened
-- All four are NULL on meetings that are not voice sessions and on voice sessions older than this migration.

ALTER TABLE meetings
    ADD COLUMN IF NOT EXISTS tenant          TEXT,
    ADD COLUMN IF NOT EXISTS product         TEXT,
    ADD COLUMN IF NOT EXISTS device          TEXT,
    ADD COLUMN IF NOT EXISTS profile_version TEXT;
