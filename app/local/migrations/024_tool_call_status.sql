-- A tool call's outcome (PL41). A call the network decision refused is recorded too, before
-- anything is sent:
--   status      executed, failed, timed_out, cancelled, invalid_args, refused
--               (app/local/tools/audit.py). NULL on rows written before this migration
--   error_code  a code, never text: why the call was refused or failed (for example
--               mode_restricted, privacy_offline, profile_network_off, unavailable)
-- A refused call never ran, so it has no query: `query` may now be empty.

ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS error_code TEXT;
ALTER TABLE tool_calls ALTER COLUMN query DROP NOT NULL;
