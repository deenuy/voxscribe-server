-- Tool calls as audit rows and receipts (PL43, TOOLS-CONNECTORS.md section 2.6). Every call through
-- the tool runtime writes one row, whatever its outcome (status, migration 024). A row with status
-- executed is a receipt: the gate lets a sentence claim an action only when a receipt of that
-- action's type exists for the turn that speaks it.
--   tool_version     the tool record's hash (app/local/tools/spec.py)
--   side_effect      read, write or irreversible, from the record; only writes back a claim
--   connector        the connector a tool came from; NULL for the engine's own tools
--   args             the validated arguments as the record's audit list keeps them: stored
--                    fields as they are, the others as a digest and a length
--   args_hash        sha256 of every validated argument
--   confirm_turn_id  the turn whose "yes" confirmed the call (read-back, PL44 and PL45)
--   result_ref       a reference to what the call made (a reminder's id), never its text
--   policy_version   the response gate's policy version in effect
--   finished_at      when the call ended, ISO 8601 like called_at
--   tenant           the tenant in scope when the call ran; NULL with isolation off
-- meeting_id is the session id. A thinker's call has a task id and may have no turn.

ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS tool_version TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS side_effect TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS connector TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS args JSONB;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS args_hash TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS confirm_turn_id TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS result_ref TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS policy_version TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS finished_at TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS tenant TEXT;
ALTER TABLE tool_calls ALTER COLUMN turn_id DROP NOT NULL;

CREATE INDEX IF NOT EXISTS tool_calls_receipts_idx
    ON tool_calls(meeting_id, turn_id) WHERE status = 'executed';
