-- L1: store prompt version alongside every generated summary
ALTER TABLE summaries ADD COLUMN IF NOT EXISTS prompt_version TEXT;

-- P4: idempotent meeting creation via a caller-supplied key
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS client_request_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS ux_meetings_client_request_id
    ON meetings(client_request_id) WHERE client_request_id IS NOT NULL;
