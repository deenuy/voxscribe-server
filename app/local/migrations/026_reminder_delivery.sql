-- Reminders and timers as registry tools (PL45) and the due-item scheduler (PL46),
-- TOOLS-CONNECTORS.md sections 2.5 and 2.7. Adds to 019's table; nothing is removed.
--
--   kind                reminder | timer. Rows written before this migration are reminders
--   time_zone           the IANA zone the time was said in (the device's, settings.vox_timezone
--                       or the operating system's); a due time is shown and spoken in it
--   label               a timer's length as spoken ("ten minute"); NULL for a reminder. A timer
--                       has no text of the person's: its `text` is empty
--   created_session_id  the voice session and turn whose yes (a reminder) or request (a timer)
--   created_turn_id     made the row
--   receipt_id          the tool_calls row (status executed) that is this row's receipt
--   claimed_at          when the scheduler took the row to deliver it (status firing); a row
--                       is claimed by one UPDATE, so two deliverers never both get it
--   fired_at            when it was delivered (status fired or missed)
--   spoken              its line was played on the device
--   speak_owed          it was delivered as a notification only (quiet hours, another tenant
--                       bound, a person speaking): its line is spoken at the next session
--   missed              it came due while the engine was not running
--   cancelled_at        when it was cancelled, by voice or by DELETE /v1/reminders/{id}
--
-- status (019) gains two values: firing (claimed, being delivered) and missed (came due while
-- the engine was not running: delivered once at the next start inside the configured window,
-- or only listed after it). pending, fired and cancelled keep their meaning.

ALTER TABLE reminders ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'reminder';
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS time_zone TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS label TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS created_session_id TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS created_turn_id TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS receipt_id TEXT;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS fired_at TIMESTAMPTZ;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS spoken BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS speak_owed BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS missed BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS reminders_owed ON reminders(fired_at) WHERE speak_owed;
CREATE INDEX IF NOT EXISTS reminders_firing ON reminders(claimed_at) WHERE status = 'firing';
