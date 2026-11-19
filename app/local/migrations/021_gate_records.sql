-- Gate records become an append-only ledger (PL09, gap T7, parked row D17).
--
-- Before: one gate_decisions row per reply, inserted as "pending" and updated in place, with no
-- policy version and with up to a thousand characters of what the person said in input_text.
-- After: one gate_records row per event (policy_applied, heard, cue, decision, outcome), written
-- once. A row carries the policy version and, in place of the text, a hash and a length.
--
-- The session id is held by value with no foreign key: deleting a meeting does not delete the
-- record of what the gate decided in it, and no row holds anything the person said.
--
-- Expand only (README, "Migrations"): gate_decisions stays a table, so an engine on the
-- previous release can still insert into and update it. Its stored text is cleared here and a
-- trigger clears whatever older code writes there from now on. This release no longer writes to
-- it. Dropping it is the contract step, in a later migration of a later release.

CREATE TABLE IF NOT EXISTS gate_records (
    seq            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,   -- append order
    id             TEXT NOT NULL UNIQUE,
    event          TEXT NOT NULL CHECK (event IN
                       ('policy_applied', 'heard', 'cue', 'decision', 'outcome', 'legacy')),
    session_id     TEXT NOT NULL,       -- meetings.id, by value
    turn_id        TEXT,                -- turns.id, by value
    turn_n         INTEGER,
    reply_id       TEXT,                -- replies.id, by value
    step_n         INTEGER,             -- order of the evaluation within the reply
    mode           TEXT NOT NULL,       -- product mode
    policy_version TEXT NOT NULL,
    posture        TEXT,                -- policy_applied rows: allowlist | denylist
    policy_digest  TEXT,                -- policy_applied rows: SHA-256 of the pack file
    engine         TEXT,                -- policy_applied rows: engine build and floor version
    surface        TEXT,
    decision       TEXT,                -- pass | refuse | replace
    category       TEXT,                -- the short reason the host and its clients see
    rule           TEXT,                -- the rule that decided, as the engine names it
    input_hash     TEXT,                -- heard rows: SHA-256(turn id, newline, transcript text)
    input_len      INTEGER,             -- heard rows: characters heard
    candidate_hash TEXT,                -- cue and decision rows: the engine's candidate key
    candidate_len  INTEGER,
    cost_us        INTEGER,             -- time the evaluation took
    recorded_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS gate_records_session ON gate_records (session_id, seq);
CREATE INDEX IF NOT EXISTS gate_records_reply ON gate_records (reply_id) WHERE reply_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS gate_records_one_policy_per_session
    ON gate_records (session_id) WHERE event = 'policy_applied';

-- Every existing row is carried over with its id, verdict and time. The text is replaced by a
-- hash and a length, taken from the turn's transcript text where the turn row exists (the gate
-- row kept only the first thousand characters) and from the gate row's own copy otherwise.
-- These rows predate versioned policy packs, so their policy_version says so.
INSERT INTO gate_records
    (id, event, session_id, turn_id, turn_n, reply_id, mode, policy_version,
     decision, category, rule, input_hash, input_len, recorded_at)
SELECT g.id, 'legacy', g.meeting_id, t.id, r.turn_n, g.reply_id, g.mode, 'unversioned',
       g.decision, g.category, g.category,
       CASE WHEN COALESCE(t.user_text, g.input_text) IS NULL THEN NULL
            ELSE encode(sha256(convert_to(
                     COALESCE(t.id, '') || E'\n' || COALESCE(t.user_text, g.input_text),
                     'UTF8')), 'hex') END,
       char_length(COALESCE(t.user_text, g.input_text)),
       g.decided_at
FROM gate_decisions g
LEFT JOIN replies r ON r.id = g.reply_id
LEFT JOIN turns t ON t.meeting_id = g.meeting_id AND t.turn_n = r.turn_n
ORDER BY g.decided_at, g.id;

-- The old table keeps its rows and columns for the previous release, but no longer what the
-- person said: the carried-over rows lose their text, and older code that still writes text
-- there has it cleared on the way in.
UPDATE gate_decisions SET input_text = NULL WHERE input_text IS NOT NULL;

CREATE OR REPLACE FUNCTION gate_decisions_keep_no_text() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.input_text := NULL;
    RETURN NEW;
END;
$$;

CREATE TRIGGER gate_decisions_no_text
    BEFORE INSERT OR UPDATE ON gate_decisions
    FOR EACH ROW EXECUTE FUNCTION gate_decisions_keep_no_text();

-- One row per reply, the verdict the host acted on, and no text column: what tools read
-- (tools/roomplay.py). A new name, so the old table stays a table.
CREATE OR REPLACE VIEW gate_outcomes AS
    SELECT id, reply_id, session_id AS meeting_id, mode, decision, category,
           recorded_at AS decided_at
    FROM gate_records
    WHERE event IN ('outcome', 'legacy');

-- Append-only: the application role may insert and read, nothing else. A database owner can
-- still drop the triggers; this stops the engine's own code, not an administrator.
CREATE OR REPLACE FUNCTION gate_records_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'gate_records is append-only: % refused', TG_OP
        USING ERRCODE = 'restrict_violation';
END;
$$;

CREATE TRIGGER gate_records_no_change
    BEFORE UPDATE OR DELETE ON gate_records
    FOR EACH ROW EXECUTE FUNCTION gate_records_append_only();

CREATE TRIGGER gate_records_no_truncate
    BEFORE TRUNCATE ON gate_records
    FOR EACH STATEMENT EXECUTE FUNCTION gate_records_append_only();
