"""Gate records: the append-only ledger of what the response gate heard and decided (PL09).

One row per event in `gate_records`. A row is written once and never changed: migration 021
refuses UPDATE, DELETE and TRUNCATE on the table.

  policy_applied  once per session: the policy in force (version, posture, pack digest, engine)
  heard           once per turn, before anything is played: a hash and a length of what the
                  person said, in place of the text (D17)
  cue             once per scripted line the gate was asked about: a cue, a lead-in, or the
                  line recorded when the reply source yielded nothing (category no_reply)
  decision        once per candidate clause the gate was asked about
  outcome         once per reply: the verdict the host acted on
  legacy          a row carried over from the old gate_decisions table by migration 021

No row holds what anyone said, and none holds a candidate. "Heard" is checked by reference:
`input_digest(turn_id, text)` over the turn's transcript text (turns.user_text, which the tenant
owns and can erase) equals the row's input_hash, and len(text) equals its input_len. Once the
tenant erases the transcript the hash refers to nothing.

Rows carry the session id by value, not as a foreign key, so deleting a meeting does not delete
the record of what the gate decided in it. Dropping a tenant's schema removes them.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger("app.local.gate_records")

EVENT_POLICY_APPLIED = "policy_applied"
EVENT_HEARD = "heard"
EVENT_CUE = "cue"
EVENT_DECISION = "decision"
EVENT_OUTCOME = "outcome"
EVENT_LEGACY = "legacy"

UNSTAMPED = "unstamped"          # policy_version of a decision from a gate that names no policy
RULE_PASSED = "passed"


def input_digest(turn_id: str | None, text: str) -> str:
    """SHA-256 over the turn id, a newline and the text. The turn id salts the hash, so the
    same words in two turns give two hashes and a table of common phrases matches nothing."""
    return hashlib.sha256(f"{turn_id or ''}\n{text}".encode()).hexdigest()


def candidate_digest(text: str) -> str:
    """The key core's policy engine gives a candidate: 16 hex characters of its SHA-256."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PolicyStamp:
    """The policy a session runs under, as loaded at start-up."""

    mode: str                 # the product mode, as the profile spells it
    pack: str                 # the core policy pack loaded for it
    policy_version: str       # the pack's own policy_version
    posture: str              # allowlist | denylist
    digest: str = ""          # SHA-256 of the pack file as shipped; empty when no pack loaded
    engine: str = ""          # the engine build and the host floor version


@dataclass(frozen=True)
class GateEvent:
    """One gate evaluation, as journalled during a reply and written when the reply ends."""

    event: str
    step_n: int
    surface: str
    decision: str
    category: str
    rule: str
    policy_version: str
    candidate_hash: str
    candidate_len: int
    cost_us: int
    recorded_at: str


def now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


_COLUMNS = ("id", "event", "session_id", "turn_id", "turn_n", "reply_id", "step_n", "mode",
            "policy_version", "posture", "policy_digest", "engine", "surface", "decision",
            "category", "rule", "input_hash", "input_len", "candidate_hash", "candidate_len",
            "cost_us", "recorded_at")
INSERT_SQL = (f"INSERT INTO gate_records ({', '.join(_COLUMNS)})"
              f" VALUES ({', '.join(['%s'] * len(_COLUMNS))})")
# One policy_applied row per session, however many turns ask for it.
INSERT_ONCE_PER_SESSION_SQL = (INSERT_SQL + " ON CONFLICT (session_id)"
                               f" WHERE event = '{EVENT_POLICY_APPLIED}' DO NOTHING")


def _row(event: str, rec, **values) -> tuple:
    stamp = getattr(rec, "policy", None)
    base = {
        "id": str(uuid.uuid4()), "event": event, "session_id": rec.meeting_id,
        "turn_id": rec.turn_id, "turn_n": rec.turn_n, "reply_id": rec.reply_id,
        "mode": stamp.mode if stamp else rec.mode,
        "policy_version": stamp.policy_version if stamp else UNSTAMPED,
        "recorded_at": now_z(),
    }
    base.update(values)
    return tuple(base.get(c) for c in _COLUMNS)


async def record_heard(pool, rec, stamp: PolicyStamp) -> None:
    """Stamp the session's policy (once) and record that this turn was heard, before anything
    is played. Errors are logged, never raised: a broken ledger does not open the gate, and
    does not close the session either."""
    rec.policy, rec.mode = stamp, stamp.mode
    try:
        async with pool.connection() as conn:
            await conn.execute(INSERT_ONCE_PER_SESSION_SQL, _row(
                EVENT_POLICY_APPLIED, rec, turn_id=None, turn_n=None, reply_id=None,
                posture=stamp.posture, policy_digest=stamp.digest or None,
                engine=stamp.engine or None))
            await conn.execute(INSERT_SQL, _row(
                EVENT_HEARD, rec, input_hash=input_digest(rec.turn_id, rec.user_text),
                input_len=len(rec.user_text)))
            await conn.commit()
    except Exception:
        log.exception("gate records: failed to write heard row meeting_id=%s", rec.meeting_id)


async def append_reply_events(conn, rec, *, decision: str, category: str) -> None:
    """The reply's journalled evaluations, then its outcome row. Runs on the caller's
    connection so the records commit with the reply row they describe."""
    events = rec.gate_events[rec.gate_events_written:]
    for ev in events:
        await conn.execute(INSERT_SQL, _row(
            ev.event, rec, step_n=ev.step_n, surface=ev.surface, decision=ev.decision,
            category=ev.category, rule=ev.rule,
            policy_version=ev.policy_version or (rec.policy.policy_version if rec.policy
                                                 else UNSTAMPED),
            candidate_hash=ev.candidate_hash, candidate_len=ev.candidate_len,
            cost_us=ev.cost_us, recorded_at=ev.recorded_at))
    rec.gate_events_written += len(events)
    # The rule behind the verdict: the last clause evaluation that did not pass, when there was
    # one; else the last scripted line that did not (a refused no-reply line has no clause).
    refused = [e for e in rec.gate_events if e.rule != RULE_PASSED]
    deciding = next((e for e in reversed(refused) if e.event == EVENT_DECISION),
                    refused[-1] if refused else None)
    if decision == "pass":
        rule = RULE_PASSED
    else:
        rule = deciding.rule if deciding is not None else category
    await conn.execute(INSERT_SQL, _row(
        EVENT_OUTCOME, rec, decision=decision, category=category, rule=rule))


_RECONSTRUCT_SQL = (
    "SELECT event, step_n, mode, policy_version, posture, policy_digest, engine, surface,"
    " decision, category, rule, input_hash, input_len, candidate_hash, candidate_len, cost_us,"
    " turn_id, recorded_at FROM gate_records"
    " WHERE reply_id = %s OR (event = %s AND session_id ="
    " (SELECT session_id FROM gate_records WHERE reply_id = %s LIMIT 1))"
    " ORDER BY seq")
_RECONSTRUCT_FIELDS = ("event", "step_n", "mode", "policy_version", "posture", "policy_digest",
                       "engine", "surface", "decision", "category", "rule", "input_hash",
                       "input_len", "candidate_hash", "candidate_len", "cost_us", "turn_id",
                       "recorded_at")


async def reconstruct(conn, reply_id: str) -> dict:
    """What the records say about one reply: the policy applied to its session, what was heard
    (by reference), each evaluation in order, and the outcome. Review reads this; nothing in
    the reply path does."""
    cur = await conn.execute(_RECONSTRUCT_SQL, (reply_id, EVENT_POLICY_APPLIED, reply_id))
    rows = [dict(zip(_RECONSTRUCT_FIELDS, r, strict=True)) for r in await cur.fetchall()]

    def first(event: str) -> dict | None:
        return next((r for r in rows if r["event"] == event), None)

    return {
        "policy": first(EVENT_POLICY_APPLIED),
        "heard": first(EVENT_HEARD),
        "cues": [r for r in rows if r["event"] == EVENT_CUE],
        "decisions": [r for r in rows if r["event"] == EVENT_DECISION],
        "outcome": first(EVENT_OUTCOME) or first(EVENT_LEGACY),
    }


def heard_matches(heard: dict | None, turn_id: str | None, transcript_text: str) -> bool:
    """True when the heard row refers to this transcript text."""
    return (heard is not None and heard["input_len"] == len(transcript_text)
            and heard["input_hash"] == input_digest(turn_id, transcript_text))
