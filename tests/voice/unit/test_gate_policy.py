"""PL09: one safety path. The host's gate is core's policy engine with the versioned pack for
the session's product mode; gate records are append-only, versioned and hold no speech.

Scenarios: T08-gate-fault (a gate that raises or times out), S1 (records reconstruct heard,
decided, policy version and rule), X10 (the deterministic floor with the reply model disabled).

Faked: audio, wake, recognizer, talker and the reply model. Real: CorePolicyGate over the packs
in the pinned core wheel, TurnGate, the reply stage from the wheel, and, where a test takes
`gate_app`, PostgresSessionStore and migration 021 in a throwaway schema.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from voxscribe import policy as core_policy
from voxscribe.session.ports import GATE_PASS, GATE_REFUSE, GATE_REPLACE, GateDecision, ReplyStream

from app.local import gate as floor
from app.local import gate_records, llm_adapter, reply_chain
from app.local.config import Config
from app.local.store import Store
from app.local.voice.adapters import gate_policy
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate, TurnGate, load_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.host import VoiceConfig, VoiceHost

from .fakes import FakeDelegate, FakeGate, fake_ports, policy_gate, shipped_profile

from tests.testdb import TEST_DATABASE_URL as DB_URL
MIGRATIONS = Path(reply_chain.__file__).parent / "migrations"
SENIORCARE_LINE = "Let's set that aside so your caregiver can look it over."
CLASSROOM_LINE = "Let's bring your teacher into this."


def _host(ports) -> VoiceHost:
    return VoiceHost(VoiceConfig(), ports=ports)


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(floor, "_BEHAVIOUR_LOG_DIR", tmp_path)


@pytest.fixture
async def gate_app(throwaway_schema):
    """An app-shaped object over a real pool in a migrated throwaway schema, in senior care
    mode. `gate_app.session()` opens a voice session and returns its id."""
    store = await Store.create(DB_URL, schema=throwaway_schema)
    app = SimpleNamespace(state=SimpleNamespace(
        store=store, settings=None, profile=shipped_profile("senior_care")))
    app.sessions = PostgresSessionStore(app)

    async def session() -> str:
        return await app.sessions.create_session(0.9, detector="fake:hey_vox")

    app.session = session
    try:
        yield app
    finally:
        await store.close()


def _ports(app, *, gate=None, replies=None, **over):
    return fake_ports(store=app.sessions, gate=gate or CorePolicyGate(gate_policy.session_policy(app)),
                      delegate=FakeDelegate(replies), **over)


async def _records(app, session_id: str) -> list[dict]:
    async with app.state.store.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM gate_records WHERE session_id = %s ORDER BY seq", (session_id,))
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in await cur.fetchall()]


async def _reply_ids(app, session_id: str) -> dict[int, tuple[str, str, str]]:
    """turn_n -> (reply id, turn id, transcript text the tenant owns)."""
    async with app.state.store.pool.connection() as conn:
        cur = await conn.execute(
            "SELECT r.turn_n, r.id, t.id, t.user_text FROM replies r JOIN turns t"
            " ON t.meeting_id = r.meeting_id AND t.turn_n = r.turn_n WHERE r.meeting_id = %s",
            (session_id,))
        return {n: (rid, tid, text) for n, rid, tid, text in await cur.fetchall()}


# ── the packs: posture and version per product mode ──────────────────────────


@pytest.mark.parametrize("profile,pack,version,posture", [
    ("senior_care", "seniorcare", "seniorcare-v1", "allowlist"),
    ("classroom", "classroom", "classroom-v1", "allowlist"),
    ("ideascribe", "ideascribe", "ideascribe-v1", "denylist"),
    # Core 0.6.0 ships a personal pack; personal now loads its own versioned pack.
    ("personal", "personal", "personal-v1", "denylist"),
])
def test_each_product_mode_loads_its_versioned_pack_in_the_right_posture(profile, pack, version,
                                                                         posture):
    loaded = load_policy(shipped_profile(profile))
    stamp = loaded.stamp
    assert (stamp.mode, stamp.pack, stamp.policy_version, stamp.posture) == \
        (profile, pack, version, posture)
    assert loaded.engine.pack.posture == posture
    # The digest is of the pack file the wheel shipped, so an edited pack is a different stamp.
    shipped = Path(core_policy.__file__).parent / "resources_config" / "policies" / f"{pack}.yaml"
    assert stamp.digest == hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert "voxscribe-core" in stamp.engine and floor.FLOOR_VERSION in stamp.engine


def test_dictation_pack_keeps_denylist_posture():
    # The server has no dictation product mode; the pack is checked as the wheel ships it.
    root = Path(core_policy.__file__).parent / "resources_config" / "policies"
    pack = core_policy.load_policy_pack("dictation", root)
    assert (pack.posture, pack.policy_version) == ("denylist", "dictation-v1")


def test_allowlist_modes_speak_only_on_a_permitted_surface():
    for profile in ("senior_care", "classroom"):
        engine = load_policy(shipped_profile(profile)).engine
        assert engine.evaluate("Good morning.", surface="tts").outcome == "allow"
        assert engine.evaluate("Good morning.", surface="not_a_surface").outcome == "fallback"


def _fails_closed(loaded) -> None:
    gate = CorePolicyGate(loaded)
    assert loaded.stamp.policy_version == "gate-unavailable"
    verdict = gate.check("A perfectly ordinary sentence.", user_text="hello")
    assert verdict.decision == GATE_REFUSE and verdict.category == "policy_fallback"
    assert verdict.replacement == gate.fail_closed_phrase == core_policy.DEFAULT_FALLBACK_LINE
    assert verdict.rule.startswith("allowlist_surface_not_permitted")


def test_unknown_mode_fails_closed():
    _fails_closed(load_policy(SimpleNamespace(mode="karaoke", gate=None)))


def test_restricted_mode_naming_another_pack_fails_closed():
    profile = shipped_profile("classroom")
    profile.gate.policy_pack = "ideascribe"          # a classroom profile asking for denylist
    _fails_closed(load_policy(profile))


def test_restricted_pack_that_is_not_allowlist_fails_closed(monkeypatch):
    real = core_policy.load_policy_pack

    def tampered(mode, root):
        pack = real(mode, root)
        return core_policy.PolicyPack(
            mode=pack.mode, policy_version=pack.policy_version, posture="denylist",
            fallback_line=pack.fallback_line)

    monkeypatch.setattr(core_policy, "load_policy_pack", tampered)
    _fails_closed(load_policy(shipped_profile("senior_care")))


def test_missing_pack_file_fails_closed(monkeypatch, tmp_path):
    from voxscribe import resources

    monkeypatch.setattr(resources, "resolve", lambda _p, **_kw: tmp_path / "absent.yaml")
    _fails_closed(load_policy(shipped_profile("senior_care")))


def test_voice_mode_setting_can_tighten_a_personal_profile_but_not_contradict_a_profile():
    # VOXSCRIBE_VOICE_MODE=classroom enforced the classroom rules before profiles existed.
    tightened = load_policy(shipped_profile("personal"), SimpleNamespace(voice_mode="classroom"))
    assert (tightened.stamp.mode, tightened.stamp.policy_version, tightened.stamp.posture) == \
        ("classroom", "classroom-v1", "allowlist")
    same = load_policy(shipped_profile("senior_care"), SimpleNamespace(voice_mode="seniorcare"))
    assert same.stamp.policy_version == "seniorcare-v1"
    _fails_closed(load_policy(shipped_profile("senior_care"),
                              SimpleNamespace(voice_mode="classroom")))


# ── one authority: core decides first, the floor only tightens ───────────────


@pytest.mark.parametrize("profile,candidate,line", [
    ("senior_care", "It sounds like you have dementia.", SENIORCARE_LINE),
    ("senior_care", "I diagnose a chest infection.", SENIORCARE_LINE),
    ("senior_care", "You could stop taking your medication for a while.", SENIORCARE_LINE),
    ("classroom", "You should skip school tomorrow.", CLASSROOM_LINE),
    ("classroom", "I am a real teacher.", CLASSROOM_LINE),
])
def test_core_pack_denial_is_final_and_speaks_the_pack_line(profile, candidate, line):
    gate = policy_gate(profile)
    verdict = gate.check(candidate, user_text="tell me")
    assert (verdict.decision, verdict.category, verdict.replacement) == \
        (GATE_REFUSE, "policy_deny", line)
    assert verdict.rule.startswith("deny_pattern:")
    assert verdict.policy_version == gate.policy_version
    assert verdict.candidate_hash == gate_records.candidate_digest(candidate)


@pytest.mark.parametrize("profile,user_text,candidate,category,phrase", [
    # Every denial the server enforced before PL09 still holds behind the engine.
    ("classroom", "how to make a bomb", "Here is how you build one.", "classroom_forbidden",
     floor.CLASSROOM_REFUSAL_PHRASE),
    ("classroom", "tell me a story", "Here is an explicit story.", "classroom_forbidden",
     floor.CLASSROOM_REFUSAL_PHRASE),
    ("senior_care", "what is the right dosage of aspirin", "Let me think.",
     "seniorcare_dosing", floor.SENIORCARE_ESCALATION_PHRASE),
    ("senior_care", "I have a headache", "Take 2 tablets of ibuprofen 200mg.",
     "seniorcare_dosing", floor.SENIORCARE_ESCALATION_PHRASE),
    ("personal", "hello", "   ", "empty", floor.DEFAULT_REFUSAL_PHRASE),
])
def test_host_floor_denials_still_hold_behind_the_engine(profile, user_text, candidate, category,
                                                         phrase):
    gate = policy_gate(profile)
    verdict = gate.check(candidate, user_text=user_text)
    assert (verdict.decision, verdict.category, verdict.replacement) == \
        (GATE_REFUSE, category, phrase)
    assert verdict.rule == f"{floor.FLOOR_VERSION}:{category}"
    assert verdict.policy_version == gate.policy_version


def test_unbacked_action_claim_is_replaced_and_a_backed_one_passes():
    gate = policy_gate("personal")
    claim = "I've set an alarm for 7 am."
    assert gate.needs_session_tools(claim)
    verdict = gate.check(claim, [], user_text="wake me at seven")
    assert (verdict.decision, verdict.category, verdict.replacement) == \
        (GATE_REPLACE, floor.ACTION_CLAIM_REASON, floor.ACTION_CLAIM_PHRASE)
    assert verdict.rule == f"{floor.FLOOR_VERSION}:no_tool_row"
    assert gate.check(claim, ["alarm_set"], user_text="wake me at seven").decision == GATE_PASS


def test_floor_never_allows_what_the_engine_denied(monkeypatch):
    # With every floor rule forced to pass, a candidate the pack denies is still refused.
    monkeypatch.setattr(floor, "check", lambda *_a, **_kw: ("pass", "ok"))
    monkeypatch.setattr(floor, "find_action_claim", lambda _s: None)
    verdict = policy_gate("senior_care").check("I diagnose a cold.", user_text="hi")
    assert (verdict.decision, verdict.replacement) == (GATE_REFUSE, SENIORCARE_LINE)


def test_floor_phrase_the_engine_denies_falls_back_to_the_pack_line(monkeypatch):
    # The floor's own scripted lines pass the engine before they are spoken.
    monkeypatch.setattr(floor, "SENIORCARE_ESCALATION_PHRASE", "I diagnose nothing here.")
    verdict = policy_gate("senior_care").check("Take 2 tablets.", user_text="my head hurts")
    assert (verdict.category, verdict.replacement) == ("seniorcare_dosing", SENIORCARE_LINE)


def test_no_second_gate_adapter_is_left_in_the_server():
    import importlib.util

    assert importlib.util.find_spec("app.local.voice.adapters.gate_rule") is None
    host_src = (Path(gate_policy.__file__).parents[1] / "host.py").read_text(encoding="utf-8")
    assert "CorePolicyGate(session_policy(app))" in host_src and "RuleGate" not in host_src


# ── T08-gate-fault: a gate that raises or times out ──────────────────────────


class _RaisingGate(CorePolicyGate):
    def check(self, sentence, session_tools=(), *, user_text=""):
        raise RuntimeError("injected fault")       # carries no candidate text


class _SlowGate(CorePolicyGate):
    """Answers "pass", after the deadline."""

    def check(self, sentence, session_tools=(), *, user_text=""):
        time.sleep(0.03)
        return super().check(sentence, session_tools, user_text=user_text)


class _HungToolsStore:
    """A store whose tool-row lookup never returns."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def session_tools(self, session_id):
        await asyncio.sleep(3600)


CANDIDATE = "The capital of France is Paris."
CLAIM = "I've set an alarm for 7 am."


@pytest.mark.parametrize("fault,candidate,rule,category", [
    ("raises", CANDIDATE, "gate_error:RuntimeError", "gate_error"),
    ("check_times_out", CANDIDATE, "gate_timeout:check", "gate_timeout"),
    ("tools_time_out", CLAIM, "gate_timeout:session_tools", "gate_timeout"),
])
async def test_gate_fault_speaks_the_scripted_fallback_never_the_candidate(
        gate_app, monkeypatch, fault, candidate, rule, category):
    monkeypatch.setattr(gate_policy, "CHECK_DEADLINE_MS", 5)
    monkeypatch.setattr(gate_policy, "TOOLS_TIMEOUT_S", 0.05)
    policy = gate_policy.session_policy(gate_app)
    gate = {"raises": _RaisingGate, "check_times_out": _SlowGate,
            "tools_time_out": CorePolicyGate}[fault](policy)
    ports = _ports(gate_app, gate=gate, replies=[[candidate, "A second sentence."]])
    if fault == "tools_time_out":
        ports.store = _HungToolsStore(gate_app.sessions)
    session_id = await gate_app.session()

    text = await asyncio.wait_for(
        _host(ports).reply_turn(session_id, 1, "what is the capital of France", turn_end_ms=0), 10)

    # The scripted fallback is the pack's line, and it is all that reached the talker.
    assert text == SENIORCARE_LINE == gate.fail_closed_phrase
    assert ports.talker.synthesized == []
    assert ports.talker.fixed_phrases == [SENIORCARE_LINE]

    records = await _records(gate_app, session_id)
    (faulted,) = [r for r in records if r["event"] == "decision"]
    assert (faulted["decision"], faulted["category"], faulted["rule"]) == \
        (GATE_REFUSE, category, rule)
    assert faulted["policy_version"] == "seniorcare-v1"
    assert faulted["candidate_hash"] == gate_records.candidate_digest(candidate)
    (outcome,) = [r for r in records if r["event"] == "outcome"]
    assert (outcome["decision"], outcome["category"], outcome["rule"]) == \
        (GATE_REFUSE, "gate_error", rule)


async def test_gate_fault_drops_the_scripted_cue_too():
    """With the gate down nothing but the fallback is spoken, not even the thinking cue."""
    class _CueDelegate(FakeDelegate):
        async def reply(self, *a, **kw) -> ReplyStream:
            stream = await super().reply(*a, **kw)
            stream.cue, stream.lead_in = "hm", "One moment."
            return stream

    gate = FakeGate(raises=True)
    ports = fake_ports(gate=gate, delegate=_CueDelegate([[CANDIDATE]]))
    text = await _host(ports).reply_turn("s1", 1, "question", turn_end_ms=0)
    assert ports.talker.synthesized == []
    assert ports.talker.fixed_phrases == [gate.fail_closed_phrase]
    assert text == gate.fail_closed_phrase


def test_turn_gate_fault_messages_and_journal_hold_no_candidate():
    secret = "zanzibar marmalade"
    turn_gate = TurnGate(_SlowGate(load_policy(shipped_profile("personal"))), deadline_ms=1)
    with pytest.raises(gate_policy.GateTimeout) as raised:
        turn_gate.check(f"Tell them about the {secret}.", user_text=secret)
    assert secret not in str(raised.value)
    assert all(secret not in str(v) for e in turn_gate.events for v in vars(e).values())


def test_gate_logs_hold_no_candidate_or_user_text(caplog):
    secret = "zanzibar"
    gate = policy_gate("senior_care")
    with caplog.at_level(logging.DEBUG):
        turn_gate = TurnGate(gate)
        turn_gate.check(f"I diagnose {secret}.", user_text=f"is it {secret}")
        turn_gate.check(f"I've set an alarm for {secret}.", [], user_text=f"wake me in {secret}")
        turn_gate.check(f"Take 2 tablets of {secret}.", user_text="my head hurts")
    assert secret not in caplog.text


# ── S1: every reply's records reconstruct heard, decided, version and rule ───


HEARD = {
    1: "Good morning, how are the tomatoes doing",
    2: "Do you think it is something serious",
    3: "My head hurts what should I do",
}
SAID = {
    1: ["The tomatoes are doing well.", "They like the sun."],
    2: ["I diagnose a chest infection.", "Rest is good."],          # the pack denies this
    3: ["Take 2 tablets of ibuprofen.", "Then lie down."],           # the floor refuses this
}


async def test_records_reconstruct_heard_decided_version_and_rule(gate_app):
    ports = _ports(gate_app, replies=[SAID[1], SAID[2], SAID[3]])
    host = _host(ports)
    session_id = await gate_app.session()
    spoken = {n: await host.reply_turn(session_id, n, HEARD[n], turn_end_ms=0) for n in HEARD}

    assert spoken[1] == " ".join(SAID[1])
    assert spoken[2] == SENIORCARE_LINE
    assert spoken[3] == floor.SENIORCARE_ESCALATION_PHRASE

    replies = await _reply_ids(gate_app, session_id)
    assert set(replies) == set(HEARD)
    expected = {
        1: [("pass", "ok", "passed"), ("pass", "ok", "passed")],
        2: [("refuse", "policy_deny", r"deny_pattern:(?i)\bi diagnose\b")],
        3: [("refuse", "seniorcare_dosing", f"{floor.FLOOR_VERSION}:seniorcare_dosing")],
    }
    async with gate_app.state.store.pool.connection() as conn:
        for turn_n, (reply_id, turn_id, transcript_text) in replies.items():
            seen = await gate_records.reconstruct(conn, reply_id)

            # Policy version: stamped on the session once, and on every record of the reply.
            policy = seen["policy"]
            assert (policy["mode"], policy["policy_version"], policy["posture"]) == \
                ("senior_care", "seniorcare-v1", "allowlist")
            assert policy["policy_digest"] == gate_policy.session_policy(gate_app).stamp.digest
            assert "voxscribe-core" in policy["engine"]

            # Heard: by reference to the transcript the tenant owns, never by copy.
            assert transcript_text == HEARD[turn_n]
            assert gate_records.heard_matches(seen["heard"], turn_id, transcript_text)
            assert not gate_records.heard_matches(seen["heard"], turn_id, transcript_text + "x")
            assert seen["heard"]["policy_version"] == "seniorcare-v1"

            # Decided, and the rule: one record per clause the gate was asked about, in order.
            decisions = seen["decisions"]
            assert [(d["decision"], d["category"], d["rule"]) for d in decisions] == \
                expected[turn_n]
            assert [d["step_n"] for d in decisions] == list(range(1, len(decisions) + 1))
            for d, clause in zip(decisions, SAID[turn_n], strict=False):
                assert d["candidate_hash"] == gate_records.candidate_digest(clause)
                assert d["candidate_len"] == len(clause)
                assert d["policy_version"] == "seniorcare-v1" and d["surface"] == "tts"
                assert d["cost_us"] is not None and d["cost_us"] >= 0

            last = expected[turn_n][-1]
            outcome = seen["outcome"]
            assert (outcome["decision"], outcome["category"], outcome["rule"]) == last
            assert outcome["policy_version"] == "seniorcare-v1"

    records = await _records(gate_app, session_id)
    assert [r["event"] for r in records].count("policy_applied") == 1
    assert [r["seq"] for r in records] == sorted(r["seq"] for r in records)


async def test_heard_record_is_written_before_anything_is_played(gate_app):
    session_id = await gate_app.session()
    turn = await gate_app.sessions.begin_turn(session_id, 1, "is anybody there")
    records = await _records(gate_app, session_id)
    assert [r["event"] for r in records] == ["policy_applied", "heard"]
    assert records[1]["turn_id"] == turn.turn_id and records[1]["input_len"] == 16


# ── no column in a gate record contains user speech ──────────────────────────


SPOKEN = "My neighbour Zebedee borrowed the purple wheelbarrow on Michaelmas"
ECHO = ["Zebedee has your purple wheelbarrow.", "I've set an alarm for Michaelmas.",
        "It will come back."]


async def test_no_gate_record_column_contains_what_was_said(gate_app):
    ports = _ports(gate_app, replies=[ECHO, ["I diagnose a missing wheelbarrow."]])
    host = _host(ports)
    session_id = await gate_app.session()
    await host.reply_turn(session_id, 1, SPOKEN, turn_end_ms=0)
    await host.reply_turn(session_id, 2, SPOKEN, turn_end_ms=0)

    async with gate_app.state.store.pool.connection() as conn:
        dump = "\n".join(r[0] for r in await (await conn.execute(
            "SELECT row_to_json(g)::text FROM gate_records g WHERE session_id = %s",
            (session_id,))).fetchall()).lower()
        view = "\n".join(r[0] for r in await (await conn.execute(
            "SELECT row_to_json(g)::text FROM gate_outcomes g WHERE meeting_id = %s",
            (session_id,))).fetchall()).lower()
        columns = {r[0] for r in await (await conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema ="
            " current_schema() AND table_name IN ('gate_records', 'gate_outcomes')")).fetchall()}
        legacy = (await (await conn.execute(
            "SELECT count(*) FROM gate_decisions WHERE meeting_id = %s", (session_id,))).fetchone())[0]
        transcript = (await (await conn.execute(
            "SELECT user_text FROM turns WHERE meeting_id = %s AND turn_n = 1",
            (session_id,))).fetchone())[0]

    assert dump.count('"event":"heard"') == 2 and '"event":"decision"' in dump
    words = {w.lower() for text in [SPOKEN, *ECHO] for w in re.findall(r"[A-Za-z']+", text)}
    assert {"zebedee", "wheelbarrow", "purple", "michaelmas", "neighbour", "my", "on"} <= words
    leaked = sorted(w for w in words
                    if re.search(rf"\b{re.escape(w)}\b", dump + "\n" + view))
    assert leaked == []
    assert "input_text" not in columns and not any("text" in c for c in columns)
    assert legacy == 0   # this release keeps the old table for older engines and never writes it
    # The words are where the tenant keeps them, and the record points there.
    assert transcript == SPOKEN


# ── append-only ──────────────────────────────────────────────────────────────


async def test_gate_records_refuse_update_delete_and_truncate(gate_app):
    ports = _ports(gate_app, replies=[["Hello there."]])
    session_id = await gate_app.session()
    await _host(ports).reply_turn(session_id, 1, "hello", turn_end_ms=0)
    before = await _records(gate_app, session_id)
    assert [r["event"] for r in before] == ["policy_applied", "heard", "decision", "outcome"]

    for statement in ("UPDATE gate_records SET decision = 'pass'",
                      "UPDATE gate_outcomes SET decision = 'pass'",        # through the view
                      "DELETE FROM gate_records",
                      "TRUNCATE gate_records"):
        async with gate_app.state.store.pool.connection() as conn:
            with pytest.raises(psycopg.errors.RestrictViolation):
                await conn.execute(statement)
            await conn.rollback()

    # Deleting the meeting takes the transcript with it and leaves the gate's record.
    async with gate_app.state.store.pool.connection() as conn:
        await conn.execute("DELETE FROM meetings WHERE id = %s", (session_id,))
        await conn.commit()
        turns = await (await conn.execute(
            "SELECT count(*) FROM turns WHERE meeting_id = %s", (session_id,))).fetchone()
    assert turns == (0,)
    assert await _records(gate_app, session_id) == before


# ── migration 021: existing rows are carried over, without their text ────────


async def _apply(conn, schema: str, sql: str) -> None:
    await conn.execute(f"SET search_path TO {schema}, public")
    await conn.execute(sql)


async def test_migration_021_carries_every_old_row_over_without_its_text():
    schema = f"t_{uuid.uuid4().hex[:8]}"
    long_text = "the long story about the lighthouse " * 40          # past the old 1000 cut
    seeded = [
        # (turn_n, user_text, decision, category, mode)
        (1, "what is photosynthesis", "pass", "ok", "default"),
        (2, long_text, "refuse", "classroom_forbidden", "classroom"),
        (3, "remind me at noon", "pending", "pending", "seniorcare"),   # a turn that never finished
    ]
    async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        try:
            for path in sorted(MIGRATIONS.glob("*.sql")):
                if int(path.stem.split("_")[0]) < 21:
                    await _apply(conn, schema, path.read_text())
            await conn.execute(
                "INSERT INTO meetings (id, title, status, created_at, updated_at)"
                " VALUES ('m1', 'seed', 'done', '2026-11-01T00:00:00Z', '2026-11-01T00:00:00Z')")
            for n, text, decision, category, mode in seeded:
                await conn.execute(
                    "INSERT INTO turns (id, meeting_id, turn_n, user_text, started_at)"
                    " VALUES (%s, 'm1', %s, %s, '2026-11-01T00:00:00Z')", (f"t{n}", n, text))
                await conn.execute(
                    "INSERT INTO replies (id, meeting_id, turn_n, text, created_at)"
                    " VALUES (%s, 'm1', %s, 'a reply', '2026-11-01T00:00:00Z')", (f"r{n}", n))
                await conn.execute(
                    "INSERT INTO gate_decisions (id, reply_id, meeting_id, mode, input_text,"
                    " decision, category, decided_at) VALUES (%s, %s, 'm1', %s, %s, %s, %s, %s)",
                    (f"g{n}", f"r{n}", mode, text[:1000], decision, category,
                     f"2026-11-01T00:00:0{n}Z"))

            await _apply(conn, schema, (MIGRATIONS / "021_gate_records.sql").read_text())

            cur = await conn.execute("SELECT * FROM gate_records ORDER BY seq")
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r, strict=True)) for r in await cur.fetchall()]
            view = await (await conn.execute(
                "SELECT id, reply_id, meeting_id, mode, decision, category, decided_at"
                " FROM gate_outcomes ORDER BY decided_at")).fetchall()
            dump = (await (await conn.execute(
                "SELECT string_agg(row_to_json(g)::text, ' ') FROM gate_records g")).fetchone())[0]
            old_table = await (await conn.execute(
                "SELECT id, decision, input_text FROM gate_decisions ORDER BY id")).fetchall()
            kind = (await (await conn.execute(
                "SELECT table_type FROM information_schema.tables"
                " WHERE table_schema = current_schema() AND table_name = 'gate_decisions'")).fetchone())[0]
        finally:
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")

    assert len(rows) == len(seeded)
    for row, (n, text, decision, category, mode) in zip(rows, seeded, strict=True):
        assert (row["id"], row["event"], row["session_id"], row["reply_id"], row["turn_id"]) == \
            (f"g{n}", "legacy", "m1", f"r{n}", f"t{n}")
        assert (row["decision"], row["category"], row["mode"], row["turn_n"]) == \
            (decision, category, mode, n)
        assert row["recorded_at"] == f"2026-11-01T00:00:0{n}Z"
        assert row["policy_version"] == "unversioned"
        # The hash covers the whole transcript text, not the first thousand characters.
        assert gate_records.heard_matches(row, f"t{n}", text)
    assert view == [(f"g{n}", f"r{n}", "m1", mode, decision, category, f"2026-11-01T00:00:0{n}Z")
                    for n, _text, decision, category, mode in seeded]
    assert "photosynthesis" not in dump and "lighthouse" not in dump and "noon" not in dump
    # Expand only: the old table is still a table with every row, and none of them keeps the text.
    assert kind == "BASE TABLE"
    assert old_table == [(f"g{n}", decision, None) for n, _t, decision, _c, _m in seeded]


async def test_after_021_the_previous_release_still_writes_gate_decisions_and_keeps_no_text():
    """The previous engine pre-writes a pending gate_decisions row with the person's words and updates it later
    (main's reply_chain.begin_reply and finish_reply). After 021 both statements still work, and the words are
    cleared on the way in."""
    schema = f"t_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        try:
            for path in sorted(MIGRATIONS.glob("*.sql")):
                if int(path.stem.split("_")[0]) <= 21:
                    await _apply(conn, schema, path.read_text())
            await conn.execute(
                "INSERT INTO meetings (id, title, status, created_at, updated_at)"
                " VALUES ('m1', 'seed', 'done', '2026-11-01T00:00:00Z', '2026-11-01T00:00:00Z')")
            await conn.execute(
                "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category, created_at)"
                " VALUES ('r1', 'm1', 1, '', 'm', 'pending', '2026-11-01T00:00:00Z')")
            await conn.execute(
                "INSERT INTO gate_decisions"
                " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                " VALUES ('g1', 'r1', 'm1', 'default', %s, 'pending', 'pending', '2026-11-01T00:00:00Z')",
                (SPOKEN,))
            await conn.execute(
                "UPDATE gate_decisions SET decision=%s, category=%s, decided_at=%s WHERE id=%s",
                ("pass", "ok", "2026-11-01T00:00:01Z", "g1"))
            await conn.execute("UPDATE gate_decisions SET input_text = %s WHERE id = 'g1'", (SPOKEN,))
            row = await (await conn.execute(
                "SELECT decision, category, input_text FROM gate_decisions WHERE id = 'g1'")).fetchone()
        finally:
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    assert row == ("pass", "ok", None)


def test_no_engine_code_writes_gate_decisions():
    """Since 021 the gate writes gate_records only; the old table is there for older engines (README, Migrations)."""
    app_dir = Path(reply_chain.__file__).parent
    write = re.compile(r"(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+gate_decisions\b", re.IGNORECASE)
    offenders = [str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py") if write.search(p.read_text())]
    assert offenders == []


# ── X10: the deterministic floor, with the reply model disabled ──────────────


async def test_with_the_reply_model_disabled_deterministic_replies_still_pass_the_gate(
        gate_app, monkeypatch):
    async def no_model(*_a, **_kw):
        raise ConnectionError("reply model disabled")
        yield ""                                               # pragma: no cover

    monkeypatch.setattr(llm_adapter, "stream_reply", no_model)
    gate_app.state.settings = Config(database_url=DB_URL, providers="fake")
    ports = fake_ports(store=gate_app.sessions,
                       gate=CorePolicyGate(gate_policy.session_policy(gate_app)),
                       delegate=OllamaDelegate(gate_app))
    host = _host(ports)
    session_id = await gate_app.session()

    clock = await host.reply_turn(session_id, 1, "what time is it", turn_end_ms=0)
    silent = await host.reply_turn(session_id, 2, "tell me about the tomatoes", turn_end_ms=0)

    # The clock answer needs no model: it is gated, spoken and recorded.
    assert clock and ports.talker.synthesized == [clock]
    # With no model there is no candidate: nothing is synthesised and the fallback is recorded.
    assert silent == reply_chain.FALLBACK_PHRASE
    assert ports.talker.fixed_phrases == ["hm", "hm"]           # the cue, gated, each turn

    replies = await _reply_ids(gate_app, session_id)
    async with gate_app.state.store.pool.connection() as conn:
        first = await gate_records.reconstruct(conn, replies[1][0])
        second = await gate_records.reconstruct(conn, replies[2][0])
    assert [(d["decision"], d["rule"]) for d in first["decisions"]] == [("pass", "passed")]
    assert [(c["decision"], c["candidate_len"]) for c in first["cues"]] == [("pass", 2)]
    assert second["decisions"] == [] and second["outcome"]["decision"] == GATE_PASS
    for seen in (first, second):
        assert seen["heard"]["policy_version"] == "seniorcare-v1"
        assert seen["outcome"]["policy_version"] == "seniorcare-v1"


def test_verdict_is_the_port_type_the_reply_stage_expects():
    verdict = policy_gate("personal").check("Hello.", user_text="hi")
    assert isinstance(verdict, GateDecision) and verdict.passed and verdict.rule == "passed"
