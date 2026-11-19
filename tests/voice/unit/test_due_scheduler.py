"""PL46: the due-item scheduler and initiated speech.

A fake scheduler clock (the test moves time; every wait the scheduler asks for is recorded),
the in-memory reminder store, the real voice host over fake voice ports (fake audio out, fake
talker, fake recognizer), the real gate or a fake one where named, and the real session store
over the in-memory pool for gate records. No database, no model, no audio device.

Acceptance criteria (TOOLS-CONNECTORS.md section 5, PL46) and the test that checks each:
  with no session open, an item due at T plays the chime then its line exactly once, within
  the configured lateness of T          test_an_item_due_with_no_session_plays_chime_then_line_once
  with the clock past T while the scheduler is stopped, a restart delivers it once as missed
                                        test_a_restart_delivers_what_came_due_once_as_missed
  inside quiet hours under personal, 0 audio frames and 1 event
                                        test_quiet_hours_send_the_event_only_and_speak_it_next_session
  when the fake gate refuses the item's words, the neutral time line plays and the gate record
  shows the refusal                     test_refused_words_play_the_neutral_time_line
  an item that comes due while the person speaks plays after that turn's reply
                                        test_an_item_due_while_the_person_speaks_plays_after_the_reply
  with the device bound to tenant A, tenant B's due item is not spoken
                                        test_another_tenants_item_is_not_spoken_here
  the scheduler wakes on a new row without a polling interval
                                        test_it_wakes_only_for_the_next_item_and_for_a_new_row
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.local import gate as floor
from app.local import tenancy
from app.local.config import QuietHours
from app.local.voice import reminder_store as rs
from app.local.voice.adapters.gate_policy import CorePolicyGate, session_policy
from app.local.voice.due_scheduler import EVENT_KIND, DueScheduler
from app.local.voice.host import (
    CHIME_DUE,
    Initiated,
    SessionBinding,
    SessionScope,
    VoiceConfig,
    VoiceHost,
    WakeState,
)
from tests.voice.unit.fakes import (
    EndlessQueue,
    FakeAudioOut,
    FakeDelegate,
    FakeGate,
    FakeRecognizer,
    chunk,
    fake_ports,
)
from tests.voice.unit.reminder_fakes import START, ZONE, FakeReminderStore, SessionRows
from tests.voice.unit.tool_fakes import ToolPool, load_profile, make_app


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(floor, "_BEHAVIOUR_LOG_DIR", tmp_path)


class SchedulerClock:
    """The scheduler's clock: now is moved by the test; each wait is recorded and returns when
    the scheduler is told of a change or the clock reaches the time it waits for."""

    def __init__(self, now=START) -> None:
        self.current = now
        self.waits: list = []
        self._ticks: list[asyncio.Event] = []

    def now(self):
        return self.current

    async def wait(self, changed: asyncio.Event, until) -> None:
        self.waits.append(until)
        while not changed.is_set() and (until is None or self.current < until):
            tick = asyncio.Event()
            self._ticks.append(tick)
            waiters = {asyncio.ensure_future(changed.wait()), asyncio.ensure_future(tick.wait())}
            _done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()

    def advance(self, **kw) -> None:
        self.current += timedelta(**kw)
        ticks, self._ticks = self._ticks, []
        for t in ticks:
            t.set()


async def _settle(scheduler: DueScheduler | None = None, rounds: int = 30) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)
    if scheduler is not None:
        await scheduler.settle()
        for _ in range(rounds):
            await asyncio.sleep(0)


class Rig:
    """An app, a host over fake ports, a store, a scheduler and the events it published."""

    def __init__(self, mode: str = "personal", *, gate=None, store=None, delegate=None,
                 recognizer=None, quiet: QuietHours | None = None, host_scope=None,
                 binding=None, pool=None, stores=None) -> None:
        self.pool = pool or ToolPool()
        extra = {"quiet_hours": quiet} if quiet is not None else {}
        self.app = make_app(self.pool, profile=load_profile(mode), vox_timezone=ZONE, **extra)
        self.clock = SchedulerClock()
        self.store = store or FakeReminderStore()
        self.audio_out = FakeAudioOut()
        ports = fake_ports(gate=gate or CorePolicyGate(session_policy(self.app)),
                           store=SessionRows(self.app), audio_out=self.audio_out,
                           delegate=delegate or FakeDelegate(),
                           recognizer=recognizer or FakeRecognizer())
        self.ports = ports
        self.host = VoiceHost(VoiceConfig(), ports=ports, scope=host_scope, binding=binding)
        self.events: list[tuple[str, dict, str | None]] = []
        self.stores = stores
        self.scheduler = self.new_scheduler()

    def new_scheduler(self) -> DueScheduler:
        store = ((lambda: self.stores[tenancy.current_tenant()]) if self.stores is not None
                 else (lambda: self.store))
        return DueScheduler(self.app, clock=self.clock, speaker=self.host, store=store,
                            publish=lambda k, d: self.events.append(
                                (k, d, tenancy.current_tenant())))

    async def add(self, minutes: float, *, kind=rs.KIND_REMINDER, text="stretch", store=None):
        return await (store or self.store).add(
            kind=kind, text=text if kind == rs.KIND_REMINDER else "",
            label="one minute" if kind == rs.KIND_TIMER else None,
            fire_at=self.clock.current + timedelta(minutes=minutes), time_zone=ZONE,
            created_at=self.clock.current)

    @property
    def said(self) -> list[str]:
        return self.ports.talker.synthesized

    def gate_rows(self, item_id: str) -> list[dict]:
        return [r for r in self.pool.table("gate_records")
                if r.get("session_id") == f"due:{item_id}"]


# ── on time, once ────────────────────────────────────────────────────────────────────────


async def test_an_item_due_with_no_session_plays_chime_then_line_once():
    rig = Rig()
    item = await rig.add(10)
    rig.scheduler.start()
    await _settle()
    assert rig.said == [] and rig.audio_out.files == []
    rig.clock.advance(minutes=10)
    await _settle(rig.scheduler)
    assert rig.audio_out.files == [CHIME_DUE] and rig.said == ["Reminder: stretch."]
    assert len(rig.audio_out.played) == 1
    (delivery,) = rig.scheduler.delivered
    assert delivery.spoken and not delivery.missed
    assert rig.store.rows[item.id].status == rs.FIRED
    assert rig.store.rows[item.id].fired_at - item.fire_at <= timedelta(seconds=1)  # lateness
    assert rig.events == [(EVENT_KIND, {"id": item.id, "kind": "reminder", "text": "stretch",
                                        "dueAt": "2026-11-19T14:10:00-05:00", "missed": False,
                                        "spoken": True}, None)]
    rig.clock.advance(minutes=30)
    rig.scheduler.notify()
    await _settle(rig.scheduler)
    assert rig.said == ["Reminder: stretch."]                    # exactly once
    rows = rig.gate_rows(item.id)
    assert [r["event"] for r in rows] == ["policy_applied", "decision", "outcome"]
    assert rows[-1]["decision"] == "pass"
    await rig.scheduler.stop()


async def test_a_timer_is_announced_as_a_timer():
    rig = Rig()
    timer = await rig.add(1, kind=rs.KIND_TIMER)
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=1)
    await _settle(rig.scheduler)
    assert rig.said == ["Your one minute timer is done."]
    assert rig.events[0][1] | {} == {"id": timer.id, "kind": "timer", "text": None,
                                     "dueAt": timer.due_at, "missed": False, "spoken": True}
    await rig.scheduler.stop()


async def test_two_racing_deliverers_fire_an_item_once():
    rig = Rig()
    await rig.add(-1)
    other = rig.new_scheduler()
    counts = await asyncio.gather(rig.scheduler.fire_due(rig.clock.now()),
                                  other.fire_due(rig.clock.now()))
    await _settle(rig.scheduler)
    await other.settle()
    assert sorted(counts) == [0, 1] and rig.said == ["Reminder: stretch."]


async def test_it_wakes_only_for_the_next_item_and_for_a_new_row():
    rig = Rig()
    first, second = await rig.add(10), await rig.add(20)
    rig.scheduler.start()
    await _settle()
    assert rig.clock.waits == [first.fire_at]                   # no interval: the next item
    rig.clock.advance(minutes=10)
    await _settle(rig.scheduler)
    assert rig.clock.waits == [first.fire_at, second.fire_at]
    new = await rig.add(5)                                      # set by voice meanwhile
    rig.scheduler.notify()
    await _settle()
    assert rig.clock.waits == [first.fire_at, second.fire_at, new.fire_at]
    await rig.scheduler.stop()


# ── restarts ─────────────────────────────────────────────────────────────────────────────


async def test_a_restart_delivers_what_came_due_once_as_missed():
    rig = Rig()
    item = await rig.add(10)
    rig.scheduler.start()
    await _settle()
    await rig.scheduler.stop()                                  # the engine stops
    rig.clock.advance(hours=2)                                  # T passes while it is down
    restarted = rig.new_scheduler()
    restarted.start()
    await _settle(restarted)
    assert rig.said == ["While I was off, you had a reminder today at two ten in the afternoon:"
                        " stretch."]
    assert rig.events[-1][1]["missed"] is True and rig.events[-1][1]["spoken"] is True
    assert rig.store.rows[item.id].status == rs.MISSED
    await restarted.stop()
    again = rig.new_scheduler()                                 # and once more: nothing
    again.start()
    await _settle(again)
    assert len(rig.said) == 1 and len(rig.events) == 1
    await again.stop()


async def test_past_the_missed_window_an_item_is_only_listed():
    rig = Rig()
    item = await rig.add(10)
    rig.clock.advance(days=2)
    rig.scheduler.start()
    await _settle(rig.scheduler)
    assert rig.said == [] and rig.events == []
    assert rig.store.rows[item.id].state == "missed"
    await rig.scheduler.stop()


async def test_a_row_left_firing_by_a_crash_is_delivered_after_the_restart():
    rig = Rig()
    item = await rig.add(-1)
    await rig.store.claim_due(rig.clock.now())                  # claimed, then the engine died
    rig.scheduler.start()
    await _settle(rig.scheduler)
    assert rig.store.rows[item.id].status == rs.MISSED and len(rig.said) == 1
    await rig.scheduler.stop()


# ── quiet hours (TC5) ────────────────────────────────────────────────────────────────────


async def test_quiet_hours_send_the_event_only_and_speak_it_next_session():
    # 14:00 in Toronto; quiet from 14:05 to 15:00.
    rig = Rig(quiet=QuietHours(enabled=True, start="14:05", end="15:00"),
              recognizer=FakeRecognizer(["what time is it please"]),
              delegate=FakeDelegate([["It is twenty past two."]]))
    item = await rig.add(10)
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=10)
    await _settle(rig.scheduler)
    assert rig.audio_out.played == [] and rig.audio_out.files == []      # 0 audio frames
    assert [e[1]["spoken"] for e in rig.events] == [False]                 # 1 event
    assert rig.store.rows[item.id].speak_owed
    # The next session: the line is said after its first reply.
    rig.clock.advance(minutes=50)
    await asyncio.wait_for(rig.host.run_session(EndlessQueue(), score=0.9, detector="fake"), 10)
    await _settle(rig.scheduler)
    assert rig.said[-2:] == ["It is twenty past two.",
                             "Earlier, you had a reminder today at two ten in the afternoon:"
                             " stretch."]
    assert not rig.store.rows[item.id].speak_owed and rig.store.rows[item.id].spoken
    await rig.scheduler.stop()


async def test_senior_care_due_items_keep_quiet_hours_too():
    # The senior care profile's window is 21:00 to 08:00 and it never blocks a wake.
    rig = Rig("senior_care")
    rig.clock.advance(hours=8)                                   # 22:00 in Toronto
    await rig.add(1)
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=1)
    await _settle(rig.scheduler)
    assert rig.said == [] and rig.events[0][1]["spoken"] is False
    # ... while a wake inside quiet hours is still answered there.
    assert VoiceConfig.from_profile(load_profile("senior_care")).quiet_hours_block_wake is False
    await rig.scheduler.stop()


# ── the gate ─────────────────────────────────────────────────────────────────────────────


async def test_refused_words_play_the_neutral_time_line():
    rig = Rig(gate=FakeGate(refuse="stretch"))
    item = await rig.add(3)
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=3)
    await _settle(rig.scheduler)
    assert rig.said == ["You have a reminder you set for two oh three in the afternoon."]
    rows = rig.gate_rows(item.id)
    assert [(r["event"], r["decision"]) for r in rows] == [
        ("policy_applied", None), ("decision", "refuse"), ("decision", "pass"),
        ("outcome", "replace")]
    assert rows[1]["category"] == "blocked" and rows[-1]["category"] == "blocked"
    assert rig.events[0][1]["text"] is None                     # the words are not shown either
    await rig.scheduler.stop()


async def test_senior_care_dosing_words_are_never_said_when_due():
    """TC6, fail closed: a reminder holding a dose (set in another mode, or by an older
    engine) is announced by its time only. Flagged for clinician review."""
    rig = Rig("senior_care")
    rig.clock.advance(hours=-4)                                  # 10:00, outside quiet hours
    await rig.add(5, text="take 2 pills")
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=5)
    await _settle(rig.scheduler)
    assert rig.said == ["You have a reminder you set for ten oh five in the morning."]
    assert rig.events[0][1]["text"] is None
    await rig.scheduler.stop()


# ── sessions and people ──────────────────────────────────────────────────────────────────


async def test_an_item_due_while_the_person_speaks_plays_after_the_reply():
    loop = asyncio.get_running_loop()
    holder: dict = {}

    class Speaking(FakeRecognizer):
        def open_stream(self, session_id):
            stream = super().open_stream(session_id)
            push = stream.push

            def pushed(pcm):          # runs off the loop, while the person is talking
                if not holder.get("due"):
                    holder["due"] = True
                    holder["rig"].clock.current += timedelta(minutes=10)
                    loop.call_soon_threadsafe(holder["rig"].scheduler.notify)
                return push(pcm)

            stream.push = pushed
            return stream

    rig = Rig(recognizer=Speaking(["tell me a short joke please"]),
              delegate=FakeDelegate([["Why did the chicken cross the road?"]]))
    holder["rig"] = rig
    await rig.add(10)
    rig.scheduler.start()
    await _settle()
    await asyncio.wait_for(rig.host.run_session(EndlessQueue(), score=0.9, detector="fake"), 10)
    await _settle(rig.scheduler)
    assert rig.said == ["Why did the chicken cross the road?", "Reminder: stretch."]
    assert rig.events[0][1]["spoken"] is True
    await rig.scheduler.stop()


async def test_it_waits_for_a_quiet_room_and_says_nothing_over_a_voice():
    rig = Rig()
    rig.host.initiated_hold_s = 0.2
    rig.host._last_voice_mono = rig.ports.clock.mono_ms() / 1000   # someone is talking now
    result = await rig.host.speak_initiated(Initiated("i1", None, "Reminder: stretch."))
    assert (result.spoken, result.owed, result.reason) == (False, True, "person_speaking")
    assert rig.said == [] and rig.audio_out.files == []


async def test_it_stops_at_once_when_someone_starts_speaking():
    class Playing(FakeAudioOut):
        """Plays until stopped."""

        def __init__(self):
            super().__init__()
            self.playing = False

        async def play(self, pcm, sample_rate):
            await super().play(pcm, sample_rate)
            self.playing = True

        def stop_playback(self):
            super().stop_playback()
            self.playing = False

        @property
        def is_playing(self):
            return self.playing

    rig = Rig()
    out = Playing()
    rig.ports.audio_out = out
    rig.host.set_state(WakeState.ARMED)                      # the microphone is open
    task = asyncio.create_task(rig.host.speak_initiated(Initiated("i1", None, "Reminder: x.")))
    await _settle()
    for q in rig.ports.audio_in.queues:                       # a person starts talking
        for _ in range(20):
            q.put_nowait(chunk(0.2))
    result = await asyncio.wait_for(task, 5)
    assert result.spoken and out.stops == 1 and not out.is_playing


async def test_another_tenants_item_is_not_spoken_here():
    class Isolated(SessionScope):
        tenant_required = True

    class Enforced(ToolPool):
        enforced = True

        class provisioner:   # noqa: N801 - the pool's attribute name
            @staticmethod
            async def known_tenants():
                return ["t_a", "t_b"]

    stores = {"t_a": FakeReminderStore(), "t_b": FakeReminderStore()}
    rig = Rig(host_scope=Isolated(), binding=SessionBinding(tenant="t_a"), pool=Enforced(),
              stores=stores)
    a = await rig.add(5, text="water the plants", store=stores["t_a"])
    b = await rig.add(5, text="call the bank", store=stores["t_b"])
    rig.scheduler.start()
    await _settle()
    rig.clock.advance(minutes=5)
    await _settle(rig.scheduler)
    assert rig.said == ["Reminder: water the plants."]
    by_tenant = {t: d for _k, d, t in rig.events}
    assert by_tenant["t_a"]["id"] == a.id and by_tenant["t_a"]["spoken"] is True
    assert by_tenant["t_b"]["id"] == b.id and by_tenant["t_b"]["spoken"] is False
    assert stores["t_b"].rows[b.id].speak_owed                  # said at B's next session
    await rig.scheduler.stop()


async def test_with_no_tenant_bound_nothing_is_spoken():
    class Isolated(SessionScope):
        tenant_required = True

    rig = Rig(host_scope=Isolated())
    result = await rig.host.speak_initiated(Initiated("i1", "t_a", "Reminder: stretch."))
    assert (result.spoken, result.owed, result.reason) == (False, True, "tenant_not_bound")
    assert rig.said == []


async def test_no_reminder_words_reach_the_logs(caplog):
    import logging

    rig = Rig()
    await rig.add(1, text="call the dentist")
    with caplog.at_level(logging.DEBUG):
        rig.scheduler.start()
        await _settle()
        rig.clock.advance(minutes=1)
        await _settle(rig.scheduler)
        await rig.scheduler.stop()
    assert caplog.records and "dentist" not in caplog.text


def test_the_event_is_in_the_schema():
    from app.local import event_schema

    payload = {"id": "r1", "kind": "timer", "text": None,
               "dueAt": "2026-11-19T14:10:00-05:00", "missed": False, "spoken": True}
    assert event_schema.validate(EVENT_KIND, payload).kind == "timer"
    assert EVENT_KIND not in event_schema.SESSION_EVENTS | event_schema.DEVICE_EVENTS


async def test_it_never_opens_a_schema_and_holds_off_while_a_tenant_is_erased():
    class Enforced(ToolPool):
        enforced = True

        class provisioner:   # noqa: N801 - the pool's attribute name
            _ready = {"t_a"}                     # t_b's schema exists, unopened by this process

            @staticmethod
            async def known_tenants():
                return ["t_a", "t_b"]

    stores = {"t_a": FakeReminderStore(), "t_b": FakeReminderStore()}
    rig = Rig(pool=Enforced(), stores=stores)
    assert await rig.scheduler._tenants() == ["t_a"]
    # A delivery of t_a's is in flight (waiting for a session to end) when t_a erases.
    rig.host._claimed = True
    await rig.add(-1, store=stores["t_a"])
    await rig.scheduler.fire_due(rig.clock.now())
    await _settle()
    assert len(rig.scheduler._deliveries) == 1
    async with rig.scheduler.erasing("t_a"):
        assert rig.scheduler._deliveries == {}
    assert rig.events == [] and rig.said == []                 # nothing written after the erase
