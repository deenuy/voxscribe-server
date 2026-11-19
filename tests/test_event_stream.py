"""The event stream: an id per event, replay after a reconnect, an explicit reset when the gap is
older than the buffer, the versioned schema, and meeting buses that give every watcher every
event (PL06). In memory, no database.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path

import pytest
from starlette.requests import Request

from app.local import event_schema, events, meeting_bus, tenancy
from app.local.errors import InvalidRequestError
from app.local.routers import events as events_router
from app.local.voice.adapters import delegate_ollama

A, B = tenancy.tenant_key("pl06-stream-a"), tenancy.tenant_key("pl06-stream-b")
APP_DIR = Path(events.__file__).parent


@pytest.fixture(autouse=True)
def fresh_streams(monkeypatch):
    monkeypatch.setattr(events, "_streams", type(events._streams)())
    monkeypatch.setattr(events, "_replay_buffer", events.DEFAULT_REPLAY_BUFFER)
    monkeypatch.setattr(events, "_max_streams", events.DEFAULT_MAX_STREAMS)


def _publish(tenant: str | None, n: int, start: int = 0) -> None:
    with tenancy.tenant_scope(tenant) if tenant else nullcontext():
        for i in range(start, start + n):
            events.publish("job.update", {"jobId": f"j{i}", "kind": "test", "state": "done", "detail": {}})


async def _read(sub: events.Subscription, n: int) -> list[events.Event]:
    return [await asyncio.wait_for(sub.next(), 1) for _ in range(n)]


def _job_ids(got: list[events.Event]) -> list[str]:
    return [e.data["jobId"] for e in got]


# ── replay ────────────────────────────────────────────────────────────────────


async def test_a_reconnecting_client_receives_every_missed_event_once_and_in_order():
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, 3)
            seen = await _read(sub, 3)
        last_id = seen[-1].id

        _publish(A, 5, start=3)                        # missed while disconnected
        _publish(B, 4)                                 # someone else's: never replayed to A

        async with events.subscribe(last_id) as sub:
            _publish(A, 2, start=8)                    # live, after the reconnect
            got = await _read(sub, 7)
            with pytest.raises(asyncio.TimeoutError):  # nothing twice, nothing extra
                await asyncio.wait_for(sub.next(), 0.1)

    assert _job_ids(seen) == ["j0", "j1", "j2"]
    assert _job_ids(got) == [f"j{i}" for i in range(3, 10)]
    assert len({e.id for e in seen + got}) == 10


async def test_a_client_whose_gap_is_older_than_the_buffer_receives_the_reset_event():
    events.configure(replay_buffer=4)
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, 1)
            (first,) = await _read(sub, 1)

        _publish(A, 10, start=1)                       # more than the buffer holds

        async with events.subscribe(first.id) as sub:
            (reset,) = await _read(sub, 1)
            _publish(A, 1, start=11)
            (live,) = await _read(sub, 1)

        # The reset's id is a place to resume from: reconnecting with it replays, not resets.
        async with events.subscribe(reset.id) as sub:
            (again,) = await _read(sub, 1)

    assert reset.type == event_schema.RESET == "stream.reset"
    assert reset.data == {"reason": "gap", "lastEventId": first.id}
    event_schema.validate(reset.type, reset.data)
    assert live.data["jobId"] == "j11" and again.data["jobId"] == "j11"


async def test_an_id_this_stream_did_not_issue_gets_a_reset_not_silence():
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, 1)
            (real,) = await _read(sub, 1)
        for bad in ("0000beef-1", f"{real.id.split('-')[0]}-99", "not-an-id"):
            async with events.subscribe(bad) as sub:
                (reset,) = await _read(sub, 1)
            assert reset.type == "stream.reset" and reset.data["reason"] == "unknown_id"


async def test_a_replay_never_crosses_tenants():
    with tenancy.tenant_scope(B):
        async with events.subscribe() as sub:
            _publish(B, 1)
            (b_event,) = await _read(sub, 1)
    _publish(A, 3)
    with tenancy.tenant_scope(A):
        async with events.subscribe(b_event.id) as sub:    # B's id presented in A's scope
            (reset,) = await _read(sub, 1)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sub.next(), 0.1)
    assert reset.type == "stream.reset"


async def test_a_slow_subscriber_loses_nothing_the_buffer_still_holds():
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, events._MAX_QUEUE + 10)           # the queue dropped its oldest ten
            assert sub.qsize() == events._MAX_QUEUE
            got = await _read(sub, events._MAX_QUEUE + 10)
    assert _job_ids(got) == [f"j{i}" for i in range(events._MAX_QUEUE + 10)]


async def test_a_subscriber_that_falls_behind_the_buffer_is_told_with_a_reset():
    events.configure(replay_buffer=events._MAX_QUEUE)
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, events._MAX_QUEUE * 2)
            first = await asyncio.wait_for(sub.next(), 1)
            rest = await _read(sub, events._MAX_QUEUE)
    assert first.type == "stream.reset" and first.data["reason"] == "gap"
    assert _job_ids(rest) == [f"j{i}" for i in range(events._MAX_QUEUE, events._MAX_QUEUE * 2)]


async def test_streams_past_the_bound_are_let_go_and_their_ids_then_reset():
    events.configure(max_streams=2)
    keys = [tenancy.tenant_key(f"pl06-many-{i}") for i in range(3)]
    ids = {}
    for key in keys:
        with tenancy.tenant_scope(key):
            async with events.subscribe() as sub:
                _publish(key, 1)
                ids[key] = (await _read(sub, 1))[0].id
    assert len(events._streams) == 2 and keys[0] not in events._streams
    with tenancy.tenant_scope(keys[0]):
        async with events.subscribe(ids[keys[0]]) as sub:
            assert (await _read(sub, 1))[0].type == "stream.reset"


@pytest.mark.parametrize("bound", [1, 2, 3])
async def test_a_new_tenant_can_subscribe_when_every_kept_stream_has_a_subscriber(bound):
    """At the bound with nothing idle to let go, the stream being made is not the one trimmed
    (it has no subscriber yet): the subscription works, on every retry, and receives events."""
    events.configure(max_streams=bound)
    held = [tenancy.tenant_key(f"pl33-held-{i}") for i in range(bound)]
    newcomer = tenancy.tenant_key("pl33-newcomer")
    async with AsyncExitStack() as stack:
        for key in held:
            with tenancy.tenant_scope(key):
                await stack.enter_async_context(events.subscribe())
        for attempt in range(2):
            with tenancy.tenant_scope(newcomer):
                async with events.subscribe() as sub:
                    assert events._streams[newcomer] is sub.stream
                    _publish(newcomer, 1, start=attempt)
                    assert _job_ids(await _read(sub, 1)) == [f"j{attempt}"]
        assert set(held) <= set(events._streams)       # nobody listening was let go for it
    # Once they are idle again, the next stream made brings the count back to the bound.
    another = tenancy.tenant_key("pl33-another")
    with tenancy.tenant_scope(another):
        async with events.subscribe():
            assert len(events._streams) == bound and another in events._streams


async def test_at_a_bound_of_one_an_idle_stream_makes_way_for_the_one_being_made():
    events.configure(max_streams=1)
    with tenancy.tenant_scope(A):
        async with events.subscribe():
            pass
    with tenancy.tenant_scope(B):
        async with events.subscribe() as sub:
            assert list(events._streams) == [B] and events._streams[B] is sub.stream


async def test_a_dropped_stream_keeps_nothing_and_tells_whoever_is_still_connected():
    with tenancy.tenant_scope(A):
        async with events.subscribe() as sub:
            _publish(A, 3)
            (first,) = await _read(sub, 1)
            assert events.drop_stream(A) is True and A not in events._streams
            last = await asyncio.wait_for(sub.next(), 1)        # the two unread events are gone
            assert last.type == "stream.reset" and last.data["lastEventId"] == first.id
            event_schema.validate(last.type, last.data)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sub.next(), 0.1)
        _publish(A, 1, start=3)                                  # no stream: not kept
        async with events.subscribe(first.id) as again:          # a new stream, a new epoch
            assert (await _read(again, 1))[0].data["reason"] == "unknown_id"
    assert events.drop_stream(B) is False


async def test_an_event_for_a_tenant_with_no_stream_is_not_kept():
    _publish(A, 5)
    assert A not in events._streams


# ── the SSE route ─────────────────────────────────────────────────────────────


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/v1/events", "headers": raw,
                    "query_string": b""})


def _parse(frame: str) -> dict:
    fields = dict(line.split(": ", 1) for line in frame.strip().split("\n"))
    fields["data"] = json.loads(fields["data"])
    return fields


async def test_the_stream_sends_an_id_per_event_and_resumes_from_the_last_event_id_header():
    payload = {"jobId": "j0", "kind": "test", "state": "done", "detail": {}}
    with tenancy.tenant_scope(A):
        response = await events_router.events_stream(_request(), schema=None, lastEventId=None)
        frames = response.body_iterator
        first_frame = asyncio.ensure_future(frames.__anext__())
        await asyncio.sleep(0)                          # the stream subscribes on its first read
        events.publish("job.update", payload)
        events.publish("job.update", {**payload, "jobId": "j1"})
        first = _parse(await asyncio.wait_for(first_frame, 1))
        await frames.aclose()                           # the client goes away after one event

        response = await events_router.events_stream(_request({"Last-Event-ID": first["id"]}),
                                                     schema=1, lastEventId=None)
        replayed = _parse(await asyncio.wait_for(response.body_iterator.__anext__(), 1))
        await response.body_iterator.aclose()

    # Without ?schema the data is the payload exactly as before; the id is on its own line.
    assert first["event"] == "job.update" and first["data"] == payload and first["id"]
    # With ?schema=1 the data is the envelope.
    envelope = event_schema.Envelope.model_validate(replayed["data"])
    assert envelope.v == event_schema.SCHEMA_VERSION and envelope.id == replayed["id"] != first["id"]
    assert envelope.type == "job.update" and envelope.data["jobId"] == "j1"


async def test_the_stream_refuses_a_schema_version_it_does_not_serve():
    with pytest.raises(InvalidRequestError):
        await events_router.events_stream(_request(), schema=event_schema.SCHEMA_VERSION + 1,
                                          lastEventId=None)


async def test_an_idle_stream_still_pings():
    with tenancy.tenant_scope(A):
        frames = events_router.event_frames(ping_s=0.01)
        assert await asyncio.wait_for(frames.__anext__(), 1) == ": ping\n\n"
        await frames.aclose()


# ── meeting bus ───────────────────────────────────────────────────────────────


async def test_two_watchers_of_one_meeting_each_receive_all_events():
    with tenancy.tenant_scope(A):
        bus = meeting_bus.get_bus("pl06-two-watchers")
        async with bus.watch() as one, bus.watch() as two:
            for i in range(5):
                await meeting_bus.publish("pl06-two-watchers", "status", {"state": "recording", "i": i})
            got_one = [one.get_nowait()["data"]["i"] for _ in range(one.qsize())]
            got_two = [two.get_nowait()["data"]["i"] for _ in range(two.qsize())]
            assert bus.watcher_count() == 2
        assert bus.watcher_count() == 0
    assert got_one == got_two == [0, 1, 2, 3, 4]


async def test_a_watcher_that_leaves_does_not_take_events_from_the_others():
    with tenancy.tenant_scope(A):
        bus = meeting_bus.get_bus("pl06-leaver")
        async with bus.watch() as stays:
            async with bus.watch():
                await meeting_bus.publish("pl06-leaver", "summary.ready", {})
            await meeting_bus.publish("pl06-leaver", "status", {"state": "done"})
            assert [stays.get_nowait()["type"] for _ in range(stays.qsize())] == ["summary.ready", "status"]


# ── the schema ────────────────────────────────────────────────────────────────

_EVENTS_PUBLISH = re.compile(r"""\bevents\.publish\(\s*["']([^"']+)["']""")
_BUS_PUBLISH = re.compile(r"""meeting_bus\.publish\(\s*[^,]+,\s*["']([^"']+)["']""")
_HOST_PUBLISH = re.compile(r"""self\._publish(?:_device)?\(\s*["']([^"']+)["']""")


def _kinds(pattern: re.Pattern) -> set[str]:
    found = set()
    for path in APP_DIR.rglob("*.py"):
        found |= set(pattern.findall(path.read_text()))
    return found


def test_every_kind_the_engine_publishes_is_in_the_schema():
    stream = _kinds(_EVENTS_PUBLISH) | _kinds(_HOST_PUBLISH) | {delegate_ollama.TASK_EVENT}
    meeting = _kinds(_BUS_PUBLISH) | {"transcript.stable", "status", "summary.ready", "reply"}
    assert stream and stream <= set(event_schema.STREAM_EVENTS), stream - set(event_schema.STREAM_EVENTS)
    assert meeting <= set(event_schema.MEETING_EVENTS), meeting - set(event_schema.MEETING_EVENTS)


def test_the_schema_exports_one_version_an_envelope_and_a_model_per_kind():
    schema = event_schema.json_schema()
    assert schema["schemaVersion"] == event_schema.SCHEMA_VERSION == 1
    assert set(schema["streamEvents"]) == set(event_schema.STREAM_EVENTS)
    assert set(schema["meetingEvents"]) == set(event_schema.MEETING_EVENTS)
    defs = schema["$defs"]
    for ref in [schema["envelope"], *schema["streamEvents"].values(), *schema["meetingEvents"].values()]:
        assert ref.removeprefix("#/$defs/") in defs
    assert set(defs["Envelope"]["properties"]) == {"v", "id", "type", "sessionId", "seq", "data"}
    json.dumps(schema)                                  # a generator can read it as JSON


def test_nothing_under_voice_publishes_with_broadcast():
    voice = APP_DIR / "voice"
    for path in voice.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "broadcast":
                pytest.fail(f"{path.relative_to(APP_DIR)} publishes with broadcast")


def test_the_buffer_bounds_come_from_configuration(monkeypatch, tmp_path):
    from app.local.config import Config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOXSCRIBE_EVENTS_REPLAY_BUFFER", "64")
    monkeypatch.setenv("VOXSCRIBE_EVENTS_MAX_STREAMS", "8")
    cfg = Config.from_env()
    assert (cfg.events_replay_buffer, cfg.events_max_streams) == (64, 8)
    monkeypatch.setenv("VOXSCRIBE_EVENTS_REPLAY_BUFFER", "0")
    with pytest.raises(SystemExit):
        Config.from_env()
