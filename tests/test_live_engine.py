"""Live socket behaviour against a real engine process.

The engine runs as a subprocess with fake providers on a throwaway schema, which is how
the desktop app runs it. The synchronous TestClient cannot share the async connection
pool with the app, so these tests talk to the real server over HTTP and WebSocket.
"""

import json
import math
import os
import signal
import struct
import subprocess
import time
import uuid

import httpx
import psycopg
import pytest
from websockets.sync.client import connect

BASE_DB = "postgresql://localhost:5432/ideascribe_dev"
TOKEN = "live-engine-test"


def tone(seconds: float, freq: float = 220.0, rate: int = 16000) -> bytes:
    return b"".join(
        struct.pack("<h", int(0.5 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(seconds * rate))
    )


def silence(seconds: float, rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


def _run_engine(**extra_env):
    """Start an engine with fake providers on a throwaway schema; yields its base URLs, then stops it."""
    from app.local.migrations import run_migrations
    import asyncio

    schema = f"t_live_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(BASE_DB, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    asyncio.run(run_migrations(BASE_DB, schema=schema))
    url = f"{BASE_DB}?options=-c%20search_path%3D{schema},public"
    env = dict(os.environ, VOXSCRIBE_TOKEN=TOKEN, **extra_env)
    proc = subprocess.Popen(
        ["uv", "run", "voxscribe-engine", "--port", "0", "--providers", "fake", "--database-url", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        ready = json.loads(proc.stdout.readline())
        port = ready["port"]
        yield {"http": f"http://127.0.0.1:{port}/v1", "ws": f"ws://127.0.0.1:{port}/v1"}
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        with psycopg.connect(BASE_DB, autocommit=True) as c:
            c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture(scope="module")
def engine():
    yield from _run_engine()


@pytest.fixture(scope="module")
def engine_heuristic():
    """The same engine with the live path segmenting through core's heuristic speech detector."""
    yield from _run_engine(VOXSCRIBE_LIVE_VAD_MODE="heuristic")


@pytest.fixture
def http(engine):
    with httpx.Client(base_url=engine["http"], headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30) as c:
        yield c


_started: list[str] = []


@pytest.fixture(autouse=True)
def stop_recordings(http):
    """The engine records one meeting at a time, so end whatever a test started."""
    yield
    for meeting_id in _started:
        status = http.get(f"/meetings/{meeting_id}").json().get("status")
        if status == "recording":
            http.post(f"/meetings/{meeting_id}/stop")
    _started.clear()
    # let finalization finish before the next test starts a meeting
    time.sleep(0.5)


def new_meeting(http, start=True) -> str:
    meeting_id = http.post("/meetings", json={"title": "live test"}).json()["id"]
    if start:
        _started.append(meeting_id)
        assert http.post(f"/meetings/{meeting_id}/start", json={"consentAcknowledged": True}).status_code == 200
    return meeting_id


def drain(ws, seconds: float) -> list[dict]:
    """Collect JSON events for a while."""
    events, end = [], time.time() + seconds
    while time.time() < end:
        try:
            msg = ws.recv(timeout=0.5)
        except TimeoutError:
            continue
        except Exception:
            break
        if isinstance(msg, str):
            events.append(json.loads(msg))
    return events


def speak(ws):
    for _ in range(20):
        ws.send(b"\x00" + tone(0.1))
        time.sleep(0.02)
    for _ in range(14):
        ws.send(b"\x00" + silence(0.1))
        time.sleep(0.02)


def test_refused_when_the_meeting_is_not_recording(engine, http):
    meeting_id = new_meeting(http, start=False)
    with pytest.raises(Exception):
        with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
            ws.recv(timeout=3)


def test_refused_without_a_valid_token(engine, http):
    meeting_id = new_meeting(http)
    for token in ("", "wrong"):
        with pytest.raises(Exception):
            with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={token}") as ws:
                ws.recv(timeout=3)


def test_status_ping_and_bad_json(engine, http):
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        assert json.loads(ws.recv(timeout=5)) == {"type": "status", "state": "recording"}
        ws.send(json.dumps({"type": "ping"}))
        assert json.loads(ws.recv(timeout=5))["type"] == "pong"
        ws.send("not json")
        err = json.loads(ws.recv(timeout=5))
        assert err["type"] == "error" and err["code"] == "invalid_message"


def test_audio_becomes_stable_segments_that_are_persisted(engine, http):
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        speak(ws)
        events = drain(ws, 4)
    stable = [e for e in events if e["type"] == "transcript.stable"]
    assert stable, "no stable segment arrived"
    persisted = http.get(f"/meetings/{meeting_id}/transcript").json()["segments"]
    assert {s["id"] for s in persisted} >= {e["segment"]["id"] for e in stable}


def test_a_new_note_is_announced_as_waiting_then_enhanced_with_a_citation(engine, http):
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        speak(ws)
        ws.send(json.dumps({"type": "note", "blockId": "b1", "text": "- the plan", "offsetMs": 500}))
        events = drain(ws, 8)
    blocks = [e["block"] for e in events if e["type"] == "enhanced.update" and e["block"]["blockId"] == "b1"]
    assert blocks, "the note produced no enhanced.update"
    assert blocks[0]["status"] in ("pending", "stable")
    stored = http.get(f"/meetings/{meeting_id}/enhanced").json()["blocks"]
    assert [b["blockId"] for b in stored] == ["b1"]
    assert {c["kind"] for b in stored for c in b["citations"]} <= {"transcript", "source"}


def test_assist_can_be_toggled_without_errors(engine, http):
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        ws.recv(timeout=5)
        ws.send(json.dumps({"type": "assist", "enabled": True}))
        ws.send(json.dumps({"type": "assist", "enabled": False}))
        ws.send(json.dumps({"type": "enhance", "live": False, "style": "concise"}))
        ws.send(json.dumps({"type": "ping"}))
        events = drain(ws, 2)
    assert [e["type"] for e in events] == ["pong"]


def test_stopping_over_http_finishes_the_meeting_with_a_summary(engine, http):
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        speak(ws)
        drain(ws, 3)
    assert http.post(f"/meetings/{meeting_id}/stop").status_code == 200
    deadline = time.time() + 40
    status = None
    while time.time() < deadline:
        status = http.get(f"/meetings/{meeting_id}").json()["status"]
        if status == "done":
            break
        time.sleep(1)
    assert status == "done"
    assert http.get(f"/meetings/{meeting_id}/summary").json()["status"] in ("ready", "none")


# ---- speech activity on the socket (ADR 0001, phase S2) ------------------------------------------------------------


def test_vad_messages_carry_time_probability_and_detector_in_the_default_mode(engine, http):
    """The default gate is still adaptive: it has no probability, so p is null and the detector is "energy"."""
    meeting_id = new_meeting(http)
    with connect(f"{engine['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
        speak(ws)
        events = drain(ws, 3)
    vad = [e for e in events if e["type"] == "vad"]
    assert [e["speech"] for e in vad] == [True, False]
    for e in vad:
        assert set(e) == {"type", "speech", "at_ms", "p", "detector"}
        assert isinstance(e["at_ms"], int) and e["p"] is None and e["detector"] == "energy"
    assert vad[0]["at_ms"] < vad[1]["at_ms"]
    assert not [e for e in events if e["type"] == "vad.level"]  # not asked for, and the energy gate has no level
    assert http.get("/health").json()["speech_activity"]["live_mode"] == "adaptive"


def _samantha_pcm() -> bytes:
    """Core's packaged golden clip: 5.9 s of a quiet room, speech from 0.9 s to 3.8 s, then quiet."""
    import numpy as np
    from voxscribe.speech.evaluate import vectors_dir
    from voxscribe.speech.wav import read_wav

    samples, rate = read_wav(vectors_dir() / "speech_samantha_clean.wav")
    assert rate == 16000
    return np.round(samples * 32767.0).astype("<i2").tobytes()


def _reference_events():
    from voxscribe.speech import SEGMENTATION, create_detector
    from voxscribe.speech.evaluate import run_item, vectors_dir
    from voxscribe.speech.wav import read_wav

    samples, rate = read_wav(vectors_dir() / "speech_samantha_clean.wav")
    run = run_item(samples, rate, create_detector("heuristic"), SEGMENTATION)
    return [(e.kind == "started", e.at_ms) for e, _ in run["events"]]


def _send_clip(ws, pcm: bytes, frame_ms: int = 100):
    step = 16000 * frame_ms // 1000 * 2
    for i in range(0, len(pcm), step):
        ws.send(b"\x00" + pcm[i : i + step])
        time.sleep(0.01)


def test_heuristic_mode_sends_detector_events_and_opt_in_levels(engine_heuristic, http):
    base = engine_heuristic
    with httpx.Client(base_url=base["http"], headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30) as h:
        assert h.get("/health").json()["speech_activity"]["live_mode"] == "heuristic"
        meeting_id = h.post("/meetings", json={"title": "heuristic"}).json()["id"]
        assert h.post(f"/meetings/{meeting_id}/start", json={"consentAcknowledged": True}).status_code == 200
        try:
            pcm = _samantha_pcm()
            with connect(f"{base['ws']}/meetings/{meeting_id}/live?token={TOKEN}&levels=1") as ws:
                _send_clip(ws, pcm)
                events = drain(ws, 3)
        finally:
            h.post(f"/meetings/{meeting_id}/stop")
            time.sleep(0.5)

    vad = [e for e in events if e["type"] == "vad"]
    # The socket reproduces the reference tracker exactly: same transitions at the same session times.
    assert [(e["speech"], e["at_ms"]) for e in vad] == _reference_events()
    for e in vad:
        assert e["detector"] == "heuristic" and isinstance(e["p"], float) and 0.0 <= e["p"] <= 1.0
    assert vad[0]["p"] >= 0.4 and vad[1]["p"] < 0.25  # the deciding frames of the segmentation policy

    levels = [e for e in events if e["type"] == "vad.level"]
    assert levels, "levels=1 asked for vad.level messages"
    assert all(set(e) == {"type", "at_ms", "p"} and 0.0 <= e["p"] <= 1.0 for e in levels)
    gaps = [b["at_ms"] - a["at_ms"] for a, b in zip(levels, levels[1:])]
    assert gaps and min(gaps) >= 100  # at most 10 per second of audio
    assert any(e["p"] >= 0.4 for e in levels) and any(e["p"] < 0.25 for e in levels)
    assert [e for e in events if e["type"] == "transcript.stable"], "the utterance reached the recognizer"
    assert [e["type"] for e in events][0] == "status"


def test_heuristic_mode_sends_no_levels_unless_asked(engine_heuristic):
    base = engine_heuristic
    with httpx.Client(base_url=base["http"], headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30) as h:
        meeting_id = h.post("/meetings", json={"title": "heuristic quiet"}).json()["id"]
        assert h.post(f"/meetings/{meeting_id}/start", json={"consentAcknowledged": True}).status_code == 200
        try:
            with connect(f"{base['ws']}/meetings/{meeting_id}/live?token={TOKEN}") as ws:
                _send_clip(ws, _samantha_pcm())
                events = drain(ws, 2)
        finally:
            h.post(f"/meetings/{meeting_id}/stop")
            time.sleep(0.5)
    assert [e["speech"] for e in events if e["type"] == "vad"] == [True, False]
    assert not [e for e in events if e["type"] == "vad.level"]
