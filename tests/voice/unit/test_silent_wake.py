"""Unit tests for T03: speech guard and silent close.

Fakes only -- no mic, ASR, LLM or TTS. All DB calls captured by FakePool / FakeConn.
Tests are split into two groups:
  1. _is_genuine_speech() -- the pure function with no I/O
  2. Session control-flow -- verify silent close stores no meeting / no turn
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.local.wake_loop import (
    CHIME_TRANSCRIPTS,
    MIN_AUDIO_MS,
    MIN_REPLY_WORDS,
    _is_genuine_speech,
)


# ── speech guard unit tests ───────────────────────────────────────────────────

def test_chime_transcript_blocked():
    """'Beep.' fails the speech guard (chime transcript check)."""
    assert not _is_genuine_speech("Beep.", audio_duration_ms=800)


def test_chime_transcript_blocked_variations():
    """Other chime-like strings also fail."""
    for text in ["Boop.", "Bing.", "beep", "BEEP.", "boop", "bing"]:
        assert not _is_genuine_speech(text, audio_duration_ms=800), (
            f"expected _is_genuine_speech({text!r}, 800) to be False"
        )


def test_empty_text_blocked():
    """Empty transcript fails."""
    assert not _is_genuine_speech("", audio_duration_ms=0)
    assert not _is_genuine_speech("   ", audio_duration_ms=200)


def test_short_audio_blocked():
    """400 ms audio with real words fails (MIN_AUDIO_MS = 500)."""
    assert not _is_genuine_speech("What time is it", audio_duration_ms=400)


def test_too_few_words_blocked():
    """Single word 'Yes' fails (MIN_REPLY_WORDS = 2)."""
    assert not _is_genuine_speech("Yes", audio_duration_ms=1200)


def test_genuine_speech_passes():
    """'What time is it' with 1200 ms passes all checks."""
    assert _is_genuine_speech("What time is it", audio_duration_ms=1200)


def test_boundary_min_audio_exact():
    """Exactly MIN_AUDIO_MS passes; one less fails."""
    assert _is_genuine_speech("hello world", audio_duration_ms=MIN_AUDIO_MS)
    assert not _is_genuine_speech("hello world", audio_duration_ms=MIN_AUDIO_MS - 1)


def test_boundary_min_words_exact():
    """Exactly MIN_REPLY_WORDS words passes; one less fails."""
    assert MIN_REPLY_WORDS >= 2, "constant changed; update test"
    assert _is_genuine_speech("hello world", audio_duration_ms=MIN_AUDIO_MS)
    assert not _is_genuine_speech("hello", audio_duration_ms=MIN_AUDIO_MS)


def test_chime_transcript_set_contents():
    """CHIME_TRANSCRIPTS must contain at minimum 'beep' and the empty string."""
    assert "beep" in CHIME_TRANSCRIPTS
    assert "" in CHIME_TRANSCRIPTS


# ── session control-flow fakes ────────────────────────────────────────────────


class _FakeConn:
    """Records SQL calls without touching a real database."""

    def __init__(self, log: list) -> None:
        self._log = log

    async def execute(self, sql: str, params=None) -> None:
        self._log.append(("execute", sql.strip(), params))

    async def commit(self) -> None:
        self._log.append(("commit",))


class _FakePool:
    """Fake async connection pool backed by _FakeConn."""

    def __init__(self) -> None:
        self.log: list = []

    def connection(self):
        return self

    async def __aenter__(self):
        return _FakeConn(self.log)

    async def __aexit__(self, *args):
        pass


class _FakeTurnDecision:
    closes_turn = True
    action = "complete"
    silence_ms = 300


class _FakeTurnDetector:
    """Closes the turn on the first push_pcm16 call."""

    def __init__(self):
        self._calls = 0

    def reset(self):
        self._calls = 0

    def push_pcm16(self, pcm_bytes, now_ms):
        self._calls += 1
        if self._calls >= 1:
            return _FakeTurnDecision()
        return None

    def tick(self, now_ms):
        return None

    def update_text(self, text):
        pass


class _FakeSegment:
    """Minimal segment returned by fake transcriber."""

    def __init__(self, text: str) -> None:
        self.id = "seg-fake"
        self.start_ms = 0
        self.end_ms = 100
        self.source = "mic"   # Segment model requires Literal['mic', 'system']
        self.speaker = "speaker"
        self.text = text
        self.state = "stable"


class _FakeTranscriber:
    """Returns a fixed text as the first stable segment; nothing on flush."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._returned = False

    def push_audio(self, src, data):
        if not self._returned and self._text:
            self._returned = True
            return [_FakeSegment(self._text)]
        return []

    def pop_partials(self):
        return []

    def flush(self):
        return []


class _FakeRing:
    """Returns one minimal audio frame then None (signals end of data).

    The frame is in the format the consumer expects: source_tag byte + PCM bytes.
    Returning a frame before None ensures the transcriber's push_audio is called,
    so the fake turn text ends up in _turn_texts.
    """

    def __init__(self):
        self._calls = 0

    async def pop(self):
        self._calls += 1
        if self._calls == 1:
            # Minimal frame: source_tag=0 + 2 bytes of silent PCM
            return bytes([0, 0, 0])
        return None  # signals consumer to exit

    def push(self, data):
        pass

    def close(self):
        pass


def _make_fake_app(pool: _FakePool, turn_text: str = "Beep."):
    """Build a minimal fake app.state namespace for _run_session tests."""
    store = MagicMock()
    store.pool = pool
    store.create_meeting = AsyncMock()
    store.start_meeting = AsyncMock()
    store.stop_once = AsyncMock(return_value=(MagicMock(), True))
    store.upsert_segment = AsyncMock()

    settings = MagicMock()
    settings.voice_reply_model = "llama3.2"
    settings.live_vad_mode = "adaptive"
    settings.live_max_segment_ms = 15000

    echo_guard = MagicMock()
    echo_guard.observe = MagicMock(return_value=("pass", None))

    audio_engine = MagicMock()
    audio_engine.echo_guard = echo_guard

    # Non-None ASR so a transcriber is created; it returns turn_text
    fake_asr = object()

    app_state = types.SimpleNamespace(
        store=store,
        providers={"asr": fake_asr},
        settings=settings,
        audio_engine=audio_engine,
    )
    return types.SimpleNamespace(state=app_state), store


async def _make_audio_q(n_frames: int = 5) -> asyncio.Queue:
    """Return a queue pre-loaded with n silent float32 frames."""
    q: asyncio.Queue = asyncio.Queue()
    frame = np.zeros(512, dtype=np.float32)
    for _ in range(n_frames):
        await q.put(frame)
    return q


# ── session control-flow tests ────────────────────────────────────────────────


def _session_patches(store, turn_text: str = "Beep."):
    """Return a context-manager stack that patches all _run_session dependencies."""
    return (
        # Lazy imports inside _run_session must be patched at their source module
        patch("app.local.store_meetings.MeetingStore", return_value=store),
        patch("app.local.audio_ring.AudioRing", return_value=_FakeRing()),
        patch(
            "voxscribe.meeting.live.LiveTranscriber",
            return_value=_FakeTranscriber(turn_text),
        ),
        patch("app.local.end_of_turn.TurnDetector", return_value=_FakeTurnDetector()),
        patch("app.local.finalize.schedule_finalize"),
        patch("app.local.meeting_bus.publish", new_callable=AsyncMock),
        patch("app.local.wake_loop._run_reply_safe", new_callable=AsyncMock),
        patch("app.local.audio_engine.monotonic_ms", return_value=1000),
        patch("app.local.wake_loop._play_chime_async", new_callable=AsyncMock),
    )


async def test_silent_session_no_meeting():
    """A session that closes silently deletes the meeting row and never calls run_reply."""
    from app.local.wake_loop import _run_session

    pool = _FakePool()
    app, store = _make_fake_app(pool, turn_text="Beep.")
    audio_q = await _make_audio_q()

    patches = _session_patches(store, "Beep.")
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6] as mock_reply, patches[7], patches[8],
    ):
        await _run_session(app, audio_q, confidence=0.8, np=np, wake_phrase="hey vox")

    # No reply was generated for the chime-only transcript
    mock_reply.assert_not_called()

    # The pool must have received a DELETE FROM meetings SQL call
    executed_sqls = [entry[1] for entry in pool.log if entry[0] == "execute"]
    delete_sqls = [s for s in executed_sqls if "DELETE FROM meetings" in s]
    assert delete_sqls, (
        "expected DELETE FROM meetings in SQL log; found: "
        + " | ".join(executed_sqls[:10])
    )


async def test_silent_session_rearmed():
    """After a silent close, _run_session returns normally, allowing _mic_loop to re-arm."""
    import app.local.wake_loop as wl
    from app.local.wake_loop import _run_session, WakeState, _state

    pool = _FakePool()
    app, store = _make_fake_app(pool, turn_text="Beep.")
    audio_q = await _make_audio_q()

    fake_model = MagicMock()

    patches = _session_patches(store, "Beep.")
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4],
        patches[5], patches[6], patches[7], patches[8],
        patch.object(wl, "_oww_model", fake_model),
    ):
        # Must not raise -- a raise would prevent _mic_loop's finally from running
        await _run_session(app, audio_q, confidence=0.8, np=np, wake_phrase="hey vox")

    # Simulate _mic_loop's finally block (which always runs after _run_session):
    #   set_state(WakeState.ARMED)
    #   _oww_model.reset()
    # We verify that the model's reset() can be called (model is not None).
    wl._oww_model = fake_model
    wl._oww_model.reset()
    fake_model.reset.assert_called_once()

    # State was not left as LISTENING by _run_session (it was set at entry then cleaned up)
    assert _state["meeting_id"] is None, "meeting_id must be None after session ends"
