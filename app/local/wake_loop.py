"""Always-on wake word loop running inside the voxscribe-engine process.

Enabled with --wake. Owns the mic stream for the lifetime of the process.
State machine: off → armed → listening → armed.

While armed, every 80 ms frame goes through the two-stage wake gate (voxscribe.wake.gate):
detector patience, a voice check, then a recognizer confirm on "hey vox". Each candidate is
recorded in wake_events, accepted or with its reject reason; only an accepted one goes on.

On an accepted wake the loop:
  1. Records a row in wake_events.
  2. Creates a meeting via store.create_meeting (same call the mic button uses).
  3. Starts the meeting via store.start_meeting (consent captured by arming).
  4. Pushes float32 mic frames (converted to PCM16 + source_tag=0) into an
     AudioRing with drop_oldest policy, drained by the same LiveTranscriber
     consumer the WebSocket path uses.
  5. On 6 s silence closes the ring, awaits the consumer, then calls
     store.stop_once + schedule_finalize — exactly what the WebSocket path does.

Lines where the wake loop calls the shared start and stop:
  start → await store.start_meeting(meeting_id)          # _run_session line ~130
  stop  → await store.stop_once(meeting_id)              # _run_session line ~195

The CAPTURE / DELIVER / wake_transcript path has been removed.
"""

import asyncio
import logging
import os
import re
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import Enum

from voxscribe.wake.gate import ASR_CONFIRM_PHRASE, FRAME_SAMPLES

log = logging.getLogger("app.local.wake")

_SAMPLE_RATE      = 16_000
_CHUNK_FRAMES     = 512         # 32 ms at 16 kHz
_RMS_FLOOR        = 0.005
_SILENCE_S        = 6.0         # end-of-turn silence in seconds
_CAP_S            = 300.0       # hard 5-minute cap per session
_CONSENT_VER      = "wake-v1"

_CHIME_ACK     = "/System/Library/Sounds/Glass.aiff"
_CHIME_END     = "/System/Library/Sounds/Pop.aiff"
_CHIME_ERR     = "/System/Library/Sounds/Basso.aiff"

# Conversation mode: how long to wait for the user to speak again between turns.
_CONV_IDLE_S = 20.0

# Phrases that end the conversation and return the loop to ARMED.
# Includes Whisper mishearings of "vox" (wox, wux, walks).
_STOP_PHRASES: dict[str, set[str]] = {
    "hey vox": {
        "thank you vox", "thanks vox",
        "thank you wox", "thanks wox",
        "thank you wux", "thanks wux",
        "thank you walks", "thanks walks",
        "goodbye vox", "bye vox", "goodbye wox", "bye wox",
        "hey vox stop", "vox stop", "wox stop", "wux stop",
        "stop listening", "stop vox", "stop wox",
    },
}


def _is_stop_phrase(text: str, wake_phrase: str) -> bool:
    """Return True if the transcribed turn signals the user wants to end the conversation."""
    norm = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    return any(p in norm for p in _STOP_PHRASES.get(wake_phrase, set()))

def _spec_text_changed(pre: str, final: str) -> bool:
    """Return True if the final transcript adds a content word relative to pre-flush text.

    Semantic drift gate (owner-decided 2026-10-17): accept speculation only if
    final has pre as a word-prefix and the addition contains no content word
    (noun, verb, number, negation). A content word would change the LLM's answer.
    """
    pre_words   = re.sub(r"[^\w\s]", " ", pre.lower()).split()
    final_words = re.sub(r"[^\w\s]", " ", final.lower()).split()
    if final_words[:len(pre_words)] != pre_words:
        return True  # words shifted — cancel
    added = [w for w in final_words[len(pre_words):] if w not in _FUNCTION_WORDS]
    return len(added) > 0

# Two-stage "hey vox" gate (T02): the detector named by VOXSCRIBE_WAKE_DETECTOR (default the
# open-vocabulary spotter), then a voice check and a recognizer confirm. Built once per process
# by app/local/voice/wake_setup.py; the gate itself is voxscribe.wake.gate.
# What a person must actually say; shown in the UI instead of the configured wake_phrase.
_TRIGGER_PHRASE = ASR_CONFIRM_PHRASE
_WAKE_EVENT_PHRASE = _TRIGGER_PHRASE.replace(" ", "_")   # wake_events.phrase, as before T02
_DETECTOR_NAME = os.environ.get("VOXSCRIBE_WAKE_DETECTOR", "").strip() or "openvocab"
_TRIGGER_DETECTOR = "trigger"    # wake_events.detector for the debug trigger route
_CHIME_DECAY_S  = 0.3            # after the acknowledgement chime ends, before capture resumes
_gate = None                     # WakeGate, built once per process

_SPEC_MIN_WORDS = 4   # don't speculate if the turn is fewer than this many words

# Function words that a flush may add without changing the utterance's meaning.
# A word NOT in this set added after flush cancels speculation (owner-decided 2026-10-17).
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "it", "this", "that", "i", "you", "my",
    "your", "we", "me", "s", "ll", "d", "re", "ve", "m", "t",
})


class _SpeculativeStream:
    """Reads an async token generator into a buffer as a background task.

    Allows the LLM stream to start before ASR flush completes.
    The async iterator interface drains the buffer first, then the live stream.

    Usage:
        buf = _SpeculativeStream(stream_reply(...))
        # do other async work (ASR flush, etc.)
        async for token in buf:  # reads accumulated tokens first
            ...
        # or: buf.cancel() if the speculation text was wrong
    """

    def __init__(self, source: "AsyncIterator[str]") -> None:
        self._q: asyncio.Queue[str | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._drain(source))

    async def _drain(self, source: "AsyncIterator[str]") -> None:
        try:
            async for token in source:
                await self._q.put(token)
        except Exception:
            pass
        await self._q.put(None)

    def cancel(self) -> None:
        self._task.cancel()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        token = await self._q.get()
        if token is None:
            raise StopAsyncIteration
        return token


class WakeState(str, Enum):
    OFF       = "off"
    ARMED     = "armed"
    LISTENING = "listening"
    ERROR     = "error"


# Module-level state — written by the loop, read by the router.
_state: dict = {
    "state": WakeState.OFF,
    "model": _DETECTOR_NAME,       # replaced by the detector's model_name once the gate is built
    "phrase": "hey vox",   # overwritten from VOXSCRIBE_WAKE_PHRASE at arm time
    "trigger": _TRIGGER_PHRASE,
    "meeting_id": None,    # set for duration of active session, None when armed
}


def get_state() -> dict:
    return dict(_state)


def set_state(s: WakeState) -> None:
    _state["state"] = s


async def run_wake_loop(app) -> None:
    """Outer wrapper: any unhandled exception resets to ARMED after 1 s."""
    log.info("wake-word-listener: starting")
    set_state(WakeState.ARMED)
    try:
        while True:
            if _state["state"] == WakeState.OFF:
                await asyncio.sleep(0.25)
                continue
            try:
                await _inner(app)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("wake-word-listener: crashed -- resetting in 1 s")
                set_state(WakeState.ERROR)
                await asyncio.sleep(1)
                set_state(WakeState.ARMED)
    except asyncio.CancelledError:
        log.info("wake-word-listener: stopped (cancelled)")
        set_state(WakeState.OFF)
        raise


_BARGE_RATIO = 2.5   # mic_rms / playback_rms threshold for barge-in detection


async def _inner(app) -> None:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("wake loop needs sounddevice and numpy") from exc

    settings = getattr(app.state, "settings", None)
    wake_phrase = getattr(settings, "wake_phrase", "hey vox").lower().strip()

    _state["phrase"] = wake_phrase
    gate = await _get_gate()
    _state["model"]  = gate.detector.model_name
    set_state(WakeState.ARMED)
    log.info("[Vox Initialized] armed detector=%s patience=%d threshold=%.2f; say %r to start",
             gate.detector.model_name, gate.detector.patience_frames, gate.config.threshold,
             _TRIGGER_PHRASE)
    _publish_wake_state(WakeState.ARMED)

    # Pre-warm reply model and Kokoro so first wake turn hits resident models.
    reply_model = getattr(settings, "voice_reply_model", "llama3.2")
    asyncio.create_task(_prewarm_llm(reply_model), name="llm-prewarm")
    asyncio.create_task(_prewarm_kokoro(), name="kokoro-prewarm")

    # Use AudioEngine.subscribe() if available (single mic stream per process).
    # Fall back to direct sd.InputStream when AudioEngine is not started.
    engine = getattr(app.state, "audio_engine", None)

    async def _run_with_engine(engine):
        audio_q = engine.subscribe()
        try:
            await _mic_loop(app, audio_q, wake_phrase, np, engine=engine)
        finally:
            engine.unsubscribe(audio_q)

    async def _run_with_direct_stream():
        loop = asyncio.get_running_loop()
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=512)

        def _cb(indata, frames, t, status):
            if status:
                log.debug("sounddevice status: %s", status)
            try:
                loop.call_soon_threadsafe(audio_q.put_nowait, indata[:, 0].copy())
            except asyncio.QueueFull:
                pass

        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=_CHUNK_FRAMES, callback=_cb):
            log.info("wake-word-listener: mic open (direct stream) phrase=%r", wake_phrase)
            await _mic_loop(app, audio_q, wake_phrase, np, engine=None)

    if engine is not None:
        log.info("wake-word-listener: mic via AudioEngine.subscribe()")
        await _run_with_engine(engine)
    else:
        log.info("wake-word-listener: mic via direct sd.InputStream")
        await _run_with_direct_stream()


async def _get_gate():
    """Build the wake gate once per process; model loading runs off the event loop."""
    global _gate
    if _gate is None:
        from app.local.voice.wake_setup import make_detector, make_gate

        def _build():
            return make_gate(make_detector(), phrase=_TRIGGER_PHRASE)

        log.info("wake-word-listener: loading wake gate detector=%s", _DETECTOR_NAME)
        _gate = await asyncio.to_thread(_build)
        log.info("wake-word-listener: wake gate ready detector=%s", _gate.detector.model_name)
    return _gate


class _PreRollQueue:
    """A session's audio source: the audio heard after the wake phrase, then live capture.

    Offers the subset of asyncio.Queue that _run_session and _drain use.
    """

    def __init__(self, pre_roll, live: asyncio.Queue, chunk_frames: int) -> None:
        from collections import deque
        self._pre = deque(pre_roll[i: i + chunk_frames]
                          for i in range(0, len(pre_roll), chunk_frames))
        self._live = live

    async def get(self):
        if self._pre:
            return self._pre.popleft()
        return await self._live.get()

    def get_nowait(self):
        if self._pre:
            return self._pre.popleft()
        return self._live.get_nowait()

    def empty(self) -> bool:
        return not self._pre and self._live.empty()


def _take_queued(q: asyncio.Queue) -> list:
    """Everything already queued, in order, without waiting."""
    out = []
    while not q.empty():
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


async def _record_wake_event(app, *, score: float, accepted: bool, reject_reason: str | None,
                             detector: str | None, speech_ratio: float | None,
                             meeting_id: str | None = None) -> None:
    """One wake_events row per candidate, accepted or not. Never raises."""
    now_z = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        async with app.state.store.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO wake_events (id, phrase, confidence, triggered_at, started_meeting_id, "
                "accepted, reject_reason, detector, speech_ratio) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), _WAKE_EVENT_PHRASE, float(score), now_z, meeting_id,
                 accepted, reject_reason, detector, speech_ratio),
            )
            await conn.commit()
    except Exception:
        log.exception("wake: failed to record wake event accepted=%s", accepted)


async def _mic_loop(app, audio_q: asyncio.Queue, wake_phrase: str, np, *, engine) -> None:
    """Two-stage wake gate over 80 ms frames (T02).

    Stage 1 is the configured WakeDetector with frame patience; stage 2 is a Silero voice check
    on the last second and a Whisper confirm on the last two seconds (voxscribe.wake.gate).
    Every candidate is written to wake_events with its score and reject reason. A rejected
    candidate plays no sound and opens no session. The peak stage 1 score is logged once a
    minute of audio. No transcript text is logged at any stage.
    """
    gate = await _get_gate()
    frame_buf = np.zeros(0, dtype=np.float32)

    while True:
        if _state["state"] == WakeState.OFF:
            log.info("wake-word-listener: disarmed -- closing mic")
            return

        chunk = await audio_q.get()
        if _capture_reset_pending:
            _apply_capture_reset(gate)
            frame_buf = np.zeros(0, dtype=np.float32)
        rms = float((chunk ** 2).mean() ** 0.5)

        # P1 barge-in: energy-over-echo-estimate gate stops playback.
        if engine is not None and engine.is_playing and rms > _RMS_FLOOR:
            pb_ref = engine.playback_ref()
            if pb_ref:
                pb_rms = float((np.array(pb_ref) ** 2).mean() ** 0.5)
                if rms > _BARGE_RATIO * max(pb_rms, 1e-6):
                    log.info("wake: barge-in detected mic_rms=%.4f pb_rms=%.4f", rms, pb_rms)
                    engine.cancel_playback()

        # Suppress detection while speaker is playing (prevents self-trigger W5).
        if engine is not None and engine.is_playing:
            continue

        # The gate scores exact 80 ms frames; capture delivers 32 ms chunks.
        frame_buf = np.concatenate([frame_buf, chunk.astype(np.float32, copy=False)])

        while frame_buf.shape[0] >= FRAME_SAMPLES:
            frame, frame_buf = frame_buf[:FRAME_SAMPLES], frame_buf[FRAME_SAMPLES:]
            # Off the loop: a candidate frame includes the voice check and the confirm pass.
            decision = await asyncio.to_thread(gate.process, frame)

            tele = gate.pop_telemetry()
            if tele is not None:
                log.info("wake: peak_score=%.3f frames=%d peak_rms=%.4f",
                         tele.peak_score, tele.frames, tele.peak_rms)
            if decision is None:
                continue

            if not decision.accepted:
                log.info("wake: candidate rejected detector=%s score=%.3f reason=%s speech_ratio=%s",
                         decision.detector, decision.score, decision.reject_reason,
                         "n/a" if decision.speech_ratio is None else f"{decision.speech_ratio:.2f}")
                await _record_wake_event(app, score=decision.score, accepted=False,
                                         reject_reason=decision.reject_reason,
                                         detector=decision.detector,
                                         speech_ratio=decision.speech_ratio)
                continue
            # Quiet hours: the wake is recorded as rejected, and nothing is played or opened.
            if await _rejected_for_quiet_hours(app, decision.score):
                frame_buf = np.zeros(0, dtype=np.float32)
                gate.reset()  # or the same phrase scores again on the next frames
                continue

            # B4.7: ignore trigger while already listening.
            if _state["state"] == WakeState.LISTENING:
                log.info("wake-word-listener: already LISTENING -- ignored (B4.7)")
                continue

            log.info("wake-word-listener: DETECTED detector=%s score=%.3f speech_ratio=%s "
                     "monotonic_ms=%d", decision.detector, decision.score,
                     "n/a" if decision.speech_ratio is None else f"{decision.speech_ratio:.2f}",
                     int(time.monotonic() * 1000))

            # Keep "hey vox, how are you" in one breath: the audio after the phrase end, what is
            # left in the frame buffer, and what queued during the confirm pass are all from
            # before the chime. Then play the chime and drop everything captured while it played
            # and decayed, so the chime is never transcribed and never reaches the gate.
            # (T03 moves cues into the audio engine and mutes them from capture instead.)
            pre_roll = np.concatenate([decision.pre_roll, frame_buf, *_take_queued(audio_q)])
            frame_buf = np.zeros(0, dtype=np.float32)
            await _play_chime_async(_CHIME_ACK)
            await asyncio.sleep(_CHIME_DECAY_S)
            _drain(audio_q)

            _publish_wake_state(WakeState.LISTENING)

            try:
                await _run_session(app, _PreRollQueue(pre_roll, audio_q, _CHUNK_FRAMES),
                                   confidence=decision.score, np=np, wake_phrase=wake_phrase,
                                   detector=decision.detector,
                                   speech_ratio=decision.speech_ratio)
            except Exception:
                log.exception("wake-word-listener: session failed")
                await _play_chime_async(_CHIME_ERR)
            finally:
                set_state(WakeState.ARMED)
                _publish_wake_state(WakeState.ARMED)
                log.info("[Vox Stopped] back to armed; say %r to start a new conversation",
                         _TRIGGER_PHRASE)
                _drain(audio_q)
                frame_buf = np.zeros(0, dtype=np.float32)
                # Detector state, held audio and patience run all predate the session.
                gate.reset()


async def _run_session(app, audio_q, confidence: float, np,
                        wake_phrase: str = "hey vox", *, detector: str = _TRIGGER_DETECTOR,
                        speech_ratio: float | None = None) -> None:
    """Multi-turn conversation: one meeting, many turns, until stop phrase or idle timeout.

    audio_q is the live capture queue, or a _PreRollQueue that yields the audio heard after the
    wake phrase first. detector and speech_ratio describe the accepted wake for wake_events; the
    debug trigger route passes neither.
    """
    from voxscribe.meeting.live import LiveTranscriber

    from app.local import meeting_bus
    from app.local.audio_engine import monotonic_ms as _mono_ms
    from app.local.audio_ring import AudioRing
    from app.local.end_of_turn import TurnDetector
    from app.local.finalize import schedule_finalize
    from app.local.models import Segment
    from app.local.store_meetings import MeetingStore

    set_state(WakeState.LISTENING)
    pool     = app.state.store.pool
    store    = MeetingStore(pool)
    asr      = getattr(app.state, "providers", {}).get("asr")
    settings = getattr(app.state, "settings", None)
    _engine      = getattr(app.state, "audio_engine", None)
    _echo_guard  = _engine.echo_guard if _engine is not None else None

    silence_needed = int(_SILENCE_S * _SAMPLE_RATE / _CHUNK_FRAMES)
    cap_frames     = int(_CAP_S    * _SAMPLE_RATE / _CHUNK_FRAMES)

    # Detect TurnDetector availability once before the loop.
    try:
        _test_td = TurnDetector()
        _test_td.reset()
        del _test_td
        use_turn_detector = True
        log.info("wake-word-listener: using S4 turn detector")
    except RuntimeError:
        use_turn_detector = False
        log.info("wake-word-listener: S4 unavailable, falling back to silence counter")

    meeting_id = str(uuid.uuid4())
    now_z      = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    await store.create_meeting(
        meeting_id=meeting_id,
        title=f"Wake session {datetime.now(UTC).strftime('%H:%M')}",
        origin="wake",
        client_request_id=f"wake-{meeting_id}",
    )
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE meetings SET consent_version=%s, consent_at=%s WHERE id=%s",
            (_CONSENT_VER, now_z, meeting_id),
        )
        await conn.commit()

    await _record_wake_event(app, score=confidence, accepted=True, reject_reason=None,
                             detector=detector, speech_ratio=speech_ratio, meeting_id=meeting_id)

    await store.start_meeting(meeting_id)
    _state["meeting_id"] = meeting_id
    log.info("[Vox Initialized] meeting_id=%s phrase=%r", meeting_id, wake_phrase)

    # Bounded conversation history: [{"role": "user"|"assistant", "content": ...}, ...]
    _history: list[dict] = []

    # LLM model for speculative start (read once outside the loop)
    _reply_model = getattr(settings, "voice_reply_model", "llama3.2")

    turn_n = 0
    try:
        # ── conversation loop: one iteration per user turn ──────────────────
        while True:
            turn_n += 1
            _turn_texts: list[str] = []

            ring = AudioRing(capacity_ms=60_000.0, policy="drop_oldest")
            transcriber: object | None = None
            if asr:
                transcriber = LiveTranscriber(
                    asr,
                    session_id=meeting_id,
                    vad_mode=getattr(settings, "live_vad_mode", "adaptive"),
                    max_segment_ms=getattr(settings, "live_max_segment_ms", 15000) or None,
                    partial_update_interval_ms=500,
                )

            # Consumer drains the ring and accumulates stable ASR segments.
            # Default-arg trick binds current-iteration objects by value.
            # turn_detector is passed in so the consumer can call update_text()
            # for completeness-aware EOT decisions (research rec. item 2).
            _turn_detector_ref: list = []  # mutable slot filled after TD is created

            async def _consumer(ring_t=ring, tr_t=transcriber, tt=_turn_texts,
                                 td_ref=_turn_detector_ref, turn_t=turn_n):
                while True:
                    frame = await ring_t.pop()
                    if frame is None:
                        break
                    if tr_t is None:
                        continue
                    src, data = frame[0], frame[1:]
                    try:
                        new_stable = await asyncio.to_thread(tr_t.push_audio, src, data)
                        for seg in new_stable:
                            _publish_wake_segment(meeting_id, turn_t, seg)
                        for seg in tr_t.pop_partials():
                            _publish_wake_segment(meeting_id, turn_t, seg)
                        for seg in new_stable:
                            db = Segment(id=seg.id, startMs=seg.start_ms, endMs=seg.end_ms,
                                         source=seg.source, speaker=seg.speaker,
                                         text=seg.text, state=seg.state)
                            try:
                                await store.upsert_segment(meeting_id, db)
                            except Exception:
                                log.exception("wake: failed to persist segment")
                            await meeting_bus.publish(meeting_id, "transcript.stable", {
                                "id": db.id, "startMs": db.startMs, "endMs": db.endMs,
                                "source": db.source, "speaker": db.speaker,
                                "text": db.text, "state": db.state,
                            })
                            if db.text:
                                tt.append(db.text)
                                # Feed running transcript to TurnDetector for
                                # completeness-aware end-of-turn decisions.
                                td = td_ref[0] if td_ref else None
                                if td is not None:
                                    try:
                                        td.update_text(" ".join(tt))
                                    except Exception:
                                        pass
                    except Exception:
                        log.exception("wake consumer: transcription error")

            consumer_task = asyncio.create_task(_consumer())

            if use_turn_detector:
                turn_detector = TurnDetector()
                turn_detector.reset()
                _turn_detector_ref.append(turn_detector)  # expose to consumer

            silence_run  = 0
            total_frames = 0
            # Turn detector clock: samples fed to it since reset(). Its speech events are stamped on
            # this stream timeline, so tick() must be too; engine uptime made every pause look hours long.
            stream_samples = 0

            # Inter-turn idle deadline: first turn uses a quick watchdog; subsequent
            # turns give the user _CONV_IDLE_S seconds to speak again.
            idle_deadline = time.monotonic() + ((_SILENCE_S + 2.0) if turn_n == 1
                                                 else _CONV_IDLE_S)

            log.info("[Vox Listening] turn=%d (say your question, or '%s' to stop)",
                     turn_n, "thank you vox")

            _turn_end_ms = 0
            timed_out    = False
            try:
                while True:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        chunk = await asyncio.wait_for(audio_q.get(),
                                                        timeout=min(remaining, 0.5))
                    except asyncio.TimeoutError:
                        timed_out = True
                        break

                    rms    = float((chunk ** 2).mean() ** 0.5)

                    # Reset the idle clock whenever audio above noise floor is heard.
                    if turn_n > 1 and rms > _RMS_FLOOR * 4:
                        idle_deadline = time.monotonic() + _CONV_IDLE_S

                    if _echo_guard is not None:
                        d_eg, _ = _echo_guard.observe(chunk, None)
                        if d_eg == "muted":
                            silence_run = 0
                            continue

                    pcm16       = (chunk * 32767).clip(-32768, 32767).astype("int16")
                    pcm16_bytes = pcm16.tobytes()
                    ring.push(bytes([0]) + pcm16_bytes)
                    total_frames += 1

                    if use_turn_detector:
                        stream_samples += len(pcm16)
                        stream_ms = stream_samples * 1000 // _SAMPLE_RATE
                        d = turn_detector.push_pcm16(pcm16_bytes, stream_ms)
                        if total_frames % 5 == 0:
                            d = turn_detector.tick(stream_ms) or d
                        if d is not None and d.closes_turn:
                            log.info("wake: turn %d closed action=%s silence_ms=%d",
                                     turn_n, d.action, d.silence_ms)
                            break
                    else:
                        silence_run = silence_run + 1 if rms < _RMS_FLOOR else 0
                        if silence_run >= silence_needed:
                            log.info("wake: turn %d end-of-silence frames=%d", turn_n, total_frames)
                            break

                    if total_frames >= cap_frames:
                        log.info("wake: turn %d hard cap frames=%d", turn_n, total_frames)
                        break

            except Exception:
                log.exception("wake: audio collection error turn=%d", turn_n)
                timed_out = True

            if timed_out:
                if turn_n > 1:
                    log.info("[Vox Stopped] idle timeout after %d turn(s) — returning to armed",
                             turn_n - 1)
                else:
                    log.info("[Vox Stopped] first-turn audio timeout")
                ring.close()
                consumer_task.cancel()
                break

            _turn_end_ms = _mono_ms()

            # ── Speculative LLM start + parallel ASR flush ───────────────────
            # Capture text already confirmed by ASR (pre-flush). Start the LLM
            # immediately so token generation overlaps the 2.5 s Whisper flush.
            # This collapses the sequential ASR-flush → LLM-start chain and
            # brings first-audio latency from ~4.7 s to ~2.0 s on warm models.
            _prefetch_text = " ".join(_turn_texts).strip()
            _pre_stream: _SpeculativeStream | None = None

            if (_prefetch_text
                    and len(_prefetch_text.split()) >= _SPEC_MIN_WORDS
                    and not _is_stop_phrase(_prefetch_text, wake_phrase)):
                from app.local.llm_adapter import stream_reply as _llm_stream
                _pre_stream = _SpeculativeStream(
                    _llm_stream(_prefetch_text, model=_reply_model,
                                history=_history if _history else None)
                )
                log.info("wake: speculative LLM started text=%r", _prefetch_text[:80])

            # Play chime then drain ring + flush ASR (flush is the slow path).
            await _play_chime_async(_CHIME_END)
            ring.close()
            try:
                await asyncio.wait_for(consumer_task, timeout=120.0)
            except asyncio.TimeoutError:
                consumer_task.cancel()
                log.warning("wake: consumer drain timeout turn=%d", turn_n)

            if transcriber is not None:
                try:
                    tail = await asyncio.to_thread(transcriber.flush)
                    for seg in tail:
                        _publish_wake_segment(meeting_id, turn_n, seg)
                        db = Segment(id=seg.id, startMs=seg.start_ms, endMs=seg.end_ms,
                                     source=seg.source, speaker=seg.speaker,
                                     text=seg.text, state=seg.state)
                        try:
                            await store.upsert_segment(meeting_id, db)
                        except Exception:
                            pass
                        await meeting_bus.publish(meeting_id, "transcript.stable", {
                            "id": db.id, "startMs": db.startMs, "endMs": db.endMs,
                            "source": db.source, "speaker": db.speaker,
                            "text": db.text, "state": db.state,
                        })
                        if db.text:
                            _turn_texts.append(db.text)
                except Exception:
                    log.warning("wake: flush failed turn=%d", turn_n)

            _asr_final_ms = _mono_ms()
            turn_text     = " ".join(_turn_texts).strip()

            if not turn_text:
                log.info("[Vox Stopped] empty turn — no speech detected (turn=%d)", turn_n)
                if _pre_stream:
                    _pre_stream.cancel()
                break

            if _is_stop_phrase(turn_text, wake_phrase):
                log.info("[Vox Stopped] stop phrase %r detected (turn=%d)", turn_text, turn_n)
                if _pre_stream:
                    _pre_stream.cancel()
                break

            # Validate speculative stream: semantic drift gate (owner-decided 2026-10-17).
            # Cancel if flush added a content word that would change the LLM's answer.
            if _pre_stream is not None:
                if _spec_text_changed(_prefetch_text, turn_text):
                    log.info("wake: speculation cancelled — content drift pre=%r final=%r",
                             _prefetch_text[:60], turn_text[:60])
                    _pre_stream.cancel()
                    _pre_stream = None
                else:
                    log.info("wake: speculation accepted — no content drift")

            # Await the reply so TTS finishes before we listen for the next turn.
            # (No barge-in: next turn starts only after speech completes.)
            _publish_wake("wake.reply", {"meetingId": meeting_id, "turn": turn_n,
                                         "phase": "thinking", "userText": turn_text, "text": ""})
            reply_text = await _run_reply_safe(
                app, meeting_id, turn_text, _turn_end_ms,
                asr_final_ms=_asr_final_ms, turn_n=turn_n,
                history=_history if _history else None,
                pre_stream=_pre_stream,
            )
            _publish_wake("wake.reply", {"meetingId": meeting_id, "turn": turn_n,
                                         "phase": "done", "userText": turn_text, "text": reply_text})

            # Append this turn to conversation history for context on next turn.
            if turn_text:
                _history.append({"role": "user", "content": turn_text})
            if reply_text:
                _history.append({"role": "assistant", "content": reply_text})

            # Discard any TTS echo buffered in the audio queue during playback.
            _drain(audio_q)

    finally:
        _state["meeting_id"] = None
        try:
            meeting, stopped_now = await store.stop_once(meeting_id)
        except Exception:
            log.exception("wake: stop_once failed meeting_id=%s", meeting_id)
            stopped_now = False
        if stopped_now:
            await meeting_bus.publish(meeting_id, "status", {"state": "processing"})
            schedule_finalize(app, meeting_id)
            log.info("wake: finalize scheduled meeting_id=%s turns=%d", meeting_id, turn_n)
        else:
            log.info("wake: session already stopped meeting_id=%s", meeting_id)


def _publish_wake(kind: str, data: dict) -> None:
    # Broadcast like wake_state: the wake loop runs outside any tenant scope, so a scoped event reaches no one.
    try:
        from app.local import events
        events.publish(kind, data, broadcast=True)
    except Exception:
        pass


def _publish_wake_segment(meeting_id: str, turn: int, seg) -> None:
    if not seg.text:
        return
    _publish_wake("wake.transcript", {
        "meetingId": meeting_id, "turn": turn,
        "segment": {"id": seg.id, "text": seg.text, "state": seg.state,
                    "startMs": seg.start_ms, "endMs": seg.end_ms},
    })


def _publish_wake_state(state: WakeState) -> None:
    try:
        from app.local import events
        events.publish("wake_state", dict(_state) | {"state": state.value}, broadcast=True)
    except Exception:
        pass


def _play_chime(path: str) -> None:
    try:
        subprocess.run(["afplay", path], check=False, timeout=3)
    except Exception:
        pass


async def _play_chime_async(path: str) -> None:
    try:
        await asyncio.to_thread(subprocess.run, ["afplay", path], check=False, timeout=3)
    except Exception:
        pass


async def _run_reply_safe(
    app,
    meeting_id: str,
    turn_text: str,
    turn_end_ms: int,
    *,
    asr_final_ms: int | None = None,
    turn_n: int = 1,
    history: "list[dict] | None" = None,
    pre_stream: "_SpeculativeStream | None" = None,
) -> str:
    """Run the reply chain and return the spoken reply text (empty string on error)."""
    try:
        from app.local.reply_chain import run_reply
        log.info("[Vox Thinking] turn=%d text=%r history_turns=%d",
                 turn_n, turn_text[:120], len(history) // 2 if history else 0)
        reply_text = await run_reply(
            app, meeting_id, turn_text, turn_end_ms,
            asr_final_ms=asr_final_ms, turn_n=turn_n,
            history=history, pre_stream=pre_stream,
        )
        log.info("[Vox Ready] turn=%d reply complete", turn_n)
        return reply_text or ""
    except Exception:
        log.exception("wake: reply chain failed meeting_id=%s", meeting_id)
        return ""


async def _prewarm_llm(model: str) -> None:
    """Send a one-token request so Ollama loads the reply model into Metal GPU memory.

    Sends the real system prompt so Ollama caches the prefix KV; subsequent
    requests reuse the cached context and skip prefix prefill (~100-200ms savings).
    Runs once at arm time. keep_alive in llm_adapter keeps it resident between turns.
    Non-critical: failure is logged and ignored.
    """
    from app.local.llm_adapter import stream_reply
    try:
        log.info("wake: pre-warming reply model=%s", model)
        async for _ in stream_reply("hi", model=model):
            break  # first token enough; model + system-prompt KV now resident
        log.info("wake: reply model warmed model=%s", model)
    except Exception:
        log.debug("wake: prewarm failed (non-critical) model=%s", model)


async def _prewarm_kokoro() -> None:
    """Synthesize a silent phrase so Kokoro-ONNX loads its model into memory.

    First synthesis after cold start pays a ~300-500ms load penalty; pre-warm
    eliminates that from the first real reply.
    Non-critical: failure is logged and ignored.
    """
    try:
        from app.local.tts_engine import prerender_phrase
        log.info("wake: pre-warming Kokoro TTS")
        await prerender_phrase("hm")  # _THINKING_PHRASE; already in cache if warmed
        log.info("wake: Kokoro TTS warmed")
    except Exception:
        log.debug("wake: Kokoro prewarm failed (non-critical)")


def _drain(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


# ----------------------------------------------------------------------
# Capture health and quiet hours (T06)

_capture_reset_pending = False   # set by the capture watchdog after it reopened the stream


def on_capture_health(event) -> None:
    """Receive a CaptureHealth event from the capture watchdog (app.local.voice.capture_watchdog).

    After a reopen the wake model still holds features from before the gap, so the next frame the
    armed loop reads resets it. Every event is also published for the app.
    """
    global _capture_reset_pending
    if event.action == "reopened":
        _capture_reset_pending = True
        log.info("wake: capture recovered issue=%s; wake model reset on next frame", event.issue)
    elif event.action == "reopen_failed":
        log.warning("wake: capture reopen failed issue=%s; watchdog will retry", event.issue)
    try:
        from app.local import events
        events.publish("capture_health", {"issue": event.issue, "action": event.action,
                                          "value": event.value}, broadcast=True)
    except Exception:
        pass


def _apply_capture_reset(gate) -> None:
    global _capture_reset_pending
    _capture_reset_pending = False
    gate.reset()


async def _rejected_for_quiet_hours(app, confidence: float) -> bool:
    """True when quiet hours are active; the wake is then stored with accepted=False."""
    settings = getattr(app.state, "settings", None)
    quiet = getattr(settings, "quiet_hours", None)
    if quiet is None or not quiet.active():
        return False
    log.info("wake: rejected (quiet hours) score=%.3f", confidence)
    now_z = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        async with app.state.store.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO wake_events (id, phrase, confidence, triggered_at, started_meeting_id, "
                "accepted, reject_reason) VALUES (%s, %s, %s, %s, NULL, FALSE, %s)",
                (str(uuid.uuid4()), "hey_vox", float(confidence), now_z, "quiet_hours"),
            )
            await conn.commit()
    except Exception:
        log.exception("wake: failed to record rejected wake")
    return True
