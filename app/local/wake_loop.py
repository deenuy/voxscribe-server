"""Always-on wake word loop running inside the voxscribe-engine process.

Enabled with --wake. Owns the mic stream for the lifetime of the process.
State machine: off → armed → listening → armed.

On trigger the loop:
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
import re
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import Enum

log = logging.getLogger("app.local.wake")

_SAMPLE_RATE      = 16_000
_CHUNK_FRAMES     = 512         # 32 ms at 16 kHz
_RMS_FLOOR        = 0.005
_SILENCE_S        = 6.0         # end-of-turn silence in seconds
_CAP_S            = 300.0       # hard 5-minute cap per session
_CONSENT_VER      = "wake-v1"
_DEBOUNCE_S       = 2.0         # min seconds between consecutive triggers

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

_OWW_MODEL_NAME = "hey_jarvis"   # stand-in until owner trains "hey vox" (40 clips needed)
_OWW_THRESHOLD  = 0.5            # detection threshold; tuned per device on fixture set
_OWW_CHUNK      = 1280           # 80 ms at 16 kHz — openWakeWord requirement
_oww_model = None                # loaded once per process

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
    "model": _OWW_MODEL_NAME,
    "phrase": "hey vox",   # overwritten from VOXSCRIBE_WAKE_PHRASE at arm time
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
    _state["model"]  = _OWW_MODEL_NAME
    set_state(WakeState.ARMED)
    log.info("[Vox Initialized] armed phrase=%r detector=%s threshold=%.2f — say the phrase to start",
             wake_phrase, _OWW_MODEL_NAME, _OWW_THRESHOLD)
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


async def _mic_loop(app, audio_q: asyncio.Queue, wake_phrase: str, np, *, engine) -> None:
    """openWakeWord keyword detector.

    Feeds 80 ms (1280-sample) int16 chunks to the OWW model and triggers a
    session when the score exceeds _OWW_THRESHOLD. Sub-megabyte model, 0.5 ms
    inference per chunk — no ASR model runs while armed. Stand-in phrase is
    hey_jarvis until the owner's 40 "hey vox" clips are recorded and trained.
    """
    from openwakeword.model import Model

    global _oww_model
    if _oww_model is None:
        log.info("wake-word-listener: loading OWW model=%s", _OWW_MODEL_NAME)
        _oww_model = await asyncio.to_thread(
            Model, wakeword_models=[_OWW_MODEL_NAME], inference_framework="onnx"
        )
        log.info("wake-word-listener: OWW model loaded")

    chunk_buf: list[float] = []
    last_trigger_t = 0.0

    while True:
        if _state["state"] == WakeState.OFF:
            log.info("wake-word-listener: disarmed -- closing mic")
            return

        chunk = await audio_q.get()
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

        # Accumulate samples; OWW requires exactly _OWW_CHUNK int16 samples per call.
        chunk_buf.extend(chunk.tolist())

        while len(chunk_buf) >= _OWW_CHUNK:
            oww_chunk_f = np.array(chunk_buf[:_OWW_CHUNK], dtype=np.float32)
            chunk_buf   = chunk_buf[_OWW_CHUNK:]
            oww_i16     = (oww_chunk_f * 32767).clip(-32768, 32767).astype(np.int16)
            prediction  = await asyncio.to_thread(_oww_model.predict, oww_i16)
            score       = float(prediction.get(_OWW_MODEL_NAME, 0.0))

            if score < _OWW_THRESHOLD:
                continue

            # Debounce: ignore re-triggers within _DEBOUNCE_S of the last one.
            now = time.monotonic()
            if now - last_trigger_t < _DEBOUNCE_S:
                continue
            last_trigger_t = now

            triggered_ms = int(now * 1000)
            log.info("wake-word-listener: DETECTED model=%s score=%.3f monotonic_ms=%d",
                     _OWW_MODEL_NAME, score, triggered_ms)
            chunk_buf.clear()  # discard stale audio so it cannot re-trigger

            # B4.7: ignore trigger while already listening.
            if _state["state"] == WakeState.LISTENING:
                log.info("wake-word-listener: already LISTENING -- ignored (B4.7)")
                continue

            await _play_chime_async(_CHIME_ACK)
            chime_done_ms = int(time.monotonic() * 1000)
            log.info("wake-word-listener: chime latency=%d ms", chime_done_ms - triggered_ms)
            await asyncio.sleep(0.2)
            _drain(audio_q)

            _publish_wake_state(WakeState.LISTENING)

            try:
                await _run_session(app, audio_q, confidence=score, np=np,
                                   wake_phrase=wake_phrase)
            except Exception:
                log.exception("wake-word-listener: session failed")
                await _play_chime_async(_CHIME_ERR)
            finally:
                set_state(WakeState.ARMED)
                _publish_wake_state(WakeState.ARMED)
                log.info("[Vox Stopped] back to armed — say 'hey vox' to start a new conversation")
                _drain(audio_q)


async def _run_session(app, audio_q: asyncio.Queue, confidence: float, np,
                        wake_phrase: str = "hey vox") -> None:
    """Multi-turn conversation: one meeting, many turns, until stop phrase or idle timeout."""
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

    event_id = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO wake_events (id, phrase, confidence, triggered_at, started_meeting_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (event_id, "hey_vox", float(confidence), now_z, meeting_id),
        )
        await conn.commit()

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
                )

            # Consumer drains the ring and accumulates stable ASR segments.
            # Default-arg trick binds current-iteration objects by value.
            # turn_detector is passed in so the consumer can call update_text()
            # for completeness-aware EOT decisions (research rec. item 2).
            _turn_detector_ref: list = []  # mutable slot filled after TD is created

            async def _consumer(ring_t=ring, tr_t=transcriber, tt=_turn_texts,
                                 td_ref=_turn_detector_ref):
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

                    now_ms = _mono_ms()
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
                        d = turn_detector.push_pcm16(pcm16_bytes, now_ms)
                        if total_frames % 5 == 0:
                            d = turn_detector.tick(now_ms) or d
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
            reply_text = await _run_reply_safe(
                app, meeting_id, turn_text, _turn_end_ms,
                asr_final_ms=_asr_final_ms, turn_n=turn_n,
                history=_history if _history else None,
                pre_stream=_pre_stream,
            )

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
