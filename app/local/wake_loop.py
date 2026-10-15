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
import struct
import subprocess
import time
import uuid
from datetime import UTC, datetime
from enum import Enum

log = logging.getLogger("app.local.wake")

_SAMPLE_RATE   = 16_000
_CHUNK_FRAMES  = 512         # 32 ms at 16 kHz
_RMS_FLOOR     = 0.005
_THRESHOLD     = 0.5         # OWW score to count as a hit
_PATIENCE      = 2           # consecutive hits required
_SILENCE_S     = 6.0         # end-of-turn silence in seconds
_CAP_S         = 300.0       # hard 5-minute cap per session
_CONSENT_VER   = "wake-v1"

_WAKE_MODEL    = "hey_jarvis_v0.1.onnx"
_CHIME_ACK     = "/System/Library/Sounds/Glass.aiff"
_CHIME_END     = "/System/Library/Sounds/Pop.aiff"
_CHIME_ERR     = "/System/Library/Sounds/Basso.aiff"


class WakeState(str, Enum):
    OFF       = "off"
    ARMED     = "armed"
    LISTENING = "listening"
    ERROR     = "error"


# Module-level state — written by the loop, read by the router.
_state: dict = {
    "state": WakeState.OFF,
    "model": _WAKE_MODEL,
    "phrase": "hey_vox",
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

    try:
        from openwakeword.model import Model as OWWModel
        from openwakeword.utils import download_models
    except ImportError as exc:
        raise RuntimeError("wake loop needs openwakeword") from exc

    download_models()
    log.info("wake-word-listener: loading model=%s threshold=%.2f patience=%d",
             _WAKE_MODEL, _THRESHOLD, _PATIENCE)
    oww = OWWModel(wakeword_models=[_WAKE_MODEL], inference_framework="onnx")
    log.info("wake-word-listener: model loaded state=ARMED")
    set_state(WakeState.ARMED)

    _publish_wake_state(WakeState.ARMED)

    # Pre-warm reply model and Kokoro so first wake turn hits resident models.
    settings = getattr(app.state, "settings", None)
    reply_model = getattr(settings, "voice_reply_model", "llama3.2")
    asyncio.create_task(_prewarm_llm(reply_model), name="llm-prewarm")
    asyncio.create_task(_prewarm_kokoro(), name="kokoro-prewarm")

    # Use AudioEngine.subscribe() if available (single mic stream per process).
    # Fall back to direct sd.InputStream when AudioEngine is not started.
    engine = getattr(app.state, "audio_engine", None)

    async def _run_with_engine(engine):
        audio_q = engine.subscribe()
        try:
            await _mic_loop(app, audio_q, oww, np, engine=engine)
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
            log.info("wake-word-listener: mic open (direct stream) state=ARMED")
            await _mic_loop(app, audio_q, oww, np, engine=None)

    if engine is not None:
        log.info("wake-word-listener: mic via AudioEngine.subscribe()")
        await _run_with_engine(engine)
    else:
        log.info("wake-word-listener: mic via direct sd.InputStream")
        await _run_with_direct_stream()


async def _mic_loop(app, audio_q: asyncio.Queue, oww, np, *, engine) -> None:
    """Core detection loop. Runs under either AudioEngine or direct stream."""
    above = 0

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

        if rms < _RMS_FLOOR:
            above = 0
            continue

        # Suppress wake detection while speaker is playing (prevents self-trigger).
        # This is the P1 near-field guard; real AEC comes in P4.
        if engine is not None and engine.is_playing:
            above = 0
            continue

        scores = await asyncio.to_thread(oww.predict, chunk)
        best = max(scores.values()) if scores else 0.0

        above = above + 1 if best >= _THRESHOLD else 0
        if above < _PATIENCE:
            continue

        above = 0
        triggered_ms = int(time.monotonic() * 1000)
        log.info("wake-word-listener: DETECTED score=%.3f model=%s monotonic_ms=%d",
                 best, _WAKE_MODEL, triggered_ms)

        # B4.7: ignore trigger while already listening
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
            await _run_session(app, audio_q, best, np)
        except Exception:
            log.exception("wake-word-listener: session failed")
            await _play_chime_async(_CHIME_ERR)
        finally:
            set_state(WakeState.ARMED)
            _publish_wake_state(WakeState.ARMED)
            log.info("wake-word-listener: back to ARMED")
            _drain(audio_q)


async def _run_session(app, audio_q: asyncio.Queue, confidence: float, np) -> None:
    """Create a meeting, run the shared ring+consumer path, stop on turn end."""
    from app.local.audio_ring import AudioRing
    from app.local.store_meetings import MeetingStore
    from app.local.finalize import schedule_finalize
    from app.local import meeting_bus
    from app.local.models import Segment
    from voxscribe.meeting.live import LiveTranscriber
    from app.local.end_of_turn import TurnDetector

    set_state(WakeState.LISTENING)
    pool    = app.state.store.pool
    store   = MeetingStore(pool)
    asr     = getattr(app.state, "providers", {}).get("asr")
    settings = getattr(app.state, "settings", None)

    meeting_id = str(uuid.uuid4())
    now_z      = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    # ── create meeting (same store call as mic button, origin="wake") ─────────
    await store.create_meeting(
        meeting_id=meeting_id,
        title=f"Wake session {datetime.now(UTC).strftime('%H:%M')}",
        origin="wake",
        client_request_id=f"wake-{meeting_id}",
    )

    # record consent (user opted in by arming)
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE meetings SET consent_version=%s, consent_at=%s WHERE id=%s",
            (_CONSENT_VER, now_z, meeting_id),
        )
        await conn.commit()

    # write wake_events row
    event_id = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO wake_events (id, phrase, confidence, triggered_at, started_meeting_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (event_id, "hey_vox", float(confidence), now_z, meeting_id),
        )
        await conn.commit()

    # ── start meeting (shared start) ─────────────────────────────────────────
    await store.start_meeting(meeting_id)   # ← shared start (see module docstring)
    log.info("wake-word-listener: session started meeting_id=%s wake_event=%s",
             meeting_id, event_id)

    ring = AudioRing(capacity_ms=60_000.0, policy="drop_oldest")

    transcriber = None
    if asr:
        transcriber = LiveTranscriber(
            asr,
            session_id=meeting_id,
            vad_mode=getattr(settings, "live_vad_mode", "adaptive"),
            max_segment_ms=getattr(settings, "live_max_segment_ms", 15000) or None,
        )

    _turn_texts: list[str] = []  # accumulate stable text for LLM input

    async def _publish_stable(segs):
        for seg in segs:
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
                _turn_texts.append(db.text)

    async def _consumer():
        while True:
            frame = await ring.pop()
            if frame is None:
                break
            if transcriber is None:
                continue
            src  = frame[0]
            data = frame[1:]
            try:
                new_stable = await asyncio.to_thread(transcriber.push_audio, src, data)
                await _publish_stable(new_stable)
            except Exception:
                log.exception("wake consumer: transcription error")

    consumer = asyncio.create_task(_consumer())

    # ── push mic frames into ring until turn ends or hard cap ────────────────
    # TurnDetector (parked S4 policy) replaces the fixed 6-second silence counter.
    # Falls back to the silence counter if the speech stack is not importable.
    try:
        turn_detector = TurnDetector()
        turn_detector.reset()
        use_turn_detector = True
        log.info("wake-word-listener: using S4 turn detector")
    except RuntimeError:
        use_turn_detector = False
        log.info("wake-word-listener: S4 turn detector unavailable, falling back to silence counter")

    silence_run    = 0
    silence_needed = int(_SILENCE_S * _SAMPLE_RATE / _CHUNK_FRAMES)
    cap_frames     = int(_CAP_S    * _SAMPLE_RATE / _CHUNK_FRAMES)
    total_frames   = 0

    from app.local.audio_engine import monotonic_ms as _mono_ms

    # Any exception after start_meeting must still call stop_once so the meeting
    # does not stay stuck in "recording" status forever (CK6 fix).
    _turn_end_ms: int = 0
    try:
        try:
            while True:
                chunk = await asyncio.wait_for(audio_q.get(), timeout=_SILENCE_S + 2.0)
                now_ms = _mono_ms()

                # float32 → PCM16, prepend source_tag byte (0 = mic)
                pcm16       = (chunk * 32767).clip(-32768, 32767).astype("int16")
                pcm16_bytes = pcm16.tobytes()
                frame_bytes = bytes([0]) + pcm16_bytes
                ring.push(frame_bytes)
                total_frames += 1

                if use_turn_detector:
                    decision = turn_detector.push_pcm16(pcm16_bytes, now_ms)
                    if total_frames % 5 == 0:
                        decision = turn_detector.tick(now_ms) or decision
                    if decision is not None and decision.closes_turn:
                        log.info("wake-word-listener: turn closed action=%s silence_ms=%d frames=%d",
                                 decision.action, decision.silence_ms, total_frames)
                        break
                else:
                    rms = float((chunk ** 2).mean() ** 0.5)
                    silence_run = silence_run + 1 if rms < _RMS_FLOOR else 0
                    if silence_run >= silence_needed:
                        log.info("wake-word-listener: end-of-turn silence frames=%d", total_frames)
                        break

                if total_frames >= cap_frames:
                    log.info("wake-word-listener: hard cap reached frames=%d", total_frames)
                    break

        except asyncio.TimeoutError:
            log.info("wake-word-listener: audio timeout -- ending session")

        _turn_end_ms = _mono_ms()
        await _play_chime_async(_CHIME_END)

        # drain consumer
        ring.close()
        try:
            await asyncio.wait_for(consumer, timeout=120.0)
        except asyncio.TimeoutError:
            consumer.cancel()
            log.warning("wake: consumer drain timeout meeting_id=%s", meeting_id)

        # flush tail words still in ASR buffer
        if transcriber is not None:
            try:
                tail = await asyncio.to_thread(transcriber.flush)
                await _publish_stable(tail)
            except Exception:
                log.warning("wake: flush failed meeting_id=%s", meeting_id)

        # Record when ASR is fully ready (flush done, turn_texts populated).
        # This splits the turn_end → llm_first_token gap into:
        #   [turn_end → asr_final] ASR/consumer drain  +  [asr_final → llm_first_token] LLM warm-up.
        _asr_final_ms = _mono_ms()

        # launch reply chain as background task (LLM -> gate -> TTS -> playback)
        turn_text = " ".join(_turn_texts).strip()
        asyncio.create_task(
            _run_reply_safe(app, meeting_id, turn_text, _turn_end_ms,
                            asr_final_ms=_asr_final_ms),
            name=f"reply-{meeting_id[:8]}",
        )

    finally:
        # ── stop meeting (shared stop) — always runs after start_meeting ─────
        try:
            meeting, stopped_now = await store.stop_once(meeting_id)  # ← shared stop
        except Exception:
            log.exception("wake: stop_once failed meeting_id=%s", meeting_id)
            stopped_now = False
        if stopped_now:
            await meeting_bus.publish(meeting_id, "status", {"state": "processing"})
            schedule_finalize(app, meeting_id)
            log.info("wake-word-listener: finalize scheduled meeting_id=%s", meeting_id)
        else:
            log.info("wake-word-listener: session already stopped meeting_id=%s", meeting_id)


def _publish_wake_state(state: WakeState) -> None:
    try:
        from app.local import events
        events.publish("wake_state", {
            "state": state.value, "model": _WAKE_MODEL, "phrase": "hey_vox"
        }, broadcast=True)
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


async def _run_reply_safe(app, meeting_id: str, turn_text: str, turn_end_ms: int,
                          asr_final_ms: int | None = None) -> None:
    try:
        from app.local.reply_chain import run_reply
        await run_reply(app, meeting_id, turn_text, turn_end_ms,
                        asr_final_ms=asr_final_ms)
    except Exception:
        log.exception("wake: reply chain failed meeting_id=%s", meeting_id)


async def _prewarm_llm(model: str) -> None:
    """Send a one-token request so Ollama loads the reply model into Metal GPU memory.

    Sends the real system prompt so Ollama caches the prefix KV; subsequent
    requests reuse the cached context and skip prefix prefill (~100-200ms savings).
    Runs once at arm time. keep_alive in llm_adapter keeps it resident between turns.
    Non-critical: failure is logged and ignored.
    """
    from app.local.llm_adapter import stream_reply, _SYSTEM_PROMPT
    try:
        log.info("wake: pre-warming reply model=%s", model)
        async for _ in stream_reply("hi", model=model, system_prompt=_SYSTEM_PROMPT):
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
