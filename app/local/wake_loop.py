"""Always-on wake word loop that runs inside the voxscribe-engine process.

Enabled with --wake. Opens one sounddevice stream for the lifetime of the
process. State machine: IDLE → WAKE → CAPTURE → DELIVER → IDLE.

One process owns the mic. If --wake is on, the engine's MicrophoneSource
in the transcription path must NOT be used simultaneously on the same device.
In practice, meetings on the mini stream audio from the mic via this loop's
wake path, not from a second InputStream.

ASR: transcription in DELIVER goes through app.state.providers["asr"] — the
same WhisperASRProvider the rest of the engine uses. This keeps one ASR stack.
Note: phrase-rule normalization, audit ledger and policy gate (streaming_listen)
require a StreamProcessor scoped to a meeting; wake transcripts arrive outside
any meeting, so those layers are skipped for this first iteration.

Stand-in wake model: "hey_jarvis" (built-in openWakeWord ONNX model). The real
"hey_vox" model is trained separately from TTS-synthesized clips and loaded by
setting WAKE_MODEL_NAME="hey_vox" once that file is available.
"""

import asyncio
import logging
import subprocess
from enum import Enum, auto

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 512        # 32 ms at 16 kHz — matches VADConfig.chunk_ms
_RMS_FLOOR = 0.005         # skip OWW scoring below this RMS (CPU saver)
_WAKE_THRESHOLD = 0.5      # openWakeWord score to count as a hit
_WAKE_PATIENCE = 2         # consecutive hits required to fire
_CAPTURE_SILENCE_S = 2.0   # trailing silence that ends utterance
_CAPTURE_MAX_S = 30.0      # hard cap

# Stand-in: hey_jarvis is a built-in openWakeWord model, no download needed.
# Replace with "hey_vox" once the custom model is trained.
_WAKE_MODEL_NAME = "hey_jarvis_v0.1.onnx"  # full filename required by OWWModel

# macOS built-in system sounds — no custom asset needed for the first loop test
_CHIME_ACK = "/System/Library/Sounds/Glass.aiff"
_CHIME_END = "/System/Library/Sounds/Pop.aiff"


class _State(Enum):
    IDLE = auto()
    CAPTURE = auto()


async def run_wake_loop(app) -> None:
    """Outer wrapper: any unhandled exception resets to IDLE after 1 s."""
    log.info("wake-word-listener: starting")
    try:
        while True:
            try:
                await _inner(app)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("wake-word-listener: crashed -- resetting to IDLE in 1s")
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        log.info("wake-word-listener: stopped (cancelled)")
        raise


async def _inner(app) -> None:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "wake loop needs sounddevice and numpy. "
            "Install voxscribe-core[wakeword] in the engine's venv."
        ) from exc

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as exc:
        raise RuntimeError(
            "wake loop needs openwakeword. "
            "Install voxscribe-core[wakeword] in the engine's venv."
        ) from exc

    # Ensure models are on disk (first-run download; no-op if already present).
    from openwakeword.utils import download_models
    log.info("wake-word-listener: verifying openWakeWord model files on disk")
    download_models()

    # Load models once — they live for the lifetime of this coroutine.
    log.info("wake-word-listener: loading model model=%s threshold=%.2f patience=%d",
             _WAKE_MODEL_NAME, _WAKE_THRESHOLD, _WAKE_PATIENCE)
    oww = OWWModel(wakeword_models=[_WAKE_MODEL_NAME], inference_framework="onnx")
    log.info("wake-word-listener: model loaded ok")

    # Use the engine's ASR provider — same WhisperASRProvider as the rest of the engine.
    asr = getattr(getattr(app, "state", None), "providers", {}).get("asr")
    if asr is None:
        log.warning("wake loop: no ASR provider on app.state — transcripts will be empty")

    loop = asyncio.get_running_loop()
    # Queue cap prevents unbounded growth if the event loop is busy.
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=512)

    def _audio_cb(indata, frames, t, status):
        if status:
            log.debug("sounddevice status: %s", status)
        # Bridge from PortAudio callback thread to the asyncio event loop.
        # call_soon_threadsafe is the only safe way to enqueue from a thread.
        try:
            loop.call_soon_threadsafe(audio_q.put_nowait, indata[:, 0].copy())
        except asyncio.QueueFull:
            pass  # overrun: drop oldest chunk

    with sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_CHUNK_FRAMES,
        callback=_audio_cb,
    ):
        log.info("wake-word-listener: mic open -- state=IDLE listening for wake phrase")
        state = _State.IDLE
        above_run = 0
        capture_chunks: list = []
        silence_run = 0
        silence_needed = int(_CAPTURE_SILENCE_S / (_CHUNK_FRAMES / _SAMPLE_RATE))
        cap_max = int(_CAPTURE_MAX_S / (_CHUNK_FRAMES / _SAMPLE_RATE))

        while True:
            chunk = await audio_q.get()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # ── IDLE: scan for wake phrase ────────────────────────────────────
            if state is _State.IDLE:
                if rms < _RMS_FLOOR:
                    # Near-silence: skip model scoring entirely to keep CPU near zero.
                    above_run = 0
                    continue

                scores = oww.predict(chunk)
                best = max(scores.values()) if scores else 0.0

                if best >= _WAKE_THRESHOLD:
                    above_run += 1
                else:
                    above_run = 0

                if above_run < _WAKE_PATIENCE:
                    continue

                # ── WAKE ─────────────────────────────────────────────────────
                above_run = 0
                log.info("wake-word-listener: DETECTED score=%.2f model=%s", best, _WAKE_MODEL_NAME)

                # Chime fires on-device BEFORE any network call.
                # Scoring pauses for the chime + 200 ms so the mini cannot
                # wake itself on the chime audio.
                _play_chime(_CHIME_ACK)
                await asyncio.sleep(0.2)

                # Drain queued audio that accumulated during chime playback.
                _drain(audio_q)

                # Notify all SSE subscribers (broadcast: not scoped to a tenant).
                from app.local import events
                events.publish("wake_detected", {"phrase": "hey_vox"}, broadcast=True)

                state = _State.CAPTURE
                capture_chunks = []
                silence_run = 0
                log.info("wake-word-listener: state=IDLE->CAPTURE max_duration=%.0fs", _CAPTURE_MAX_S)

            # ── CAPTURE: accumulate until silence or hard cap ─────────────────
            elif state is _State.CAPTURE:
                capture_chunks.append(chunk)

                if rms < _RMS_FLOOR:
                    silence_run += 1
                else:
                    silence_run = 0

                silence_hit = silence_run >= silence_needed
                cap_hit = len(capture_chunks) >= cap_max

                if not (silence_hit or cap_hit):
                    continue

                # ── DELIVER ───────────────────────────────────────────────────
                reason = "silence" if silence_hit else "hard_cap"
                log.info("wake-word-listener: state=CAPTURE->DELIVER reason=%s chunks=%d",
                         reason, len(capture_chunks))

                # Lower-tone end chime marks close of capture.
                _play_chime(_CHIME_END)

                samples = np.concatenate(capture_chunks)
                transcript = await loop.run_in_executor(
                    None, _transcribe, asr, samples
                )
                log.info("wake-word-listener: transcript=%r", transcript)

                from app.local import events
                events.publish("wake_transcript", {"text": transcript}, broadcast=True)

                # ── → IDLE ────────────────────────────────────────────────────
                log.info("wake-word-listener: state=DELIVER->IDLE ready for next phrase")
                state = _State.IDLE
                capture_chunks = []
                silence_run = 0
                above_run = 0


def _transcribe(asr, samples) -> str:
    """Transcribe using the engine's ASR provider (same stack as the rest of the engine)."""
    if asr is None:
        return ""
    try:
        chunks = asr.transcribe(samples.tolist(), _SAMPLE_RATE, "wake", initial_prompt=None)
        return " ".join(c.text.strip() for c in chunks).strip()
    except Exception:
        log.exception("transcription failed in wake loop")
        return ""


def _play_chime(path: str) -> None:
    try:
        subprocess.run(["afplay", path], check=False, timeout=3)
    except Exception:
        pass  # chime failure must never crash the loop


def _drain(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
