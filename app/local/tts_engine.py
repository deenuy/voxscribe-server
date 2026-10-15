"""TTS engine: Kokoro-ONNX with macOS 'say' fallback.

Primary: kokoro-onnx (quantized model, af_heart voice, 24 kHz).
Fallback: macOS say (if Kokoro model files absent).

Kokoro model files (downloaded separately, not in repo):
  ~/.voxscribe/kokoro/onnx/model_quantized.onnx
  ~/.voxscribe/kokoro/voices.bin.npz

Voice: KOKORO_VOICE env var (default af_heart) or SAY_VOICE for say fallback.
Sample rate: 24 kHz mono float32 for both engines.

Streaming: synthesize_stream_and_play() yields audio chunk-by-chunk from
an async sentence generator, enabling first-audio before LLM finishes (V1-P01).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger("app.local.tts_engine")

_SAY_VOICE         = os.environ.get("SAY_VOICE", "Samantha")
_KOKORO_VOICE      = os.environ.get("KOKORO_VOICE", "af_heart")
_PLAYBACK_RATE_HZ  = 24_000
_SAY_TIMEOUT_S     = 15.0
_KOKORO_TIMEOUT_S  = 8.0

_KOKORO_MODEL  = Path(os.environ.get(
    "KOKORO_MODEL_PATH",
    Path.home() / ".voxscribe" / "kokoro" / "onnx" / "model_quantized.onnx",
))
_KOKORO_VOICES = Path(os.environ.get(
    "KOKORO_VOICES_PATH",
    Path.home() / ".voxscribe" / "kokoro" / "voices.bin.npz",
))

_PHRASES_DIR = Path(os.environ.get(
    "VOXSCRIBE_PHRASES_DIR",
    Path.home() / ".voxscribe" / "phrases",
))

_kokoro_instance = None
_kokoro_lock = None


def _kokoro_available() -> bool:
    return _KOKORO_MODEL.exists() and _KOKORO_VOICES.exists()


def _get_lock():
    global _kokoro_lock
    if _kokoro_lock is None:
        _kokoro_lock = asyncio.Lock()
    return _kokoro_lock


def _load_kokoro_sync():
    global _kokoro_instance
    if _kokoro_instance is not None:
        return _kokoro_instance
    try:
        from kokoro_onnx import Kokoro
        import numpy as np
        k = Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))
        _kokoro_instance = k
        log.info("tts_engine: Kokoro loaded model=%s", _KOKORO_MODEL.name)
        return k
    except Exception:
        log.exception("tts_engine: Kokoro load failed")
        return None


async def _get_kokoro():
    async with _get_lock():
        if _kokoro_instance is not None:
            return _kokoro_instance
        if not _kokoro_available():
            return None
        return await asyncio.to_thread(_load_kokoro_sync)


def _kokoro_synth_sync(kokoro, text: str, voice: str):
    """Blocking Kokoro synthesis. Returns (samples_f32, sample_rate)."""
    import numpy as np
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    return samples.astype("float32"), sr


async def _kokoro_synthesize(text: str, voice: str):
    """Async Kokoro synthesis. Returns float32 numpy array or None."""
    try:
        k = await _get_kokoro()
        if k is None:
            return None
        samples, sr = await asyncio.wait_for(
            asyncio.to_thread(_kokoro_synth_sync, k, text, voice),
            timeout=_KOKORO_TIMEOUT_S,
        )
        return samples
    except TimeoutError:
        log.warning("tts_engine: Kokoro synthesis timed out text=%r", text[:60])
        return None
    except Exception:
        log.exception("tts_engine: Kokoro synthesis failed text=%r", text[:60])
        return None


async def synthesize_and_play(
    text: str,
    audio_engine,
    *,
    voice: str = _KOKORO_VOICE,
) -> float:
    """Synthesize text and play through AudioEngine. Returns duration seconds."""
    if not text.strip():
        return 0.0

    try:
        import numpy as np
    except ImportError:
        log.error("tts_engine: numpy required")
        return 0.0

    samples = await _kokoro_synthesize(text, voice)

    if samples is None:
        log.info("tts_engine: falling back to say voice=%s", _SAY_VOICE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmppath = f.name
        try:
            await asyncio.to_thread(_say_to_wav, text, tmppath, voice=_SAY_VOICE)
            samples = await asyncio.to_thread(_read_wav_float32, tmppath, np)
        except Exception:
            log.exception("tts_engine: say fallback failed text=%r", text[:60])
            return 0.0
        finally:
            try:
                os.unlink(tmppath)
            except OSError:
                pass

    if samples is None or len(samples) == 0:
        return 0.0

    duration_s = len(samples) / _PLAYBACK_RATE_HZ
    try:
        await audio_engine.play(samples, block=True)
    except Exception:
        log.exception("tts_engine: playback failed")
    return duration_s


async def synthesize_stream_and_play(
    sentences,
    audio_engine,
    *,
    voice: str = _KOKORO_VOICE,
):
    """Synthesize and play an async iterable of sentences clause-by-clause.

    Yields (sentence, duration_s) for each sentence that played.
    First audio starts as soon as the first sentence is synthesized,
    before subsequent sentences are generated or synthesized (V1-P01).
    """
    async for sentence in sentences:
        if not sentence.strip():
            continue
        duration_s = await synthesize_and_play(sentence, audio_engine, voice=voice)
        yield sentence, duration_s


def _say_to_wav(text: str, path: str, *, voice: str) -> None:
    subprocess.run(
        [
            "say", "-v", voice,
            "--file-format", "WAVE",
            "--data-format", f"LEI16@{_PLAYBACK_RATE_HZ}",
            "-o", path,
            text,
        ],
        check=True,
        timeout=_SAY_TIMEOUT_S,
    )


def _read_wav_float32(path: str, np) -> object:
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples_i16 = np.frombuffer(raw, dtype="<i2")
    return samples_i16.astype("float32") / 32768.0


def _phrase_cache_path(text: str, voice: str) -> Path:
    key = hashlib.sha1(f"{voice}:{text}".encode()).hexdigest()[:16]
    return _PHRASES_DIR / f"{key}.wav"


async def prerender_phrase(text: str, *, voice: str = _SAY_VOICE) -> Path | None:
    """Pre-render a fixed phrase. Idempotent; returns cached path if it exists.

    Uses say (not Kokoro) for pre-rendered phrases so they work even when
    Kokoro model files are absent at startup.
    """
    path = _phrase_cache_path(text, voice)
    if path.exists():
        return path
    try:
        _PHRASES_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_say_to_wav, text, str(path), voice=voice)
        log.info("tts_engine: pre-rendered phrase len=%d -> %s", len(text), path.name)
        return path
    except Exception:
        log.exception("tts_engine: prerender failed text=%r", text[:60])
        return None


async def play_prerendered(phrase_path: Path, audio_engine) -> float:
    """Play a pre-rendered WAV file through AudioEngine. Returns duration seconds."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    try:
        samples = await asyncio.to_thread(_read_wav_float32, str(phrase_path), np)
    except Exception:
        log.exception("tts_engine: failed to read prerendered %s", phrase_path)
        return 0.0
    if samples is None or len(samples) == 0:
        return 0.0
    duration_s = len(samples) / _PLAYBACK_RATE_HZ
    try:
        await audio_engine.play(samples, block=True)
    except Exception:
        log.exception("tts_engine: prerendered playback failed")
    return duration_s
