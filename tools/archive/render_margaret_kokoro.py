"""Render Margaret session audio using Kokoro TTS + room-tone gaps.

Replaces render_margaret.py (macOS say / Grandma voice) with Kokoro af_heart
at 0.85x speed.  Silence gaps are filled with room-tone noise at ~0.008 RMS
so the HeuristicDetector's floor-learner sees steady low-level energy instead
of digital silence, giving speech a clean SNR margin.

Writes: scratch/seniorcare/session_XX/session_XX.wav
Updates: scratch/seniorcare/manifest.json with sha256 per session.

Usage:
    uv run python tools/render_margaret_kokoro.py [--sessions 1-10] [--force]
    uv run python tools/render_margaret_kokoro.py --verify-sample session_01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import wave
from pathlib import Path

import numpy as np

SCRATCH = Path("/Users/deenuy/Documents/Workspace/Projects/voxscribe/scratch/seniorcare")
TARGET_SR = 16_000          # HeuristicDetector expects 16 kHz
KOKORO_SR = 24_000          # Kokoro native rate
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 0.85         # slower → more modulation headroom for S4 detector
ROOM_TONE_RMS = 0.008       # fills gaps; above _RMS_FLOOR=0.005, below speech SNR

log = logging.getLogger("render_margaret_kokoro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_kokoro = None


def _load_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    from pathlib import Path as _P
    model  = _P.home() / ".voxscribe" / "kokoro" / "onnx" / "model_quantized.onnx"
    voices = _P.home() / ".voxscribe" / "kokoro" / "voices.bin.npz"
    if not model.exists() or not voices.exists():
        raise FileNotFoundError(f"Kokoro model files not found at {model.parent}")
    from kokoro_onnx import Kokoro
    _kokoro = Kokoro(str(model), str(voices))
    log.info("Kokoro loaded voice=%s speed=%.2f", KOKORO_VOICE, KOKORO_SPEED)
    return _kokoro


def _synth(text: str) -> np.ndarray:
    """Return float32 array at KOKORO_SR."""
    k = _load_kokoro()
    samples, sr = k.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
    assert sr == KOKORO_SR, f"expected {KOKORO_SR} Hz, got {sr}"
    return samples.astype("float32")


def _resample(audio: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Integer-ratio resample via scipy or naive linear interpolation."""
    if src_hz == dst_hz:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_hz, dst_hz)
        return resample_poly(audio, dst_hz // g, src_hz // g).astype("float32")
    except ImportError:
        # Fallback: linear interpolation (low quality but good enough for SR)
        n_out = int(len(audio) * dst_hz / src_hz)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype("float32")


def _room_tone(dur_s: float) -> np.ndarray:
    """Pink-ish noise at ROOM_TONE_RMS, TARGET_SR samples."""
    n = int(dur_s * TARGET_SR)
    if n <= 0:
        return np.zeros(0, dtype="float32")
    rng = np.random.default_rng(seed=42)
    # 1/f noise approximation: white noise low-pass filtered
    white = rng.standard_normal(n).astype("float32")
    # Simple leaky integrator for pink-ish character
    out = np.zeros(n, dtype="float32")
    v = 0.0
    for i in range(n):
        v = 0.97 * v + 0.03 * float(white[i])
        out[i] = v
    # Normalise to target RMS
    cur = float((out ** 2).mean() ** 0.5)
    if cur > 1e-9:
        out *= ROOM_TONE_RMS / cur
    return out


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())


def render_session(session_id: str, force: bool = False) -> str | None:
    sid = session_id.zfill(2)
    sess_dir = SCRATCH / f"session_{sid}"
    truth_path = sess_dir / "truth.json"
    out_path = sess_dir / f"session_{sid}.wav"

    if out_path.exists() and not force:
        log.info("session_%s: already rendered, skipping (use --force)", sid)
        with open(out_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    if not truth_path.exists():
        log.warning("session_%s: truth.json not found, skipping", sid)
        return None

    with open(truth_path) as f:
        truth = json.load(f)

    spans = truth.get("speech_spans", [])
    total_dur = truth.get("duration_s", 300.0)

    if not spans:
        log.warning("session_%s: no speech_spans", sid)
        return None

    parts: list[np.ndarray] = []
    prev_end = 0.0

    for span in spans:
        start_s = float(span["start_s"])
        end_s   = float(span["end_s"])
        text    = span["text"]

        # Gap before this span → room tone
        gap = start_s - prev_end
        if gap > 0.05:
            parts.append(_room_tone(gap))

        # Speech: synthesise with Kokoro, resample to 16 kHz
        log.info("session_%s: synthesising span %.1f-%.1fs", sid, start_s, end_s)
        raw = _synth(text)
        resampled = _resample(raw, KOKORO_SR, TARGET_SR)
        parts.append(resampled)
        prev_end = end_s

    # Trailing room tone to reach total_dur
    tail = total_dur - prev_end
    if tail > 0.05:
        parts.append(_room_tone(tail))

    audio = np.concatenate(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(out_path, audio, TARGET_SR)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    dur = len(audio) / TARGET_SR
    log.info("session_%s: wrote %.1f s WAV → %s", sid, dur, out_path.name)
    return sha


def verify_s4(wav_path: Path, clip_s: float = 10.0) -> bool:
    """Feed a clip through HeuristicDetector; return True if any frame is speech."""
    import sys
    core = Path(__file__).parents[2] / "voxscribe-core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))

    from voxscribe.speech.heuristic import HeuristicDetector
    detector = HeuristicDetector()

    with wave.open(str(wav_path), "r") as wf:
        sr   = wf.getframerate()
        n    = min(int(clip_s * sr), wf.getnframes())
        raw  = wf.readframes(n)

    pcm16 = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    frames = detector.process(pcm16)
    speech_frames = sum(1 for f in frames if f.probability >= 0.5)
    pct = 100.0 * speech_frames / max(1, len(frames))
    log.info("verify_s4: %s → %d/%d speech frames (%.0f%%) in first %.0f s",
             wav_path.name, speech_frames, len(frames), pct, clip_s)
    return speech_frames > 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", default="1-10")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify-sample", metavar="SESSION_ID",
                   help="verify S4 fires on first 10s of this session (e.g. session_01)")
    args = p.parse_args()

    if args.verify_sample:
        sid = args.verify_sample.replace("session_", "").zfill(2)
        wav = SCRATCH / f"session_{sid}" / f"session_{sid}.wav"
        if not wav.exists():
            log.error("WAV not found: %s", wav)
            return
        ok = verify_s4(wav)
        print(f"S4 fires: {ok}")
        return

    # Parse session list
    session_ids: list[str] = []
    for part in args.sessions.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            session_ids.extend(f"{i:02d}" for i in range(int(lo), int(hi) + 1))
        else:
            session_ids.append(part.strip().zfill(2))

    manifest_path = SCRATCH / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    for sid in session_ids:
        sha = render_session(sid, force=args.force)
        if sha:
            key = f"session_{sid}"
            manifest.setdefault(key, {})
            manifest[key]["wav_sha256_kokoro"] = sha

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("manifest updated: %s", manifest_path)


if __name__ == "__main__":
    main()
