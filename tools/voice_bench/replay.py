#!/usr/bin/env python3
"""Offline wake-and-turn baseline harness.

Scores fixture audio files through the current wake model and the core turn
detector without touching the live engine or the microphone.  Prints a table
of recall, false-accept rate, and turn-close timing suitable for pasting into
ROOT-CAUSE or a benchmark row.

Usage
-----
  # Score default fixture directories through the shipped gate:
  uv run tools/voice_bench/replay.py

  # The pre-T02 single-frame trigger, for the baseline row:
  uv run tools/voice_bench/replay.py --model hey_jarvis_v0.1

  # Explicit paths:
  uv run tools/voice_bench/replay.py \\
    --positive ideascribe-app/docs/PRD/fixtures/ \\
    --negative scratch/voicebench/fixtures/ \\
    --turns scratch/voicebench/fixtures/ \\
    --output /tmp/baseline_$(date +%Y%m%d).json

  # Wake only, skip turn timing:
  uv run tools/voice_bench/replay.py --no-turns

  # T02 bake-off: each "hey vox" adapter through the full two-stage gate
  # (patience, Silero voice check, Whisper confirm). Positives default to hey_vox_*.
  uv run tools/voice_bench/replay.py --no-turns --model openvocab \\
    --positive ideascribe-app/docs/PRD/fixtures/ --negative scratch/voicebench/fixtures/ \\
    --heldout scratch/wake/heldout/ --reject scratch/wake/hey_jarvis/
  uv run tools/voice_bench/replay.py --no-turns --model oww ...

  # Stage 1 only (no voice check, no confirm), for the ablation rows:
  uv run tools/voice_bench/replay.py --no-turns --model openvocab --stage1-only

Relative paths that do not exist from the current directory are resolved against the
workspace root, so the commands above work from voxscribe-server/ or from the workspace.

Output
------
Printed to stdout and optionally to --output as JSON.

  WAKE BASELINE  2026-11-17  model=hey_jarvis_v0.1.onnx  threshold=0.50
  -----------------------------------------------------------------------
  positive     : 3 files  detections=N  recall=N/3  (NN%)
  negative     : 20 files  detections=N  rate=N.N per hour
  score range  : positive min=0.XXX max=0.XXX  negative max=0.XXX

  TURN TIMING BASELINE  detector=heuristic  policy=TURN
  -------------------------------------------------------
  fixtures     : N files
  close timing : mean=X.Xs  p50=X.Xs  p95=X.Xs
  (turn detector receives stream time, not engine uptime; clock bug irrelevant here)

Environment
-----------
  VOXSCRIBE_WAKE_DETECTOR  default --model: openvocab (shipped) or oww
  VOXSCRIBE_WAKE_MODEL     OWW model name or path used by --model oww (default: the trained
                           hey_vox head in app/local/voice/models/)
  VOXSCRIBE_WAKE_THRESHOLD  single-frame score threshold for the pre-T02 path (default: 0.50)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import wave
from pathlib import Path
from statistics import mean, quantiles
from typing import NamedTuple

log = logging.getLogger("replay")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # workspace root
_DEFAULT_POSITIVE = _REPO_ROOT / "ideascribe-app/docs/PRD/fixtures"
_DEFAULT_NEGATIVE = _REPO_ROOT / "scratch/voicebench/fixtures"
_DEFAULT_TURNS    = _REPO_ROOT / "scratch/voicebench/fixtures"

# Default: the detector the product ships (T02), through the full gate. Any other --model
# value is an openWakeWord model scored the pre-T02 way (single frame, no gate).
_WAKE_MODEL  = os.environ.get("VOXSCRIBE_WAKE_DETECTOR", "").strip() or "openvocab"
_THRESHOLD   = float(os.environ.get("VOXSCRIBE_WAKE_THRESHOLD", "0.50"))
_SAMPLE_RATE = 16000
_FRAME_MS    = 80
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000  # 1280
_INT16_SCALE = 32768.0

# Gate path: every file is replayed with silence-level noise around it so a short clip
# (the hey_vox fixtures are under 1 s) sits in a stream the way live audio does.
_GATE_MODELS = ("openvocab", "oww")
_DEFAULT_POSITIVE_GLOB = "hey_vox_*"
_PAD_LEAD_S = 1.0
_PAD_TAIL_S = 1.5
_PAD_NOISE_RMS = 1e-4           # about -80 dBFS: not digital silence, far below speech
_PAD_SEED = 0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _read_pcm16_mono(path: Path) -> bytes:
    """Read any WAV or AIFF file as 16-bit mono 16 kHz PCM bytes.

    Resampling and channel mixing are done with soundfile + numpy if available;
    otherwise the file is read raw and a warning is emitted if the format differs.
    """
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="int16", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1).astype(np.int16)
        else:
            data = data[:, 0]
        if sr != _SAMPLE_RATE:
            try:
                from math import gcd

                from scipy.signal import resample_poly
                g = gcd(sr, _SAMPLE_RATE)
                data = resample_poly(data, _SAMPLE_RATE // g, sr // g).astype(np.int16)
            except ImportError:
                log.warning("%s: sample rate %d != %d and scipy unavailable; skipping resample",
                            path.name, sr, _SAMPLE_RATE)
        return data.tobytes()
    except ImportError:
        pass

    # Fallback: stdlib wave (WAV only, no AIFF, no resample)
    if path.suffix.lower() in (".aiff", ".aif"):
        import aifc
        import audioop
        with aifc.open(str(path), "rb") as af:
            nch = af.getnchannels()
            sw  = af.getsampwidth()
            sr  = af.getframerate()
            raw = af.readframes(af.getnframes())
        if nch == 2:
            raw = audioop.tomono(raw, sw, 0.5, 0.5)
        if sw != 2:
            raw = audioop.lin2lin(raw, sw, 2)
        if sr != _SAMPLE_RATE:
            raw, _ = audioop.ratecv(raw, 2, 1, sr, _SAMPLE_RATE, None)
        return raw
    else:
        with wave.open(str(path), "rb") as wf:
            nch = wf.getnchannels()
            sw  = wf.getsampwidth()
            sr  = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if nch == 2:
            import audioop
            raw = audioop.tomono(raw, sw, 0.5, 0.5)
        if sw != 2:
            import audioop
            raw = audioop.lin2lin(raw, sw, 2)
        if sr != _SAMPLE_RATE:
            import audioop
            raw, _ = audioop.ratecv(raw, 2, 1, sr, _SAMPLE_RATE, None)
        return raw


def _audio_files(directory: Path, exts: tuple[str, ...] = (".wav", ".aiff", ".aif"),
                 pattern: str = "*") -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(pattern) if p.suffix.lower() in exts)


def _resolve(path: Path) -> Path:
    """Relative paths that do not exist here are taken from the workspace root."""
    if path.is_absolute() or path.exists():
        return path
    return _REPO_ROOT / path


# ---------------------------------------------------------------------------
# Wake scoring
# ---------------------------------------------------------------------------

class WakeResult(NamedTuple):
    path: Path
    detected: bool
    max_score: float
    n_frames: int
    detection_frame: int | None


def _score_file_oww(path: Path, model: object, threshold: float, model_key: str) -> WakeResult:
    """Score one file through the loaded OWW model, resetting before and after."""
    model.reset()  # type: ignore[attr-defined]
    pcm = _read_pcm16_mono(path)
    n_frames = len(pcm) // (2 * _FRAME_SAMPLES)
    max_score = 0.0
    detected = False
    detection_frame: int | None = None

    for i in range(n_frames):
        chunk_bytes = pcm[i * 2 * _FRAME_SAMPLES : (i + 1) * 2 * _FRAME_SAMPLES]
        import numpy as np
        # openWakeWord's melspectrogram expects int16-scaled samples, not [-1, 1] floats.
        chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
        scores = model.predict(chunk)  # type: ignore[attr-defined]
        score = float(scores.get(model_key, 0.0))
        if score > max_score:
            max_score = score
        if not detected and score >= threshold:
            detected = True
            detection_frame = i

    model.reset()
    return WakeResult(path=path, detected=detected, max_score=max_score,
                      n_frames=n_frames, detection_frame=detection_frame)


def run_wake_baseline(
    positive_dir: Path,
    negative_dir: Path,
    model_name: str,
    threshold: float,
    positive_glob: str = _DEFAULT_POSITIVE_GLOB,
) -> dict:
    try:
        from openwakeword.model import Model as OWWModel
    except ImportError:
        log.error("openwakeword not installed; cannot score wake fixtures")
        return {}

    try:
        oww = OWWModel(wakeword_models=[model_name], inference_framework="onnx")
    except Exception as exc:
        log.error("Could not load wake model %r: %s", model_name, exc)
        return {}
    model_key = next(iter(oww.models))  # the key OWW predicts under, e.g. "hey_jarvis_v0.1"

    pos_files = _audio_files(positive_dir, pattern=positive_glob)
    neg_files = _audio_files(negative_dir)

    if not pos_files and not neg_files:
        log.warning("No audio files found in positive=%s or negative=%s", positive_dir, negative_dir)
        return {}

    pos_results = [_score_file_oww(p, oww, threshold, model_key) for p in pos_files]
    neg_results = [_score_file_oww(p, oww, threshold, model_key) for p in neg_files]

    pos_detected = sum(1 for r in pos_results if r.detected)
    neg_detected = sum(1 for r in neg_results if r.detected)

    # False-accept rate: detections per hour of negative audio
    neg_duration_s = sum(len(_read_pcm16_mono(p)) / 2 / _SAMPLE_RATE for p in neg_files) if neg_files else 0.0
    neg_duration_h = neg_duration_s / 3600.0
    fa_per_hour = (neg_detected / neg_duration_h) if neg_duration_h > 0 else float("inf")

    pos_scores = [r.max_score for r in pos_results]
    neg_scores = [r.max_score for r in neg_results]

    return {
        "model": model_name,
        "threshold": threshold,
        "positive": {
            "files": len(pos_files),
            "detected": pos_detected,
            "recall_pct": round(100.0 * pos_detected / len(pos_files), 1) if pos_files else None,
            "scores": {f.path.name: round(f.max_score, 4) for f in pos_results},
            "min_score": round(min(pos_scores), 4) if pos_scores else None,
            "max_score": round(max(pos_scores), 4) if pos_scores else None,
        },
        "negative": {
            "files": len(neg_files),
            "detected": neg_detected,
            "duration_s": round(neg_duration_s, 1),
            "fa_per_hour": round(fa_per_hour, 2),
            "max_score": round(max(neg_scores), 4) if neg_scores else None,
        },
    }


# ---------------------------------------------------------------------------
# Two-stage gate (T02): the same gate the live wake loop builds
# ---------------------------------------------------------------------------

def _padded(path: Path, rng) -> tuple[object, int]:
    """File as float32 with noise-floor padding; returns (audio, sample where the clip ends)."""
    import numpy as np
    clip = np.frombuffer(_read_pcm16_mono(path), dtype=np.int16).astype(np.float32) / _INT16_SCALE
    lead = int(_PAD_LEAD_S * _SAMPLE_RATE)
    tail = int(_PAD_TAIL_S * _SAMPLE_RATE)
    out = (rng.standard_normal(lead + clip.shape[0] + tail) * _PAD_NOISE_RMS).astype(np.float32)
    out[lead: lead + clip.shape[0]] += clip
    return out, lead + clip.shape[0]


class _PeakRecorder:
    """Passes a WakeDetector through, remembering the highest score since the last reset."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self.patience_frames = inner.patience_frames
        self.peak = 0.0

    def reset(self) -> None:
        self.peak = 0.0
        self._inner.reset()

    def score(self, pcm_float32_mono) -> float:
        s = self._inner.score(pcm_float32_mono)
        self.peak = max(self.peak, s)
        return s


def _replay_file(gate, path: Path, rng) -> dict:
    audio, clip_end = _padded(path, rng)
    gate.reset()
    decisions = []
    n = audio.shape[0] // _FRAME_SAMPLES
    for i in range(n):
        frame = audio[i * _FRAME_SAMPLES: (i + 1) * _FRAME_SAMPLES]
        d = gate.process(frame)
        if d is not None:
            decisions.append(d)
    accepted = [d for d in decisions if d.accepted]
    # gate.ring.end_sample counts from the gate's creation; express decisions relative to this file.
    file_start = gate.ring.end_sample - n * _FRAME_SAMPLES
    first = accepted[0] if accepted else None
    return {
        "file": path.name,
        "accepted": len(accepted),
        "candidates": len(decisions),
        "reasons": [d.reject_reason or "accepted" for d in decisions],
        "peak_score": round(gate.detector.peak, 4),
        "speech_ratio": None if first is None or first.speech_ratio is None
        else round(first.speech_ratio, 3),
        "accept_after_clip_end_ms": None if first is None
        else int((first.at_sample - file_start - clip_end) * 1000 / _SAMPLE_RATE),
        "duration_s": round((audio.shape[0] - int((_PAD_LEAD_S + _PAD_TAIL_S) * _SAMPLE_RATE))
                            / _SAMPLE_RATE, 2),
    }


def _summarise(rows: list[dict], *, hours_from_rows: bool = False) -> dict:
    from collections import Counter
    files = len(rows)
    hit = sum(1 for r in rows if r["accepted"] > 0)
    out = {
        "files": files,
        "accepted_files": hit,
        "accepted_events": sum(r["accepted"] for r in rows),
        "candidates": sum(r["candidates"] for r in rows),
        "reasons": dict(Counter(x for r in rows for x in r["reasons"])),
        "rows": rows,
    }
    if files:
        out["recall_pct"] = round(100.0 * hit / files, 1)
    if hours_from_rows:
        dur = sum(r["duration_s"] for r in rows)
        out["duration_s"] = round(dur, 1)
        out["fa_per_hour"] = round(out["accepted_events"] / (dur / 3600.0), 2) if dur else None
    return out


def run_gate_bench(model: str, sets: dict[str, tuple[Path, str]], *, stage1_only: bool) -> dict:
    """Replay each set through make_detector(model) + make_gate(); see app/local/voice/wake_setup."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.local.voice.wake_setup import make_detector, make_gate

    detector = _PeakRecorder(make_detector(model))
    gate = make_gate(detector, voice=not stage1_only, confirm=not stage1_only)
    rng = np.random.default_rng(_PAD_SEED)
    result: dict = {"model": detector.model_name, "patience_frames": detector.patience_frames,
                    "stages": "stage1" if stage1_only else "stage1+voice+confirm",
                    "threshold": gate.config.threshold}
    for name, (directory, pattern) in sets.items():
        files = _audio_files(directory, pattern=pattern)
        if not files:
            continue
        rows = [_replay_file(gate, f, rng) for f in files]
        result[name] = _summarise(rows, hours_from_rows=(name == "negative"))
        result[name]["dir"] = str(directory)
    return result


def _print_gate(g: dict) -> None:
    print(f"\nWAKE GATE  model={g['model']}  stages={g['stages']}"
          f"  patience={g['patience_frames']}  threshold={g['threshold']}")
    print("-" * 70)
    for name in ("positive", "heldout", "reject", "negative"):
        s = g.get(name)
        if not s:
            continue
        line = (f"  {name:9s}: {s['files']} files  accepted={s['accepted_files']}/{s['files']}"
                f"  candidates={s['candidates']}  reasons={s['reasons']}")
        if name == "negative":
            line += f"  {s['duration_s']}s  fa/hour={s['fa_per_hour']}"
        print(line)
        if name == "positive":
            for r in s["rows"]:
                print(f"             {r['file']}: accepted={r['accepted']} peak={r['peak_score']}"
                      f" speech_ratio={r['speech_ratio']}"
                      f" accept_after_clip_end_ms={r['accept_after_clip_end_ms']}")


# ---------------------------------------------------------------------------
# Turn timing
# ---------------------------------------------------------------------------

def run_turn_baseline(turns_dir: Path) -> dict:
    """Score fixture files through the heuristic detector + TURN policy.

    Records frames from first speech to turn close (closes_turn=True) and
    reports the distribution.  Uses stream time (sample count), not wall clock,
    so the clock-domain bug (Defect C) does not affect this measurement.
    """
    try:
        import numpy as np
        from voxscribe.speech.heuristic import HeuristicDetector  # type: ignore[import]
        from voxscribe.speech.turns import SpeechTurnDriver  # type: ignore[import]
    except ImportError as exc:
        log.warning("voxscribe-core not installed; skipping turn baseline: %s", exc)
        return {}

    files = _audio_files(turns_dir)
    if not files:
        return {}

    close_times_ms: list[float] = []

    for path in files:
        pcm = _read_pcm16_mono(path)
        import numpy as np
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        detector = HeuristicDetector()
        driver   = SpeechTurnDriver(detector=detector)
        driver.reset()

        turn_open_ms: float | None = None

        hop = detector.hop_samples
        for i in range(0, len(samples) - hop + 1, hop):
            chunk = samples[i : i + hop]
            decisions = driver.process_audio(chunk)
            for d in decisions:
                if d.turn_open and turn_open_ms is None:
                    turn_open_ms = float(d.at_ms)
                if d.closes_turn and turn_open_ms is not None:
                    close_times_ms.append(float(d.at_ms) - turn_open_ms)
                    turn_open_ms = None

    base = {"detector": "heuristic", "policy": "TURN", "files": len(files)}
    if not close_times_ms:
        base["note"] = "no close events; fixtures may not include trailing silence"
        return base

    p50, p95 = quantiles(close_times_ms, n=100)[49], quantiles(close_times_ms, n=100)[94]
    return {
        **base,
        "fixtures_with_close": len(close_times_ms),
        "close_ms": {
            "mean": round(mean(close_times_ms), 1),
            "p50":  round(p50, 1),
            "p95":  round(p95, 1),
            "min":  round(min(close_times_ms), 1),
            "max":  round(max(close_times_ms), 1),
        },
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_wake(w: dict) -> None:
    if not w:
        print("  (wake baseline skipped)")
        return
    pos = w["positive"]
    neg = w["negative"]
    print(f"\nWAKE BASELINE  model={w['model']}  threshold={w['threshold']}")
    print("-" * 70)
    if pos["files"]:
        print(f"  positive  : {pos['files']} files  detected={pos['detected']}"
              f"  recall={pos['detected']}/{pos['files']}  ({pos['recall_pct']}%)")
        print("             scores: " + "  ".join(
            f"{k}={v}" for k, v in sorted(pos["scores"].items())
        ))
    else:
        print("  positive  : no files found")
    if neg["files"]:
        print(f"  negative  : {neg['files']} files  {neg['duration_s']}s"
              f"  detected={neg['detected']}  fa/hour={neg['fa_per_hour']}"
              f"  max_score={neg['max_score']}")
    else:
        print("  negative  : no files found")


def _print_turn(t: dict) -> None:
    if not t:
        print("  (turn baseline skipped)")
        return
    print(f"\nTURN TIMING BASELINE  detector={t.get('detector', '?')}  policy={t.get('policy', '?')}")
    print("-" * 70)
    print(f"  fixtures  : {t['files']} files  with close event: {t.get('fixtures_with_close', 0)}")
    cm = t.get("close_ms")
    if cm:
        print(f"  close ms  : mean={cm['mean']}  p50={cm['p50']}  p95={cm['p95']}"
              f"  min={cm['min']}  max={cm['max']}")
    print("  (stream time used; Defect C clock bug does not affect this measurement)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--positive", type=Path, default=_DEFAULT_POSITIVE,
                        help="directory of positive wake fixtures (default: %(default)s)")
    parser.add_argument("--negative", type=Path, default=_DEFAULT_NEGATIVE,
                        help="directory of negative (non-wake) fixtures (default: %(default)s)")
    parser.add_argument("--turns", type=Path, default=_DEFAULT_TURNS,
                        help="directory of turn-timing fixtures (default: %(default)s)")
    parser.add_argument("--model", default=_WAKE_MODEL,
                        help="openvocab or oww for the two-stage gate; any other value is an OWW "
                             "model name or path scored single-frame (default: %(default)s)")
    parser.add_argument("--positive-glob", default=_DEFAULT_POSITIVE_GLOB,
                        help="file pattern inside --positive (default: %(default)s)")
    parser.add_argument("--heldout", type=Path, default=None,
                        help="gate only: extra positives the trained model never saw")
    parser.add_argument("--reject", type=Path, default=None,
                        help="gate only: phrases that must never wake (e.g. hey jarvis clips)")
    parser.add_argument("--stage1-only", action="store_true",
                        help="gate only: skip the voice check and the recognizer confirm")
    parser.add_argument("--threshold", type=float, default=_THRESHOLD,
                        help="wake score threshold (default: %(default)s)")
    parser.add_argument("--no-turns", action="store_true",
                        help="skip the turn timing section")
    parser.add_argument("--output", type=Path, default=None,
                        help="write JSON results to this file")
    args = parser.parse_args(argv)

    print(f"\nreplay.py  positive={args.positive}  negative={args.negative}"
          f"  model={args.model}  threshold={args.threshold}")

    args.positive, args.negative, args.turns = (
        _resolve(args.positive), _resolve(args.negative), _resolve(args.turns))
    if args.model in _GATE_MODELS:
        sets = {"positive": (args.positive, args.positive_glob), "negative": (args.negative, "*")}
        if args.heldout:
            sets["heldout"] = (_resolve(args.heldout), "*")
        if args.reject:
            sets["reject"] = (_resolve(args.reject), "*")
        wake = run_gate_bench(args.model, sets, stage1_only=args.stage1_only)
    else:
        wake = run_wake_baseline(args.positive, args.negative, args.model, args.threshold,
                                 positive_glob=args.positive_glob)
    turn = {} if args.no_turns else run_turn_baseline(args.turns)

    if args.model in _GATE_MODELS:
        _print_gate(wake)
    else:
        _print_wake(wake)
    _print_turn(turn)

    result = {"wake": wake, "turn": turn}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\n  written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
