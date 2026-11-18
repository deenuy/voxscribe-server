#!/usr/bin/env python3
"""Recognizer word error with and without capture clean-up (T12 acceptance: error does not rise).

Runs the baseline speech fixture, clean and mixed with fan-like noise at 20 dB and 10 dB SNR (the
mix from scripts/eval_live_vad.py), through three capture paths and one recognizer:

  raw       the microphone signal as it was before T12
  aec       echo cancellation only (nothing is playing, so this is the high-pass and the
            canceller's passthrough)
  aec_ns    echo cancellation then noise suppression, the shipped default

and reports word error rate per input and path, plus the clean-up cost per 32 ms capture chunk.
The recognizer is faster-whisper at the engine's configured model, one offline pass per file, so
the rows differ only in the capture path. Transcripts are not printed or stored, only counts.

Usage:
  uv run tools/voice_bench/capture_wer.py
  uv run tools/voice_bench/capture_wer.py --limit-seconds 120 --inputs fan10
  uv run tools/voice_bench/capture_wer.py --output results/t12-capture/capture_wer.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from app.local.audio_engine import AcousticEchoCanceller  # noqa: E402
from app.local.config import Config  # noqa: E402

WORKSPACE = HERE.parent
DEFAULT_FIXTURE = WORKSPACE / "scratch" / "fixtures-baseline" / "fixture-10min.wav"
RATE = 16_000
CHUNK = 512
INPUTS = {"clean": None, "fan20": 20.0, "fan10": 10.0}
PATHS = ("raw", "aec", "aec_ns")
NOISE_SEED = 0


def _eval_live_vad():
    spec = importlib.util.spec_from_file_location("eval_live_vad", HERE / "scripts" / "eval_live_vad.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over words (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def clean_up(audio: np.ndarray, path: str) -> tuple[np.ndarray, float]:
    """Run the capture path in engine-sized chunks; return the audio and the mean ms per chunk."""
    if path == "raw":
        return audio, 0.0
    aec = AcousticEchoCanceller(capture_rate=RATE, echo_cancellation=True,
                                noise_suppression=(path == "aec_ns"))
    if not aec.active:
        raise SystemExit("pywebrtc-audio is not installed: uv sync --extra aec")
    n = audio.size // CHUNK * CHUNK
    out = np.empty(n, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(0, n, CHUNK):
        out[i: i + CHUNK] = aec.process_capture(audio[i: i + CHUNK])
    return out, (time.perf_counter() - t0) * 1000 / (n // CHUNK)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--reference", type=Path, default=None, help="defaults to <fixture>.reference.txt")
    ap.add_argument("--inputs", default=",".join(INPUTS))
    ap.add_argument("--paths", default=",".join(PATHS))
    ap.add_argument("--limit-seconds", type=float, default=None,
                    help="score only the start of the fixture (the reference is then cut to match)")
    ap.add_argument("--model", default=None, help="faster-whisper model; defaults to the engine's asr_model")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    ev = _eval_live_vad()
    from voxscribe.speech.wav import read_wav
    speech, rate = read_wav(args.fixture)
    if rate != RATE:
        raise SystemExit(f"{args.fixture} is {rate} Hz, expected {RATE}")
    speech = speech.astype(np.float32)
    reference_path = args.reference or args.fixture.with_suffix(".reference.txt")
    ref_words = words(reference_path.read_text())
    if args.limit_seconds:
        keep = speech.size
        speech = speech[: int(args.limit_seconds * RATE)]
        ref_words = ref_words[: round(len(ref_words) * speech.size / keep)]

    model_name = args.model or Config().asr_model
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    rows = []
    for name in args.inputs.split(","):
        snr = INPUTS[name]
        audio = speech if snr is None else ev.mix(speech, snr, np.random.default_rng(NOISE_SEED))
        for path in args.paths.split(","):
            processed, ms_per_chunk = clean_up(audio, path)
            segments, _ = model.transcribe(processed, language="en", beam_size=1)
            hyp = words(" ".join(s.text for s in segments))
            errors = word_errors(ref_words, hyp)
            row = {"input": name, "path": path, "ref_words": len(ref_words), "hyp_words": len(hyp),
                   "errors": errors, "wer": round(errors / len(ref_words), 4),
                   "ms_per_chunk": round(ms_per_chunk, 3)}
            rows.append(row)
            print(f"{name:6s} {path:7s} wer={row['wer']:.4f} errors={errors} "
                  f"hyp_words={len(hyp)} ms_per_32ms_chunk={row['ms_per_chunk']}", flush=True)

    report = {"run_at": datetime.now(UTC).isoformat(), "fixture": args.fixture.name,
              "seconds": round(speech.size / RATE, 1), "model": model_name, "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
