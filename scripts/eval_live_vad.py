#!/usr/bin/env python
"""Evaluate the live path's turn detection modes on the baseline fixture (ADR 0001, phase S2 exit gate).

Drives core's LiveTranscriber directly, the way the live socket does (no socket, no database): PCM16 16 kHz
frames of 100 ms, pushed in order, ASR through the server's own LocalASRProvider (faster-whisper small.en on the
CPU, int8, with its own VAD filter on, exactly as in production). Partial decodes are off so that the ASR seconds
measure the stable path alone and the comparison between modes is not clouded by wall-clock pacing.

Inputs (all 16 kHz mono):
  clean      voxscribe-baseline/fixtures/fixture-10min.wav with its reference text
  fan20      the same mixed with fan-like noise at 20 dB SNR (SNR over the active speech frames)
  fan10      the same at 10 dB SNR
  noise      3 minutes with no speech: the fan, keyboard-like clicks, and a mains hum that starts at 90 s

Modes: adaptive (the current default energy gate), silero and heuristic (core's detectors under the
SEGMENTATION policy).

Metrics per mode and input:
  word_recall         unique reference words found in the stable transcript (lowercase, non-word
                      characters to space), 0..1
  fourgram_recall     unique reference 4-grams found in the stable transcript, the P0 harness rule, 0..1
  noise_segments      stable segments sharing no 4-gram with the reference (likely hallucinations);
                      segments shorter than four words have no 4-gram and count here too
  segments            stable segments produced
  asr_seconds         wall seconds inside the recognizer
  utterances          utterances the gate opened
  speech_seconds      audio handed to utterances (pre-roll and hangover included)
  audio_seconds       audio pushed
  wall_seconds        the whole run

Usage:
  python scripts/eval_live_vad.py [--modes adaptive,silero,heuristic] [--inputs clean,fan20,fan10,noise]
                                  [--out results/eval-live-vad.json] [--work-dir results/eval-live-vad-inputs]
                                  [--limit-seconds N]  (shorten every input, for a smoke run)
The JSON report is written after every run, so a partial run is still usable. `--markdown` prints the
comparison table for docs/eval-live-vad.md from an existing report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BASELINE = Path("/Users/deenuy/Documents/Workspace/voxscribe-baseline/fixtures")
FIXTURE = BASELINE / "fixture-10min.wav"
REFERENCE = BASELINE / "fixture-10min.reference.txt"
MANIFEST = BASELINE / "fixture-10min.manifest.json"

RATE = 16000
FRAME_MS = 100
MODES = ("adaptive", "silero", "heuristic")
INPUTS = ("clean", "fan20", "fan10", "noise")
NOISE_SECONDS = 180.0
HUM_START_S = 90.0

log = logging.getLogger("eval_live_vad")


# ---- scoring (the P0 harness rules) ----------------------------------------------------------------------------


def words_of(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def fourgrams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def score(reference: str, segments: list[str]) -> dict:
    ref_words = words_of(reference)
    ref_set = set(ref_words)
    ref_grams = fourgrams(ref_words)
    transcript = " ".join(segments)
    trans_words = words_of(transcript)
    trans_grams = fourgrams(trans_words)
    noise = [s for s in segments if not (fourgrams(words_of(s)) & ref_grams)]
    short = [s for s in noise if len(words_of(s)) < 4]
    return {
        "word_recall": round(len(ref_set & set(trans_words)) / len(ref_set), 4) if ref_set else None,
        "fourgram_recall": round(len(ref_grams & trans_grams) / len(ref_grams), 4) if ref_grams else None,
        "fourgram_precision": round(len(ref_grams & trans_grams) / len(trans_grams), 4) if trans_grams else None,
        "noise_segments": len(noise),
        # A segment under four words has no 4-gram, so it counts above whatever it says; most are fragments
        # at a cut or a chunk boundary ("it.", "See roadmap"). The long ones are the likely hallucinations.
        "short_segments": len(short),
        "noise_segments_long": len(noise) - len(short),
        "noise_segment_texts": [s for s in noise if len(words_of(s)) >= 4],
        "segments": len(segments),
        "transcript_words": len(trans_words),
    }


# ---- inputs ------------------------------------------------------------------------------------------------------


def read_fixture() -> np.ndarray:
    from voxscribe.speech.wav import read_wav

    samples, rate = read_wav(FIXTURE)
    if rate != RATE:
        raise SystemExit(f"{FIXTURE} is {rate} Hz, expected {RATE}")
    return samples.astype(np.float32)


def fan_noise(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Fan-like noise: broadband hiss shaped toward low frequencies, with a slow blade-rate flutter.

    White noise through two cascaded one-pole low-passes (about 400 Hz) plus a tenth of unfiltered hiss, times
    (1 + 0.15 sin 2 pi 17 t). Unit RMS; the caller scales it.
    """
    n = int(seconds * RATE)
    white = rng.standard_normal(n).astype(np.float64)
    a = math.exp(-2 * math.pi * 400.0 / RATE)
    low = white.copy()
    try:
        from scipy.signal import lfilter
    except ImportError:
        lfilter = None
    for _ in range(2):
        if lfilter is not None:
            low = lfilter([1 - a], [1, -a], low)
            continue
        out = np.empty_like(low)
        acc = 0.0
        for i in range(n):  # sequential IIR; 11 minutes of audio takes about ten seconds per pass
            acc = a * acc + (1 - a) * low[i]
            out[i] = acc
        low = out
    t = np.arange(n) / RATE
    fan = (low / (np.sqrt(np.mean(low**2)) + 1e-12)) + 0.1 * white
    fan *= 1.0 + 0.15 * np.sin(2 * math.pi * 17.0 * t)
    return (fan / (np.sqrt(np.mean(fan**2)) + 1e-12)).astype(np.float32)


def active_rms(x: np.ndarray, frame_ms: int = 20, floor_db: float = -45.0) -> float:
    """RMS over the frames that carry signal (above `floor_db` dBFS), so pauses do not dilute the speech level."""
    n = RATE * frame_ms // 1000
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = rms[rms > 10 ** (floor_db / 20)]
    return float(np.sqrt(np.mean(active**2))) if active.size else float(np.sqrt(np.mean(x**2)))


def mix(speech: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = fan_noise(len(speech) / RATE, rng)
    level = active_rms(speech) / (10 ** (snr_db / 20))
    out = speech + noise * level
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def noise_only(seconds: float, speech_level: float, rng: np.random.Generator) -> np.ndarray:
    """Fan at the 10 dB SNR level, keyboard-like clicks every 0.3 to 1.5 s, and a 60 Hz hum with harmonics
    from `HUM_START_S` on, at the speech level minus 12 dB."""
    n = int(seconds * RATE)
    out = fan_noise(seconds, rng) * (speech_level / (10 ** (10 / 20)))
    t = 0.5
    click_level = speech_level * 0.7
    while t < seconds:
        start = int(t * RATE)
        length = int(0.004 * RATE)
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.0008 * RATE))
        out[start : start + length] += (burst / (np.max(np.abs(burst)) + 1e-12) * click_level).astype(np.float32)
        t += float(rng.uniform(0.3, 1.5))
    tt = np.arange(n) / RATE
    hum = sum(np.sin(2 * math.pi * 60.0 * k * tt) / k for k in range(1, 6))
    hum = hum / np.sqrt(np.mean(hum**2)) * (speech_level * 10 ** (-12 / 20))
    hum[: int(HUM_START_S * RATE)] = 0.0
    out = out + hum.astype(np.float32)
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def build_inputs(work_dir: Path, wanted: list[str], limit_seconds: float | None) -> dict[str, Path]:
    """Write the evaluation inputs as 16 kHz PCM16 WAV files (deterministic, seed 2026) and return their paths."""
    from voxscribe.speech.wav import write_wav16

    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    speech = read_fixture()
    level = active_rms(speech)
    paths: dict[str, Path] = {}
    for name in wanted:
        path = work_dir / f"{name}.wav"
        if name == "clean":
            signal = speech
        elif name == "fan20":
            signal = mix(speech, 20.0, rng)
        elif name == "fan10":
            signal = mix(speech, 10.0, rng)
        elif name == "noise":
            signal = noise_only(NOISE_SECONDS, level, rng)
        else:
            raise SystemExit(f"unknown input {name!r}; choose from {INPUTS}")
        if limit_seconds:
            signal = signal[: int(limit_seconds * RATE)]
        write_wav16(path, signal, RATE)
        paths[name] = path
        log.info("input %s: %.1f s at %s", name, len(signal) / RATE, path)
    return paths


# ---- running -----------------------------------------------------------------------------------------------------


def build_transcriber(mode: str, asr, session_id: str):
    from voxscribe.meeting.live import LiveTranscriber
    from voxscribe.speech import SEGMENTATION, create_detector

    common = dict(
        max_segment_ms=15000,  # the server's default live_max_segment_ms
        partial_update_interval_ms=10**9,  # partials off: measure the stable path
        skip_partials_when_behind=False,
    )
    if mode == "adaptive" or mode == "fixed":
        return LiveTranscriber(asr, session_id, vad_mode=mode, **common)
    return LiveTranscriber(
        asr, session_id, vad_mode="detector", detector=create_detector(mode), policy=SEGMENTATION, **common
    )


def run_one(mode: str, name: str, path: Path, asr, reference: str | None) -> dict:
    from voxscribe.speech.wav import read_wav

    samples, rate = read_wav(path)
    pcm = np.round(samples * 32767.0).astype("<i2").tobytes()
    step = RATE * FRAME_MS // 1000 * 2
    t = build_transcriber(mode, asr, f"eval-{mode}-{name}")
    stable = []
    started = time.monotonic()
    cpu0 = time.process_time()
    for i in range(0, len(pcm), step):
        stable += t.push_audio(0, pcm[i : i + step])
    stable += t.close()
    wall = time.monotonic() - started
    cpu = time.process_time() - cpu0
    texts = [s.text for s in stable]
    result = {
        "mode": mode,
        "input": name,
        "audio_seconds": round(len(samples) / rate, 3),
        "utterances": t.utterances,
        "speech_seconds": round(t.speech_ms / 1000.0, 3),
        "asr_calls": t.asr_calls,
        "asr_seconds": round(t.asr_seconds, 3),
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "segments": len(stable),
        "segment_texts": texts,
    }
    if reference is not None:
        result.update(score(reference, texts))
    else:
        result["noise_segments"] = len(stable)
        result["short_segments"] = sum(1 for s in texts if len(words_of(s)) < 4)
        result["noise_segments_long"] = len(stable) - result["short_segments"]
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--inputs", default=",".join(INPUTS))
    p.add_argument("--out", default=str(HERE / "results" / "eval-live-vad.json"))
    p.add_argument("--work-dir", default=str(HERE / "results" / "eval-live-vad-inputs"))
    p.add_argument("--asr-model", default="small.en")
    p.add_argument("--limit-seconds", type=float, default=None, help="shorten every input (smoke runs)")
    p.add_argument("--markdown", action="store_true", help="print the table for an existing report and exit")
    p.add_argument("--rescore", action="store_true", help="recompute the scores of an existing report from its saved segment texts and exit")
    return p.parse_args(argv)


def markdown_table(report: dict) -> str:
    runs = {(r["mode"], r["input"]): r for r in report["runs"]}
    modes = [m for m in MODES if any(k[0] == m for k in runs)]
    lines = [
        "| Input | Mode | Word recall % | 4-gram recall % | Noise segments (4+ words) | Short segments (< 4 words) "
        "| Segments | Utterances | Speech s | ASR s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in INPUTS:
        for mode in modes:
            r = runs.get((mode, name))
            if r is None:
                continue
            wr = "" if r.get("word_recall") is None else f"{100 * r['word_recall']:.1f}"
            gr = "" if r.get("fourgram_recall") is None else f"{100 * r['fourgram_recall']:.1f}"
            long_noise = r.get("noise_segments_long", r["noise_segments"])
            short = r.get("short_segments", "")
            lines.append(
                f"| {name} | {mode} | {wr} | {gr} | {long_noise} | {short} | {r['segments']} | {r['utterances']} "
                f"| {r['speech_seconds']:.1f} | {r['asr_seconds']:.1f} |"
            )
    return "\n".join(lines)


def rescore(report: dict) -> dict:
    """Recompute every speech run's scores from its saved segment texts (the noise clip has no reference)."""
    reference = REFERENCE.read_text()
    for r in report["runs"]:
        if r["input"] != "noise":
            r.update(score(reference, r["segment_texts"]))
        else:
            r["noise_segments"] = r["segments"]
            r["short_segments"] = sum(1 for s in r["segment_texts"] if len(words_of(s)) < 4)
            r["noise_segments_long"] = r["segments"] - r["short_segments"]
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    logging.getLogger("voxscribe").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # one line per decode otherwise
    args = parse_args(argv)
    out = Path(args.out)
    if args.rescore:
        report = rescore(json.loads(out.read_text()))
        out.write_text(json.dumps(report, indent=2))
        print(markdown_table(report))
        return 0
    if args.markdown:
        print(markdown_table(json.loads(out.read_text())))
        return 0

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    inputs = [i.strip() for i in args.inputs.split(",") if i.strip()]
    from voxscribe.speech import available_detectors

    for mode in modes:
        if mode not in ("adaptive", "fixed") and mode not in available_detectors():
            raise SystemExit(f"detector {mode!r} is not available here (have {available_detectors()})")

    from app.local.providers import LocalASRProvider

    asr = LocalASRProvider(model_name=args.asr_model)
    asr._load_provider()
    reference = REFERENCE.read_text()
    manifest = json.loads(MANIFEST.read_text())
    paths = build_inputs(Path(args.work_dir), inputs, args.limit_seconds)

    report = {
        "fixture": {
            "file": str(FIXTURE),
            "sha256": manifest.get("sha256"),
            "duration_s": manifest.get("duration_s"),
            "total_word_count": manifest.get("total_word_count"),
            "voice": manifest.get("voice"),
        },
        "asr": {"provider": asr.name, "device": asr.device, "compute_type": asr.compute_type, "vad_filter": True},
        "frame_ms": FRAME_MS,
        "partials": "off",
        "max_segment_ms": 15000,
        "policy": "segmentation",
        "limit_seconds": args.limit_seconds,
        "inputs": {name: {"file": str(p)} for name, p in paths.items()},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in inputs:
        for mode in modes:
            log.info("run %s on %s", mode, name)
            result = run_one(mode, name, paths[name], asr, None if name == "noise" else reference)
            report["runs"].append(result)
            log.info(
                "done %s/%s: segments=%s noise=%s recall4=%s asr=%.1fs wall=%.1fs",
                mode, name, result["segments"], result["noise_segments"], result.get("fourgram_recall"),
                result["asr_seconds"], result["wall_seconds"],
            )
            out.write_text(json.dumps(report, indent=2))
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.write_text(json.dumps(report, indent=2))
    print(markdown_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
