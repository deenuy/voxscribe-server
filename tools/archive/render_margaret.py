"""Render Margaret session audio from truth.json speech spans.

Reads each session_XX/truth.json, renders speech using macOS `say` with
Grandma (English (UK)) voice, inserts silence gaps, assembles 16 kHz mono WAV.

Writes: scratch/seniorcare/session_XX/session_XX.wav
Updates: scratch/seniorcare/manifest.json with sha256 per rendered file.

Usage: python tools/render_margaret.py [--sessions 1-10] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRATCH = Path("/Users/deenuy/Documents/Workspace/Projects/voxscribe/scratch/seniorcare")
VOICE = "Grandma (English (UK))"
RATE = 130          # wpm passed to say; Grandma voice is already slow
SAMPLE_RATE = 16000

log = logging.getLogger("render_margaret")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def gen_silence(duration_s: float, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono:d={duration_s:.4f}",
         "-ar", str(SAMPLE_RATE), "-ac", "1", out],
        capture_output=True, check=True,
    )


def speak_to_wav(text: str, out: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tf:
        aiff = tf.name
    try:
        subprocess.run(
            ["say", "-v", VOICE, "-r", str(RATE), "-o", aiff, "--", text],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff,
             "-ar", str(SAMPLE_RATE), "-ac", "1", out],
            capture_output=True, check=True,
        )
    finally:
        if os.path.exists(aiff):
            os.unlink(aiff)


def render_session(session_id: str, force: bool = False) -> str | None:
    sid = session_id.zfill(2)
    sess_dir = SCRATCH / f"session_{sid}"
    truth_path = sess_dir / "truth.json"
    out_path = sess_dir / f"session_{sid}.wav"

    if out_path.exists() and not force:
        log.info("session_%s: already rendered, skipping (use --force to re-render)", sid)
        with open(out_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    if not truth_path.exists():
        log.warning("session_%s: truth.json not found at %s, skipping", sid, truth_path)
        return None

    with open(truth_path) as f:
        truth = json.load(f)

    spans = truth.get("speech_spans", [])
    total_dur = truth.get("duration_s", 300.0)

    if not spans:
        log.warning("session_%s: no speech_spans in truth.json, skipping", sid)
        return None

    log.info("session_%s: rendering %d speech spans (%.0f s total)", sid, len(spans), total_dur)

    with tempfile.TemporaryDirectory() as tmpdir:
        parts: list[str] = []
        prev_end = 0.0

        for i, span in enumerate(spans):
            start_s = span["start_s"]
            end_s   = span["end_s"]
            text    = span["text"].strip()

            # Silence gap before this span
            gap = start_s - prev_end
            if gap > 0.05:
                sil_path = os.path.join(tmpdir, f"sil_{i:04d}.wav")
                gen_silence(gap, sil_path)
                parts.append(sil_path)

            # Speech segment
            sp_path = os.path.join(tmpdir, f"sp_{i:04d}.wav")
            speak_to_wav(text, sp_path)
            parts.append(sp_path)
            prev_end = end_s

        # Trailing silence
        tail = total_dur - prev_end
        if tail > 0.5:
            sil_path = os.path.join(tmpdir, "sil_tail.wav")
            gen_silence(tail, sil_path)
            parts.append(sil_path)

        # Concat list
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as cf:
            for p in parts:
                cf.write(f"file '{p}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file,
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             str(out_path)],
            capture_output=True, check=True,
        )

    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info("session_%s: rendered %.1f MB sha256=%s", sid, size_mb, sha[:12])
    return sha


def update_manifest(results: dict[str, str | None]) -> None:
    manifest_path = SCRATCH / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"schema_version": "v1", "sessions": []}

    rendered: dict[str, str] = {k: v for k, v in results.items() if v}
    for sess in manifest.get("sessions", []):
        sid = sess.get("session_id", "")
        if sid in rendered:
            sess["wav_sha256"] = rendered[sid]
            sess["wav_filename"] = f"session_{sid.zfill(2)}.wav"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest updated at %s", manifest_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", default="1-10",
                   help="Session range to render, e.g. 1-10 or 1,3,5")
    p.add_argument("--force", action="store_true",
                   help="Re-render even if wav already exists")
    args = p.parse_args()

    # Parse session list
    session_ids: list[str] = []
    for part in args.sessions.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            session_ids.extend(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            session_ids.append(part.strip())

    results: dict[str, str | None] = {}
    for sid in session_ids:
        try:
            sha = render_session(sid, force=args.force)
            results[sid.zfill(2)] = sha
        except Exception as exc:
            log.error("session_%s: FAILED %s", sid.zfill(2), exc)
            results[sid.zfill(2)] = None

    done = sum(1 for v in results.values() if v)
    log.info("Rendered %d/%d sessions", done, len(session_ids))
    update_manifest(results)


if __name__ == "__main__":
    main()
