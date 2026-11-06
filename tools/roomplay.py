#!/usr/bin/env python3
"""Acoustic test harness for the wake loop.

Plays a scripted utterance through the system speaker; the running engine
picks it up via its microphone input. Optionally plays an interjection clip
(e.g. "stop") at a given offset after the reply starts.

Pause syntax in utterance text: [PAUSE:N] inserts N seconds of silence.
Disfluency syntax: include "um" or "uh" literally in the text.

Usage examples
--------------
  # P1 exit gate, no stop
  uv run tools/roomplay.py \\
    --utterance "hey jarvis [PAUSE:1] what should I remember about [PAUSE:3] Thursday's meeting with Susan" \\
    --label p1_run1 --report /tmp/p1_run1.json

  # P1 exit gate with stop at 1 s into reply
  uv run tools/roomplay.py \\
    --utterance "hey jarvis [PAUSE:1] what should I remember about [PAUSE:3] Thursday's meeting with Susan" \\
    --stop-at 1.0 \\
    --label p1_run2 --report /tmp/p1_run2.json

  # Multi-pause sweep (C3 check)
  uv run tools/roomplay.py \\
    --utterance "tell me [PAUSE:N] something interesting" \\
    --pause-sweep 1.5,3,5,8,11 \\
    --report /tmp/c3_sweep.json

Environment variables:
  VOXSCRIBE_DATABASE_URL   DB URL (default postgresql://localhost:5432/ideascribe_dev)
  ROOMPLAY_VOICE           macOS say voice for utterance generation (default Samantha)
  ROOMPLAY_POLL_S          DB poll interval in seconds (default 0.3)
  ROOMPLAY_TIMEOUT_S       Max seconds to wait for ledger row after utterance (default 90)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("roomplay")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    level=logging.INFO)

_DB_URL       = os.environ.get("VOXSCRIBE_DATABASE_URL",
                               "postgresql://localhost:5432/ideascribe_dev")
_ENGINE_URL   = os.environ.get("ROOMPLAY_ENGINE_URL", "http://127.0.0.1")
_ENGINE_PORT  = int(os.environ.get("ROOMPLAY_ENGINE_PORT", "0"))   # 0 = auto-detect from ready line
_VOICE        = os.environ.get("ROOMPLAY_VOICE", "Samantha")
_POLL_S       = float(os.environ.get("ROOMPLAY_POLL_S", "0.3"))
_TIMEOUT_S    = float(os.environ.get("ROOMPLAY_TIMEOUT_S", "90"))
_SAMPLE_RATE  = 24_000
_CHANNELS     = 1


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _say_to_wav(text: str, out_path: Path, voice: str = _VOICE,
                rate: int = _SAMPLE_RATE) -> None:
    """Render text to a WAV file using macOS say."""
    subprocess.run(
        ["say", "-v", voice,
         "--file-format", "WAVE",
         "--data-format", f"LEI16@{rate}",
         "-o", str(out_path), text],
        check=True, timeout=30,
    )


def _silence_wav(duration_s: float, out_path: Path,
                 rate: int = _SAMPLE_RATE) -> None:
    """Write a WAV file containing pure silence."""
    n_frames = int(duration_s * rate)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _concat_wavs(parts: list[Path], out_path: Path) -> None:
    """Concatenate WAV files (all must share rate/channels/sampwidth)."""
    frames_list = []
    params = None
    for p in parts:
        with wave.open(str(p), "rb") as wf:
            if params is None:
                params = wf.getparams()
            frames_list.append(wf.readframes(wf.getnframes()))
    with wave.open(str(out_path), "wb") as wf:
        wf.setparams(params)
        for frames in frames_list:
            wf.writeframes(frames)


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def build_utterance_wav(template: str, out_path: Path,
                        pause_n: float | None = None) -> float:
    """Build a WAV from template text with [PAUSE:N] markers.

    If pause_n is set, replaces 'N' in all [PAUSE:N] markers with that value.
    Returns duration in seconds.
    """
    if pause_n is not None:
        template = template.replace("[PAUSE:N]", f"[PAUSE:{pause_n}]")

    # Split on [PAUSE:...] markers
    import re
    parts_raw = re.split(r"(\[PAUSE:[0-9.]+\])", template)

    tmpdir = Path(tempfile.mkdtemp())
    part_wavs: list[Path] = []
    try:
        for i, chunk in enumerate(parts_raw):
            m = re.match(r"\[PAUSE:([0-9.]+)\]", chunk)
            if m:
                dur = float(m.group(1))
                p = tmpdir / f"part_{i}_silence.wav"
                _silence_wav(dur, p)
                part_wavs.append(p)
            else:
                text = chunk.strip()
                if text:
                    p = tmpdir / f"part_{i}.wav"
                    _say_to_wav(text, p)
                    part_wavs.append(p)

        if not part_wavs:
            raise ValueError("utterance produced no audio parts")

        _concat_wavs(part_wavs, out_path)
        return _wav_duration_s(out_path)
    finally:
        for p in part_wavs:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db_connect():
    import psycopg
    return psycopg.connect(_DB_URL)


def _latest_meeting_after(conn, since_z: str) -> dict | None:
    """Return the most recent meeting created after since_z, or None."""
    row = conn.execute(
        "SELECT id, title, status, created_at FROM meetings "
        "WHERE created_at > %s ORDER BY created_at DESC LIMIT 1",
        (since_z,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "title": row[1], "status": row[2], "created_at": str(row[3])}


def _ledger_row(conn, meeting_id: str) -> dict | None:
    row = conn.execute(
        "SELECT turn_end_ms, asr_final_ms, llm_request_sent_ms,"
        "       llm_first_token_ms, tts_first_audio_ms, "
        "       playback_end_ms, recorded_at "
        "FROM latency_ledger WHERE meeting_id = %s "
        "ORDER BY recorded_at DESC LIMIT 1",
        (meeting_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "turn_end_ms":          row[0],
        "asr_final_ms":         row[1],
        "llm_request_sent_ms":  row[2],
        "llm_first_token_ms":   row[3],
        "tts_first_audio_ms":   row[4],
        "playback_end_ms":      row[5],
        "recorded_at":          str(row[6]),
    }


def _gate_row(conn, meeting_id: str) -> dict | None:
    row = conn.execute(
        "SELECT decision, category, mode, decided_at "
        "FROM gate_decisions WHERE meeting_id = %s "
        "ORDER BY decided_at DESC LIMIT 1",
        (meeting_id,),
    ).fetchone()
    if row is None:
        return None
    return {"decision": row[0], "category": row[1],
            "mode": row[2], "decided_at": str(row[3])}


def _meeting_row(conn, meeting_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, title, status, origin, started_at, ended_at "
        "FROM meetings WHERE id = %s",
        (meeting_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "title": row[1], "status": row[2],
        "origin": row[3], "started_at": str(row[4]), "ended_at": str(row[5]),
    }


# ---------------------------------------------------------------------------
# Speaker playback (sounddevice, not afplay, so volume is independent of
# system volume setting and the mic gets a clean near-field signal)
# ---------------------------------------------------------------------------

def _read_wav_float32(path: Path):
    """Read WAV as float32 array and sample rate."""
    import numpy as np
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples_i16 = np.frombuffer(raw, dtype="<i2")
    return samples_i16.astype("float32") / 32768.0, rate


def _play_wav_blocking(path: Path) -> None:
    """Play WAV through default system output, blocking until done.

    Uses sounddevice.play() so amplitude is system-volume-independent
    (afplay respects the macOS volume knob, giving near-zero RMS at the mic
    when volume is low).
    """
    import sounddevice as sd
    samples, rate = _read_wav_float32(path)
    sd.play(samples, samplerate=rate, device="MacBook Pro Speakers")
    sd.wait()


def _play_wav_nonblocking(path: Path):
    """Start playing WAV; caller must call sounddevice.stop() when done."""
    import sounddevice as sd
    samples, rate = _read_wav_float32(path)
    sd.play(samples, samplerate=rate, device="MacBook Pro Speakers")


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def _trigger_session(engine_base_url: str) -> bool:
    """POST /v1/wake/trigger to start a session without OWW detection.

    Returns True if the session started, False on error.
    """
    import urllib.request, urllib.error, json as _json
    url = f"{engine_base_url}/v1/wake/trigger"
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = _json.loads(resp.read())
            log.info("trigger response: %s", body)
            return bool(body.get("triggered"))
    except urllib.error.HTTPError as exc:
        log.error("trigger failed: HTTP %d: %s", exc.code, exc.read()[:200])
        return False
    except Exception as exc:
        log.error("trigger failed: %s", exc)
        return False


def _wait_for_armed(engine_base_url: str, timeout_s: float = 60.0) -> bool:
    """Poll GET /v1/wake/state until state is not 'listening' (session closed).

    Returns True if state reached 'armed', False if timeout or error.
    """
    import urllib.request, urllib.error, json as _json
    url = f"{engine_base_url}/v1/wake/state"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = _json.loads(resp.read())
                state = body.get("state", "")
                if state != "listening":
                    log.info("engine state: %s (ready for next run)", state)
                    return True
        except Exception as exc:
            log.debug("state poll error: %s", exc)
        time.sleep(0.5)
    log.warning("_wait_for_armed: timed out after %.0fs", timeout_s)
    return False


def run_one(
    utterance_path: Path,
    utterance_dur_s: float,
    *,
    stop_at_s: float | None = None,
    label: str = "run",
    engine_base_url: str = "",
) -> dict:
    """Play one utterance through the speaker and collect the engine's response.

    Returns a result dict with timing data and pass/fail verdicts.

    The utterance WAV should contain only the content AFTER the wake phrase
    (e.g. the question/statement). The trigger endpoint starts the session,
    then this plays the audio into the mic so the engine captures it.
    """
    result: dict = {
        "label": label,
        "utterance_wav": str(utterance_path),
        "utterance_dur_s": utterance_dur_s,
        "stop_at_s": stop_at_s,
        "played_at": datetime.now(UTC).isoformat(),
        "mode": "trigger+roomplay [SIM:wake]",
        "meeting": None,
        "ledger": None,
        "gate": None,
        "verdicts": {},
        "error": None,
    }

    conn = None
    try:
        conn = _db_connect()
        since_z = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        # -- trigger session via API (bypasses OWW acoustic detection) ----
        if engine_base_url:
            log.info("[%s] triggering session via %s ...", label, engine_base_url)
            if not _trigger_session(engine_base_url):
                result["error"] = "trigger failed -- see logs"
                return result
            time.sleep(0.5)  # let session start and chime play
        else:
            log.warning("[%s] no engine_base_url -- relying on acoustic wake detection", label)

        # -- play utterance (content speech) into mic ----------------------
        log.info("[%s] playing utterance (%.1f s) ...", label, utterance_dur_s)
        play_start_mono = time.monotonic()
        _play_wav_blocking(utterance_path)
        utterance_end_mono = time.monotonic()
        log.info("[%s] utterance done, waiting for engine ...", label)

        # -- wait for engine to create a meeting ---------------------------
        meeting = None
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline:
            conn.autocommit = True
            meeting = _latest_meeting_after(conn, since_z)
            if meeting:
                break
            time.sleep(_POLL_S)

        if meeting is None:
            result["error"] = "timeout: no meeting created"
            log.warning("[%s] %s", label, result["error"])
            return result

        meeting_id = meeting["id"]
        log.info("[%s] meeting started: %s", label, meeting_id)

        # -- stop injection: play "stop" at stop_at_s after tts_first_audio ----
        if stop_at_s is not None:
            # Poll for tts_first_audio_ms, then play stop clip
            stop_proc: subprocess.Popen | None = None
            stop_clip_path = _make_stop_clip()
            while time.monotonic() < deadline:
                ledger = _ledger_row(conn, meeting_id)
                if ledger and ledger["tts_first_audio_ms"] is not None:
                    # wait stop_at_s more seconds from now
                    log.info("[%s] tts_first_audio detected, playing stop in %.1f s",
                             label, stop_at_s)
                    time.sleep(stop_at_s)
                    log.info("[%s] playing 'stop' interjection", label)
                    stop_play_mono = time.monotonic()
                    _play_wav_nonblocking(stop_clip_path)
                    import sounddevice as sd
                    sd.wait()
                    result["stop_played_at_mono"] = stop_play_mono
                    break
                time.sleep(_POLL_S)

        # -- wait for ledger row -------------------------------------------
        while time.monotonic() < deadline:
            ledger = _ledger_row(conn, meeting_id)
            if ledger:
                break
            time.sleep(_POLL_S)

        if ledger is None:
            result["error"] = "timeout: no ledger row"
            log.warning("[%s] %s", label, result["error"])
        else:
            result["ledger"] = ledger
            log.info("[%s] ledger: %s", label, json.dumps(ledger, default=str))

        gate = _gate_row(conn, meeting_id)
        result["gate"] = gate

        meeting_full = _meeting_row(conn, meeting_id)
        result["meeting"] = meeting_full

        # -- verdicts -------------------------------------------------------
        v = {}
        if ledger:
            te  = ledger["turn_end_ms"]
            lft = ledger["llm_first_token_ms"]
            tfa = ledger["tts_first_audio_ms"]
            pbe = ledger["playback_end_ms"]

            if tfa is not None and te is not None:
                first_audio_ms = tfa - te
                v["first_audio_ms"] = first_audio_ms
                v["latency_ok"] = first_audio_ms < 1500  # 1.5s gate
                log.info("[%s] turn_end->first_audio: %d ms (ok=%s)",
                         label, first_audio_ms, v["latency_ok"])
            else:
                v["first_audio_ms"] = None
                v["latency_ok"] = False

            if lft is not None and te is not None:
                v["llm_first_token_ms"] = lft - te

        if gate:
            v["gate_decision"] = gate["decision"]
            v["gate_category"] = gate["category"]

        # stop check: playback_end - stop_play should be < 300ms
        if stop_at_s is not None and "stop_played_at_mono" in result and ledger:
            pbe = ledger.get("playback_end_ms")
            if pbe is not None and ledger.get("tts_first_audio_ms") is not None:
                # playback_end_ms is monotonic_ms() at time of recording
                # We can't compare directly to system monotonic without calibration
                # Record as "stop injected; see playback_end_ms for server-side time"
                v["stop_injected"] = True
                v["stop_note"] = (
                    "stop played; check playback_end_ms vs tts_first_audio_ms "
                    "to verify <300ms halt after stop signal"
                )

        result["verdicts"] = v

    except Exception as exc:
        result["error"] = str(exc)
        log.exception("[%s] run failed", label)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    return result


def _make_stop_clip() -> Path:
    """Generate a 'stop' WAV clip to the temp dir if it doesn't exist."""
    p = Path(tempfile.gettempdir()) / "roomplay_stop.wav"
    if not p.exists():
        _say_to_wav("stop", p)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utterance", required=True,
                        help="Utterance text. Use [PAUSE:N] for N-second silences.")
    parser.add_argument("--stop-at", type=float, default=None, metavar="SECS",
                        help="Play 'stop' this many seconds after first TTS audio starts.")
    parser.add_argument("--pause-sweep", default=None, metavar="1.5,3,5,8,11",
                        help="Replace [PAUSE:N] with each value; runs one utterance per value.")
    parser.add_argument("--label", default="run", help="Label for this run (used in report key).")
    parser.add_argument("--report", default=None, metavar="FILE.json",
                        help="Write JSON report to this file. Default: print to stdout.")
    parser.add_argument("--keep-wav", action="store_true",
                        help="Keep generated WAV files after the run.")
    parser.add_argument("--engine-url", default="",
                        help="Base URL of running engine (e.g. http://127.0.0.1:55115). "
                             "When set, uses /v1/wake/trigger to start sessions instead "
                             "of relying on acoustic OWW detection. Wake phrase tagged [SIM].")
    args = parser.parse_args()

    engine_base_url = args.engine_url.rstrip("/")

    results: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        if args.pause_sweep:
            pause_values = [float(x) for x in args.pause_sweep.split(",")]
            for i, pv in enumerate(pause_values):
                label = f"{args.label}_p{pv}"
                wav_path = tmpdir_path / f"utterance_{i}.wav"
                log.info("building utterance for pause=%.1f s ...", pv)
                dur = build_utterance_wav(args.utterance, wav_path, pause_n=pv)
                if args.keep_wav:
                    import shutil
                    shutil.copy(wav_path, f"/tmp/{label}.wav")
                result = run_one(wav_path, dur, stop_at_s=args.stop_at,
                                 label=label, engine_base_url=engine_base_url)
                results.append(result)
                # wait for engine to return to armed state before next trigger
                if engine_base_url and i < len(pause_values) - 1:
                    _wait_for_armed(engine_base_url, timeout_s=90.0)
                else:
                    time.sleep(2.0)
        else:
            wav_path = tmpdir_path / "utterance.wav"
            log.info("building utterance ...")
            dur = build_utterance_wav(args.utterance, wav_path)
            if args.keep_wav:
                import shutil
                shutil.copy(wav_path, f"/tmp/{args.label}.wav")
            result = run_one(wav_path, dur, stop_at_s=args.stop_at,
                             label=args.label, engine_base_url=engine_base_url)
            results.append(result)

    # -- summary -----------------------------------------------------------
    print("\n=== ROOMPLAY QUICK STATS ===")
    for r in results:
        v = r.get("verdicts", {})
        err = r.get("error")
        fa = v.get("first_audio_ms")
        gate = v.get("gate_decision", "unknown")
        ok = v.get("latency_ok", False)
        if err:
            print(f"  {r['label']:30s}  ERROR: {err}")
        else:
            print(f"  {r['label']:30s}  first_audio={fa} ms  gate={gate}  latency_ok={ok}")
            if r.get("ledger"):
                L = r["ledger"]
                print(f"    ledger: turn_end={L['turn_end_ms']}  llm_first={L['llm_first_token_ms']}"
                      f"  tts_first={L['tts_first_audio_ms']}  pb_end={L['playback_end_ms']}")
            if r.get("gate"):
                G = r["gate"]
                print(f"    gate:   decision={G['decision']}  category={G['category']}  mode={G['mode']}")
            if r.get("meeting"):
                M = r["meeting"]
                print(f"    meeting: id={M['id'][:8]}...  status={M['status']}  origin={M['origin']}")
    print("=== END ===\n")

    if args.report:
        out = {"runs": results, "generated_at": datetime.now(UTC).isoformat()}
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2, default=str)
        log.info("report written to %s", args.report)
    else:
        print(json.dumps({"runs": results}, indent=2, default=str))

    any_error = any(r.get("error") for r in results)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
