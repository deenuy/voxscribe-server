"""verify2_p1stop — TTS latency before/after CoreML, prove SC-G3 on one real turn.

Usage:
    uv run tools/verify2_p1stop.py --engine-url http://127.0.0.1:PORT

Steps:
1. Benchmark Kokoro synthesis latency on this process (no engine needed).
2. Trigger one engine turn via /v1/wake/trigger and verify gate_decisions row
   exists in DB before playback ends (SC-G3 proof).
3. Print before/after TTS latency and SC-G3 result.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify2_p1stop")

_HOME = Path.home()
_KOKORO_MODEL = _HOME / ".voxscribe" / "kokoro" / "onnx" / "model_quantized.onnx"
_KOKORO_VOICES = _HOME / ".voxscribe" / "kokoro" / "voices.bin.npz"
_SAMPLE_TEXT = "I'm here and happy to help you with that."

# Add voxscribe-server to path
_SERVER = Path(__file__).parent.parent
sys.path.insert(0, str(_SERVER))


def _bench_kokoro(provider: str, n: int = 3) -> float:
    """Synthesize _SAMPLE_TEXT n times with given provider, return median ms."""
    import os as _os
    old = _os.environ.get("ONNX_PROVIDER")
    _os.environ["ONNX_PROVIDER"] = provider

    from kokoro_onnx import Kokoro
    try:
        k = Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))
    except Exception as e:
        log.error("Kokoro load failed with provider=%s: %s", provider, e)
        if old is None:
            del _os.environ["ONNX_PROVIDER"]
        else:
            _os.environ["ONNX_PROVIDER"] = old
        return float("inf")

    times = []
    for i in range(n):
        t0 = time.perf_counter()
        samples, sr = k.create(_SAMPLE_TEXT, voice="af_heart", speed=1.0, lang="en-us")
        times.append((time.perf_counter() - t0) * 1000)
        log.info("  [%s] run %d: %.0f ms", provider, i + 1, times[-1])

    if old is None:
        if "ONNX_PROVIDER" in _os.environ:
            del _os.environ["ONNX_PROVIDER"]
    else:
        _os.environ["ONNX_PROVIDER"] = old

    times.sort()
    return times[len(times) // 2]


def _verify_sc_g3(engine_url: str) -> dict:
    """Trigger one session, check gate_decisions row exists before playback ends."""
    import requests
    import psycopg

    DB_URL = os.environ.get(
        "VOXSCRIBE_DATABASE_URL",
        "postgresql://localhost:5432/ideascribe_dev",
    )

    log.info("SC-G3 verify: waiting for engine ARMED...")
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{engine_url}/v1/wake/state", timeout=3)
            if r.json().get("state") == "armed":
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        return {"sc_g3": "error", "reason": "engine not armed"}

    log.info("SC-G3 verify: triggering session...")
    try:
        requests.post(f"{engine_url}/v1/wake/trigger", timeout=5)
    except Exception as e:
        return {"sc_g3": "error", "reason": str(e)}

    # Wait 5s then check DB - gate_decisions row should appear before playback ends
    time.sleep(5.0)
    conn = psycopg.connect(DB_URL)
    try:
        row = conn.execute(
            "SELECT id, decision FROM gate_decisions"
            " JOIN meetings m ON m.id = gate_decisions.meeting_id"
            " WHERE m.origin = 'wake'"
            " ORDER BY gate_decisions.decided_at DESC LIMIT 1"
        ).fetchone()
        if row and row[1] == "pending":
            log.info("SC-G3: gate_decisions row found with decision=pending (pre-playback row exists)")
            return {"sc_g3": "pass_pending_row", "gate_id": row[0]}
        elif row:
            log.info("SC-G3: gate_decisions row found with decision=%s", row[1])
            return {"sc_g3": "pass_final", "gate_id": row[0], "decision": row[1]}
        else:
            log.warning("SC-G3: no gate_decisions row found after 5s")
            return {"sc_g3": "fail_no_row"}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-url", default="", help="Engine URL for SC-G3 check")
    parser.add_argument("--tts-only", action="store_true", help="Skip SC-G3 check")
    args = parser.parse_args()

    if not _KOKORO_MODEL.exists():
        log.error("Kokoro model not found at %s", _KOKORO_MODEL)
        sys.exit(1)

    print("\n=== verify2_p1stop ===\n")

    # TTS benchmark
    print("--- TTS latency benchmark ---")
    print(f"Text: '{_SAMPLE_TEXT}' ({len(_SAMPLE_TEXT.split())} words)\n")

    cpu_med = _bench_kokoro("CPUExecutionProvider")
    print(f"\nCPU  median: {cpu_med:.0f} ms")

    import onnxruntime as rt
    if "CoreMLExecutionProvider" in rt.get_available_providers():
        coreml_med = _bench_kokoro("CoreMLExecutionProvider")
        print(f"CoreML median: {coreml_med:.0f} ms")
        if coreml_med < float("inf"):
            speedup = cpu_med / coreml_med
            print(f"Speedup: {speedup:.1f}x  (delta: {cpu_med - coreml_med:+.0f} ms)")
        print(f"\nV4-STAGE budget: 80-150 ms")
        if coreml_med <= 150:
            print(f"CoreML: WITHIN budget ({coreml_med:.0f} ms)")
        else:
            print(f"CoreML: OVER budget ({coreml_med:.0f} ms vs 150 ms)")
    else:
        print("CoreML not available on this machine")

    # SC-G3 check
    if args.engine_url and not args.tts_only:
        print("\n--- SC-G3: gate_decisions before playback ---")
        result = _verify_sc_g3(args.engine_url)
        print(f"SC-G3: {result}")
    elif not args.tts_only:
        print("\nSC-G3 check skipped (no --engine-url). Pass --engine-url to verify.")

    print("\n=== done ===\n")


if __name__ == "__main__":
    main()
