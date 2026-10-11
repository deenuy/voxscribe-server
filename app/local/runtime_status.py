"""What the local AI runtime (Ollama) is actually doing, for the status chip.

Source of truth is the runtime itself: `/api/ps` lists the models in memory right now (with the size they use) and
`/api/tags` lists the models installed. Nothing here is a hard-coded model name: the roles come from the engine's
configuration and the tags shown are the runtime's own spelling.

A model that is installed but not in memory is NOT a problem. Ollama loads a model on first use and unloads it after
it has been idle, so an engine that is perfectly ready often has nothing in memory. State therefore means:
  offline   the runtime does not answer
  loading   the runtime is being started right now (the person asked for it, and it is not up yet)
  degraded  it answers, but a required model is not installed, or the database or speech recognition is not ok
  ready     everything needed is installed and reachable (models may still be idle)
"""

import asyncio
import shutil
import time
from datetime import UTC, datetime

import httpx

PROBE_TIMEOUT_S = 1.5
START_GRACE_S = 60  # how long after "start" the chip shows loading before giving up and saying offline
START_COOLDOWN_S = 20  # a second "start" within this window does not spawn another process


async def probe_runtime(host: str) -> dict | None:
    """Ask the runtime what is installed and what is in memory. None when it does not answer."""
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
            ps, tags = await asyncio.gather(client.get(f"{host}/api/ps"), client.get(f"{host}/api/tags"))
        ps.raise_for_status()
        tags.raise_for_status()
        return {"loaded": ps.json().get("models", []) or [], "installed": tags.json().get("models", []) or []}
    except Exception:
        return None


def _match(tag: str, names: list[str]) -> str | None:
    """The runtime's own spelling of a configured tag: exact, or with the implicit :latest."""
    if tag in names:
        return tag
    if ":" not in tag and f"{tag}:latest" in names:
        return f"{tag}:latest"
    return None


def _mb(entry: dict) -> int | None:
    size = entry.get("size") or entry.get("size_vram")
    return round(size / (1024 * 1024)) if size else None


def build_status(*, settings, probe: dict | None, db_ok: bool, asr_ok: bool, start_requested_at: float | None, now: float | None = None) -> dict:
    """The status document. `probe` is None when the runtime did not answer."""
    now = time.time() if now is None else now
    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if settings.providers == "fake":
        models = [{"role": r, "tag": "fake:deterministic", "provider": "fake", "installed": True, "loaded": True, "memoryMB": None}
                  for r in ("main", "live-assist", "transcription", "embedding")]
        return {"state": "ready", "runtime": {"name": "fake", "reachable": True}, "models": models, "updatedAt": stamp}

    wanted = [("main", settings.llm_model), ("live-assist", settings.live_model), ("embedding", settings.embed_model)]
    models = []
    degraded = not db_ok
    if probe is None:
        starting = start_requested_at is not None and now - start_requested_at < START_GRACE_S
        for role, tag in wanted:
            models.append({"role": role, "tag": tag, "provider": "ollama", "installed": False, "loaded": False, "memoryMB": None})
        models.insert(2, _asr(settings, asr_ok))
        return {"state": "loading" if starting else "offline", "runtime": {"name": "ollama", "reachable": False}, "models": models, "updatedAt": stamp}

    loaded = {m.get("name") or m.get("model"): m for m in probe["loaded"]}
    installed = [m.get("name") for m in probe["installed"]]
    for role, tag in wanted:
        spelled = _match(tag, installed)
        in_memory = _match(tag, list(loaded))
        entry = loaded.get(in_memory) if in_memory else None
        if spelled is None:
            degraded = True
        models.append({
            "role": role,
            "tag": in_memory or spelled or tag,  # exactly what `ollama ps` / `ollama list` print
            "provider": "ollama",
            "installed": spelled is not None,
            "loaded": entry is not None,
            "memoryMB": _mb(entry) if entry else None,
        })
    asr = _asr(settings, asr_ok)
    models.insert(2, asr)
    if not asr["installed"]:
        degraded = True
    return {"state": "degraded" if degraded else "ready", "runtime": {"name": "ollama", "reachable": True}, "models": models, "updatedAt": stamp}


def _asr(settings, asr_ok: bool) -> dict:
    # Speech recognition is not an Ollama model: it is in memory while the engine runs.
    return {"role": "transcription", "tag": f"faster-whisper:{settings.asr_model}", "provider": "faster-whisper",
            "installed": bool(asr_ok), "loaded": bool(asr_ok), "memoryMB": None}


def runtime_command(settings) -> list[str] | None:
    """The command that starts the runtime, or None when it is not installed on this machine."""
    cmd = list(getattr(settings, "runtime_start_command", None) or ["ollama", "serve"])
    return cmd if shutil.which(cmd[0]) else None


async def spawn_runtime(cmd: list[str]) -> None:
    """Start the runtime detached, so it outlives this request and the engine's own restarts."""
    await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
