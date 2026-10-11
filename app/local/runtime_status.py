"""What the local AI runtime (Ollama) is actually doing, for the status chip.

Source of truth is the runtime itself: `/api/ps` lists the models in memory right now (with the size they use) and
`/api/tags` lists the models installed. Nothing here is a hard-coded model name: the roles come from the engine's
configuration and the tags shown are the runtime's own spelling.

A model that is installed but not in memory is NOT a problem. Ollama loads a model on first use and unloads it after
it has been idle, so an engine that is perfectly ready often has nothing in memory. State therefore means:
  offline    the runtime does not answer (each model that depends on it is an error)
  error      it answers, but a model is missing or unusable, or the database is not ok
  loading    the runtime is being started right now (the person asked for it, and it is not up yet)
  on-demand  everything is installed and reachable and at least one model is idle: ready for use
  ready      everything is installed and in memory
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


# Per-model states: a controlled vocabulary the app renders as Ready, On demand, Not downloaded, Error.
#   ready           in memory now
#   on-demand       installed, loads when it is first needed (the normal idle state, not a fault)
#   not-downloaded  required but not installed
#   error           cannot be used (its runtime is unreachable, or speech recognition is unavailable)
#   loading         its runtime is being started right now
READY, ON_DEMAND, NOT_DOWNLOADED, ERROR, LOADING = "ready", "on-demand", "not-downloaded", "error", "loading"

# The overall state is the worst of the parts, in this order. A missing required model counts as an error.
_SEVERITY = {"offline": 5, ERROR: 4, NOT_DOWNLOADED: 4, LOADING: 3, ON_DEMAND: 2, READY: 1}


def worst(states: list[str]) -> str:
    """offline > error > loading > on-demand > ready (a model that is not downloaded counts as error)."""
    top = max(states, key=lambda s: _SEVERITY.get(s, 0), default=READY)
    return ERROR if top == NOT_DOWNLOADED else top


def build_status(*, settings, probe: dict | None, db_ok: bool, asr_ok: bool, start_requested_at: float | None, now: float | None = None) -> dict:
    """The hot-path status: an overall state and one row per model role, each with its own state.

    `probe` is None when the runtime did not answer. Nothing about versions, uptime, the runtime's name or the
    database is in here (see `build_details`): this document is polled every few seconds and stays small."""
    now = time.time() if now is None else now
    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if settings.providers == "fake":
        models = [{"role": r, "tag": "fake:deterministic", "state": READY, "memoryMB": None}
                  for r in ("main", "live-assist", "transcription", "embedding")]
        overall = worst([m["state"] for m in models])
        return {"state": overall if db_ok else ERROR, "models": models, "updatedAt": stamp}

    wanted = [("main", settings.llm_model), ("live-assist", settings.live_model), ("embedding", settings.embed_model)]
    models = []
    runtime_down = probe is None
    starting = runtime_down and start_requested_at is not None and now - start_requested_at < START_GRACE_S
    loaded = {} if runtime_down else {m.get("name") or m.get("model"): m for m in probe["loaded"]}
    installed = [] if runtime_down else [m.get("name") for m in probe["installed"]]
    for role, tag in wanted:
        if runtime_down:
            models.append({"role": role, "tag": tag, "state": LOADING if starting else ERROR, "memoryMB": None})
            continue
        spelled = _match(tag, installed)
        in_memory = _match(tag, list(loaded))
        entry = loaded.get(in_memory) if in_memory else None
        state = NOT_DOWNLOADED if spelled is None else READY if entry is not None else ON_DEMAND
        models.append({"role": role, "tag": in_memory or spelled or tag,  # exactly what `ollama ps` / `ollama list` print
                       "state": state, "memoryMB": _mb(entry) if entry else None})
    models.insert(2, _asr(settings, asr_ok))
    overall = worst([m["state"] for m in models])
    if runtime_down and not starting:
        overall = "offline"  # the AI runtime does not answer; every model that depends on it shows an error
    elif not db_ok:
        overall = ERROR
    return {"state": overall, "models": models, "updatedAt": stamp}


def build_details(*, settings, runtime_reachable: bool, db: dict, version: str, started_at: float | None, now: float | None = None) -> dict:
    """Technical facts for the Settings page, kept apart from the hot status path.

    No environment name leaves the engine: the database is reported as connected or not, with its schema version,
    never by name (a name such as a development database would otherwise reach a screen)."""
    now = time.time() if now is None else now
    return {
        "engine": {
            "version": version,
            "uptimeSeconds": int(now - started_at) if started_at else None,
            "startedAt": datetime.fromtimestamp(started_at, UTC).isoformat().replace("+00:00", "Z") if started_at else None,
        },
        "runtime": {"name": "ollama" if settings.providers != "fake" else "fake", "reachable": runtime_reachable},
        "database": {"ok": bool(db.get("ok")), "schemaVersion": db.get("schemaVersion")},
        "providers": settings.providers,
        "offline": settings.privacy_mode == "offline",
        "updatedAt": datetime.fromtimestamp(now, UTC).isoformat().replace("+00:00", "Z"),
    }


def _asr(settings, asr_ok: bool) -> dict:
    # Speech recognition is not an Ollama model: it is in memory while the engine runs, and it does not depend on the runtime.
    return {"role": "transcription", "tag": f"faster-whisper:{settings.asr_model}", "state": READY if asr_ok else ERROR, "memoryMB": None}


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
