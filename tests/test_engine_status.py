"""The engine status for the app's status chip: what the runtime reports decides the state, and nothing is hard-coded."""

import dataclasses

import pytest

from app.local import runtime_status as rt
from app.local.config import Config

LOCAL = Config(providers="local")


def installed(*names):
    return [{"name": n} for n in names]


def probe(loaded=(), have=("gemma4:12b", "gemma4:e4b", "nomic-embed-text:latest")):
    return {"loaded": list(loaded), "installed": installed(*have)}


def status(p, **kw):
    return rt.build_status(settings=kw.pop("settings", LOCAL), probe=p, db_ok=kw.pop("db_ok", True), asr_ok=kw.pop("asr_ok", True), start_requested_at=kw.pop("start_requested_at", None), now=kw.pop("now", 1000.0))


def by_role(s):
    return {m["role"]: m for m in s["models"]}


def test_ready_even_when_nothing_is_in_memory():
    s = status(probe())
    assert s["state"] == "ready"  # models load on first use and unload when idle; that is not a fault
    assert all(m["loaded"] is False for m in s["models"] if m["provider"] == "ollama")
    assert by_role(s)["main"]["installed"] is True


def test_models_in_memory_show_the_runtimes_own_tag_and_their_size():
    s = status(probe(loaded=[{"name": "gemma4:12b", "size": 8 * 1024 * 1024 * 1024}, {"name": "nomic-embed-text:latest", "size": 300 * 1024 * 1024}]))
    roles = by_role(s)
    assert roles["main"] == {"role": "main", "tag": "gemma4:12b", "provider": "ollama", "installed": True, "loaded": True, "memoryMB": 8192}
    assert roles["embedding"]["tag"] == "nomic-embed-text:latest"  # the configured "nomic-embed-text" as ollama prints it
    assert roles["live-assist"]["loaded"] is False


def test_tags_come_from_configuration_not_from_the_code():
    custom = dataclasses.replace(LOCAL, llm_model="qwen3:8b", live_model="qwen3:1.7b", embed_model="bge-m3", asr_model="medium.en")
    s = status(probe(have=("qwen3:8b", "qwen3:1.7b", "bge-m3:latest")), settings=custom)
    assert [m["tag"] for m in s["models"]] == ["qwen3:8b", "qwen3:1.7b", "faster-whisper:medium.en", "bge-m3:latest"]
    assert s["state"] == "ready"


def test_a_missing_model_is_degraded_and_says_which():
    s = status(probe(have=("gemma4:12b", "nomic-embed-text:latest")))
    assert s["state"] == "degraded" and by_role(s)["live-assist"]["installed"] is False


def test_database_or_speech_trouble_is_degraded():
    assert status(probe(), db_ok=False)["state"] == "degraded"
    s = status(probe(), asr_ok=False)
    assert s["state"] == "degraded" and by_role(s)["transcription"]["installed"] is False


def test_an_unreachable_runtime_is_offline_and_a_recent_start_request_is_loading():
    assert status(None)["state"] == "offline"
    assert status(None, start_requested_at=990.0, now=1000.0)["state"] == "loading"
    assert status(None, start_requested_at=100.0, now=1000.0)["state"] == "offline"  # it never came up: say so
    assert status(None)["runtime"]["reachable"] is False


def test_fake_providers_report_ready_without_a_runtime():
    s = rt.build_status(settings=Config(providers="fake"), probe=None, db_ok=True, asr_ok=True, start_requested_at=None)
    assert s["state"] == "ready" and len(s["models"]) == 4


@pytest.mark.asyncio
async def test_the_endpoint_reports_the_runtime_and_the_updated_time(client, monkeypatch):
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, providers="local")

    async def fake_probe(host):
        return probe(loaded=[{"name": "gemma4:12b", "size": 1024 * 1024 * 1024}])

    monkeypatch.setattr(rt, "probe_runtime", fake_probe)
    body = (await client.get("/v1/engine/status")).json()
    assert body["state"] in ("ready", "degraded") and body["updatedAt"].endswith("Z")
    assert {m["role"] for m in body["models"]} == {"main", "live-assist", "transcription", "embedding"}


@pytest.mark.asyncio
async def test_start_runtime_spawns_once_and_is_safe_to_repeat(client, monkeypatch):
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, providers="local")
    spawned = []

    async def down(host):
        return None

    async def spawn(cmd):
        spawned.append(cmd)

    monkeypatch.setattr(rt, "probe_runtime", down)
    monkeypatch.setattr(rt, "spawn_runtime", spawn)
    monkeypatch.setattr(rt, "runtime_command", lambda s: ["ollama", "serve"])
    first = await client.post("/v1/engine/start")
    second = await client.post("/v1/engine/start")
    assert first.status_code == 202 and first.json()["state"] == "loading"
    assert second.status_code == 202 and len(spawned) == 1  # the second request did not start another process

    async def up(host):
        return probe()

    monkeypatch.setattr(rt, "probe_runtime", up)
    already = await client.post("/v1/engine/start")
    assert already.status_code == 200 and already.json()["runtime"]["reachable"] is True and len(spawned) == 1


@pytest.mark.asyncio
async def test_start_runtime_says_so_when_it_is_not_installed(client, monkeypatch):
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, providers="local")

    async def down(host):
        return None

    monkeypatch.setattr(rt, "probe_runtime", down)
    monkeypatch.setattr(rt, "runtime_command", lambda s: None)
    res = await client.post("/v1/engine/start")
    assert res.status_code == 409 and res.json()["error"] == "runtime_not_installed"
