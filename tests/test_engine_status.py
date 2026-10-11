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


def test_idle_models_are_on_demand_and_the_whole_is_on_demand_not_a_fault():
    s = status(probe())
    roles = by_role(s)
    assert [roles[r]["state"] for r in ("main", "live-assist", "embedding")] == ["on-demand"] * 3
    assert roles["transcription"]["state"] == "ready"  # speech recognition is in memory while the engine runs
    assert s["state"] == "on-demand"  # the worst child; the app shows this as "Ready" because it is usable


def test_models_in_memory_are_ready_with_their_size_and_the_runtimes_own_tag():
    s = status(probe(loaded=[{"name": "gemma4:12b", "size": 8 * 1024 * 1024 * 1024}, {"name": "nomic-embed-text:latest", "size": 300 * 1024 * 1024}]))
    roles = by_role(s)
    assert roles["main"] == {"role": "main", "tag": "gemma4:12b", "state": "ready", "memoryMB": 8192}
    assert roles["embedding"]["tag"] == "nomic-embed-text:latest" and roles["embedding"]["state"] == "ready"
    assert roles["live-assist"]["state"] == "on-demand"


def test_everything_in_memory_is_ready():
    s = status(probe(loaded=[{"name": "gemma4:12b", "size": 1}, {"name": "gemma4:e4b", "size": 1}, {"name": "nomic-embed-text:latest", "size": 1}]))
    assert s["state"] == "ready"


def test_tags_come_from_configuration_not_from_the_code():
    custom = dataclasses.replace(LOCAL, llm_model="qwen3:8b", live_model="qwen3:1.7b", embed_model="bge-m3", asr_model="medium.en")
    s = status(probe(have=("qwen3:8b", "qwen3:1.7b", "bge-m3:latest")), settings=custom)
    assert [m["tag"] for m in s["models"]] == ["qwen3:8b", "qwen3:1.7b", "faster-whisper:medium.en", "bge-m3:latest"]


def test_a_missing_model_is_not_downloaded_and_the_whole_is_an_error():
    s = status(probe(have=("gemma4:12b", "nomic-embed-text:latest")))
    assert by_role(s)["live-assist"]["state"] == "not-downloaded" and s["state"] == "error"


def test_speech_recognition_or_database_trouble_is_an_error():
    s = status(probe(), asr_ok=False)
    assert by_role(s)["transcription"]["state"] == "error" and s["state"] == "error"
    assert status(probe(), db_ok=False)["state"] == "error"


def test_when_the_runtime_stops_the_whole_is_offline_and_every_model_that_needs_it_is_an_error():
    s = status(None)
    assert s["state"] == "offline"
    roles = by_role(s)
    assert [roles[r]["state"] for r in ("main", "live-assist", "embedding")] == ["error"] * 3
    assert roles["transcription"]["state"] == "ready"  # not affected: it does not use the runtime


def test_a_recent_start_request_is_loading_and_an_old_one_that_never_worked_is_offline():
    s = status(None, start_requested_at=990.0, now=1000.0)
    assert s["state"] == "loading" and by_role(s)["main"]["state"] == "loading"
    assert status(None, start_requested_at=100.0, now=1000.0)["state"] == "offline"


def test_the_overall_state_is_the_worst_child_in_a_fixed_order():
    assert rt.worst(["ready", "on-demand"]) == "on-demand"
    assert rt.worst(["on-demand", "loading", "ready"]) == "loading"
    assert rt.worst(["loading", "error", "on-demand"]) == "error"
    assert rt.worst(["error", "offline"]) == "offline"
    assert rt.worst(["ready", "not-downloaded"]) == "error"  # a missing required model is an error
    assert rt.worst([]) == "ready"


def test_the_hot_status_carries_no_infrastructure():
    s = status(probe())
    assert set(s) == {"state", "models", "updatedAt"}
    assert all(set(m) == {"role", "tag", "state", "memoryMB"} for m in s["models"])


def test_fake_providers_report_ready_without_a_runtime():
    s = rt.build_status(settings=Config(providers="fake"), probe=None, db_ok=True, asr_ok=True, start_requested_at=None)
    assert s["state"] == "ready" and len(s["models"]) == 4


@pytest.mark.asyncio
async def test_the_endpoint_reports_per_model_state_and_the_updated_time(client, monkeypatch):
    app = client._transport.app
    app.state.settings = dataclasses.replace(app.state.settings, providers="local")

    async def fake_probe(host):
        return probe(loaded=[{"name": "gemma4:12b", "size": 1024 * 1024 * 1024}])

    monkeypatch.setattr(rt, "probe_runtime", fake_probe)
    body = (await client.get("/v1/engine/status")).json()
    assert set(body) == {"state", "models", "updatedAt"} and body["updatedAt"].endswith("Z")
    assert {m["role"]: m["state"] for m in body["models"]}["main"] == "ready"


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
    assert second.status_code == 202 and second.json()["state"] == "loading" and len(spawned) == 1  # no second process, and a real answer

    async def up(host):
        return probe()

    monkeypatch.setattr(rt, "probe_runtime", up)
    already = await client.post("/v1/engine/start")
    assert already.status_code == 200 and already.json()["state"] == "on-demand" and len(spawned) == 1


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


@pytest.mark.asyncio
async def test_details_carry_version_uptime_runtime_and_database_but_never_an_environment_name(client):
    from app.local.version import engine_version

    body = (await client.get("/v1/engine/details")).json()
    assert body["engine"]["version"] == engine_version() and body["engine"]["version"] != "0.1.0"
    assert body["engine"]["uptimeSeconds"] >= 0 and body["engine"]["startedAt"].endswith("Z")
    assert body["database"] == {"ok": True, "schemaVersion": body["database"]["schemaVersion"]} and body["database"]["schemaVersion"] >= 5
    assert body["runtime"]["reachable"] is True and body["providers"] == "fake" and body["offline"] is True
    # The test database is named like a development one: its name must not appear anywhere in the answer.
    name = (await client._transport.app.state.store.health_check())[2]
    assert name and name not in (await client.get("/v1/engine/details")).text
    assert name not in (await client.get("/v1/engine/status")).text
    assert name not in (await client.get("/v1/health")).text  # the health document names the kind of database, never the database
    health = (await client.get("/v1/health")).json()
    assert health["version"] == body["engine"]["version"]


@pytest.mark.asyncio
async def test_details_report_a_database_that_is_down_instead_of_failing(client, monkeypatch):
    async def broken():
        raise RuntimeError("down")

    monkeypatch.setattr(client._transport.app.state.store, "health_check", broken)
    body = (await client.get("/v1/engine/details")).json()
    assert body["database"] == {"ok": False, "schemaVersion": None}
    assert (await client.get("/v1/engine/status")).json()["state"] == "error"


# ------------------------------------------------------------------ the models list reflects reality


def test_a_tag_without_a_version_matches_latest():
    from app.local.model_service import tag_installed

    assert tag_installed("nomic-embed-text", ["nomic-embed-text:latest", "gemma4:12b"]) is True
    assert tag_installed("gemma4:12b", ["gemma4:12b"]) is True
    assert tag_installed("gemma4:e2b", ["gemma4:e4b"]) is False  # a different version is a different model
    assert tag_installed("bge-m3", ["bge-m3-large:latest"]) is False


def test_roles_default_to_what_the_engine_is_configured_to_use_and_choices_win():
    from app.local import model_service as ms

    ms._selected_roles.clear()
    cfg = Config(providers="local", llm_model="gemma4:12b", live_model="gemma4:e4b")
    assert ms.roles_for("gemma4:12b", cfg) == ["main"]
    assert ms.roles_for("gemma4:e4b", cfg) == ["live"]
    assert ms.roles_for("gemma4:e2b", cfg) == []
    ms.set_selected_roles("gemma4:e2b", ["live"])  # the person picked another live model
    assert ms.roles_for("gemma4:e2b", cfg) == ["live"] and ms.roles_for("gemma4:e4b", cfg) == []
    assert ms.roles_for("gemma4:12b", cfg) == ["main"]
    ms._selected_roles.clear()


@pytest.mark.asyncio
async def test_the_models_endpoint_marks_the_models_in_use(client):
    from app.local import model_service as ms

    ms._selected_roles.clear()
    items = {m["id"]: m for m in (await client.get("/v1/models")).json()["models"]}
    assert items["gemma4:12b"]["selectedRoles"] == ["main"] and items["gemma4:12b"]["selected"] is True
    assert items["gemma4:e4b"]["selectedRoles"] == ["live"]


def test_choosing_a_model_for_a_role_takes_the_role_from_the_old_one():
    from app.local import model_service as ms

    ms._selected_roles.clear()
    cfg = Config(providers="local")
    ms.assign_role("main", "gemma4:12b")
    ms.assign_role("live", "gemma4:e4b")
    ms.assign_role("main", "gemma4:e2b")
    assert ms.roles_for("gemma4:e2b", cfg) == ["main"] and ms.roles_for("gemma4:12b", cfg) == []
    ms.assign_role("live", "gemma4:e2b")  # one model may hold both roles
    assert ms.roles_for("gemma4:e2b", cfg) == ["main", "live"] and ms.roles_for("gemma4:e4b", cfg) == []
    ms._selected_roles.clear()


@pytest.mark.asyncio
async def test_roles_chosen_in_an_earlier_run_come_back_after_a_restart(client):
    from app.local import model_service as ms

    pool = client._transport.app.state.store.pool
    async with pool.connection() as conn:
        await conn.execute("INSERT INTO settings (key, value) VALUES ('model_role_main', '\"gemma4:e2b\"') ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value")
        await conn.commit()
    ms._selected_roles.clear()
    assert await ms.restore_selected_roles(pool) >= 1
    assert ms.roles_for("gemma4:e2b", Config(providers="local")) == ["main"]
    ms._selected_roles.clear()
