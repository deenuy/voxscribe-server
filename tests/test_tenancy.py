"""Tenant isolation: every signed-in person is a tenant, and nothing of theirs is visible to, or changeable by,
another person. These tests run the real engine with identity switched on."""

import asyncio
import base64
import time
import uuid
from pathlib import Path

import httpx
import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.local import connector_tokens, events, meeting_bus, tenancy
from app.local.config import Config
from app.local.identity import IdentityError, IdentityVerifier
from app.local.job_runner import JobRunner
from app.local.migrations import run_migrations

DB_URL = "postgresql://localhost:5432/ideascribe_dev"
ISSUER, AUDIENCE = "http://issuer.test", "ideascribe"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class Keys:
    """A signing key and the public document an account service would publish for it."""

    def __init__(self, kid="k1"):
        self.private = Ed25519PrivateKey.generate()
        self.kid = kid
        raw = self.private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.jwk = {"kty": "OKP", "crv": "Ed25519", "x": _b64(raw), "kid": kid, "alg": "EdDSA", "use": "sig"}

    def token(self, sub, *, typ="access", aud=AUDIENCE, iss=ISSUER, exp_in=900, kid=None, extra=None):
        now = int(time.time())
        claims = {"iss": iss, "aud": aud, "sub": sub, "typ": typ, "iat": now, "exp": now + exp_in, "role": "user", **(extra or {})}
        return jwt.encode(claims, self.private, algorithm="EdDSA", headers={"kid": kid or self.kid})


@pytest.fixture
def keys():
    return Keys()


@pytest.fixture
async def tenant_engine(keys):
    """The engine with identity on. Yields (client, app, make_headers). Drops every tenant schema it created."""
    from app.local.app import create_app

    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    base = f"t_{uuid.uuid4().hex[:8]}"  # the shared (platform) schema, as the other tests use
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {base}")
        await conn.commit()
    await run_migrations(DB_URL, schema=base)
    settings = Config(database_url=DB_URL, data_dir=str(data_dir), providers="fake", identity_issuer=ISSUER, identity_audience=AUDIENCE, identity_jwks_url="http://jwks.test/keys")
    app = await create_app(settings=settings, auth_disabled=True, schema=base)

    async def fetch():
        return {"keys": [keys.jwk]}

    app.state.identity = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="http://jwks.test/keys", fetch=fetch)
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    created: list[str] = []

    def as_user(sub):
        created.append(tenancy.tenant_key(sub))
        return {"X-Identity-Token": keys.token(sub)}

    yield client, app, as_user
    await client.aclose()
    await ctx.__aexit__(None, None, None)
    async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
        for key in set(created):
            await conn.execute(f'DROP SCHEMA IF EXISTS "{key}" CASCADE')
        await conn.execute(f"DROP SCHEMA IF EXISTS {base} CASCADE")


# --------------------------------------------------------------------------- who may call

@pytest.mark.asyncio
async def test_a_request_without_a_verified_person_is_refused(tenant_engine, keys):
    client, _, _ = tenant_engine
    assert (await client.get("/v1/meetings")).json()["error"] == "identity_required"
    bad = {
        "garbage": "not.a.jwt",
        "expired": keys.token("u1", exp_in=-60),
        "wrong audience": keys.token("u1", aud="someone-else"),
        "wrong issuer": keys.token("u1", iss="http://evil.test"),
        "a refresh token": keys.token("u1", typ="refresh"),
        "an unknown key": keys.token("u1", kid="unknown"),
        "signed by another key": Keys("k1").token("u1"),  # same kid, different private key
    }
    for what, token in bad.items():
        res = await client.get("/v1/meetings", headers={"X-Identity-Token": token})
        assert res.status_code == 401 and res.json()["error"] == "identity_invalid", what


@pytest.mark.asyncio
async def test_an_algorithm_downgrade_is_refused(tenant_engine, keys):
    client, _, _ = tenant_engine
    forged = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": "u1", "typ": "access", "exp": int(time.time()) + 600}, "secret-that-is-long-enough-for-hs256-000", algorithm="HS256", headers={"kid": keys.kid})
    assert (await client.get("/v1/meetings", headers={"X-Identity-Token": forged})).status_code == 401
    unsigned = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": "u1", "typ": "access", "exp": int(time.time()) + 600}, None, algorithm="none", headers={"kid": keys.kid})
    assert (await client.get("/v1/meetings", headers={"X-Identity-Token": unsigned})).status_code == 401


@pytest.mark.asyncio
async def test_health_and_runtime_status_need_no_person_but_everything_else_does(tenant_engine):
    client, _, _ = tenant_engine
    assert (await client.get("/v1/health")).status_code == 200
    assert (await client.get("/v1/engine/status")).status_code == 200
    for path in ("/v1/meetings", "/v1/sources", "/v1/labels", "/v1/projects", "/v1/tasks", "/v1/settings", "/v1/stats", "/v1/models"):
        assert (await client.get(path)).status_code == 401, path


@pytest.mark.asyncio
async def test_a_verifier_keeps_working_offline_from_its_cache_and_finds_a_rotated_key(keys, tmp_path):
    cache = tmp_path / "jwks.json"
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return {"keys": [keys.jwk]}

    v = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="x", jwks_file=str(cache), fetch=fetch)
    assert (await v.verify(keys.token("u1"))).subject == "u1" and cache.exists()

    async def down():
        raise OSError("offline")

    offline = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="x", jwks_file=str(cache), fetch=down)
    assert (await offline.verify(keys.token("u2"))).subject == "u2"  # served from the cached copy

    rotated = Keys("k2")
    state = {"keys": [keys.jwk]}

    async def fetch2():
        return {"keys": state["keys"]}

    v2 = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="x", fetch=fetch2)
    await v2.verify(keys.token("u1"))
    state["keys"] = [keys.jwk, rotated.jwk]
    v2._attempted_at = 0  # the account service published a new key since the last look
    assert (await v2.verify(rotated.token("u3"))).subject == "u3"
    with pytest.raises(IdentityError):
        await IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="x", fetch=down).verify(keys.token("u1"))


# ------------------------------------------------------------------- what each person sees

async def make_things(client, headers, tag):
    project = (await client.post("/v1/projects", json={"name": f"Project {tag}"}, headers=headers)).json()
    meeting = (await client.post("/v1/meetings", json={"title": f"Scribe {tag}", "projectId": project["id"]}, headers=headers)).json()
    task = (await client.post(f"/v1/meetings/{meeting['id']}/tasks", json={"title": f"Task {tag}"}, headers=headers)).json()
    label = (await client.post("/v1/labels", json={"name": "shared-name"}, headers=headers)).json()
    thread = (await client.post("/v1/chat/threads", json={"meetingId": meeting["id"]}, headers=headers)).json()
    return {"project": project, "meeting": meeting, "task": task, "label": label, "thread": thread}


@pytest.mark.asyncio
async def test_two_people_never_see_each_others_data(tenant_engine):
    client, _, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    a = await make_things(client, alice, "A")
    b = await make_things(client, bob, "B")  # the same label name is fine: names are unique per person

    for path, key in (("/v1/meetings", "items"), ("/v1/projects", "items"), ("/v1/tasks", "items")):
        mine = (await client.get(path, headers=alice)).json()[key]
        theirs = (await client.get(path, headers=bob)).json()[key]
        assert len(mine) == 1 and len(theirs) == 1, path
        assert {i["id"] for i in mine}.isdisjoint({i["id"] for i in theirs}), path
    assert [l["id"] for l in (await client.get("/v1/labels", headers=alice)).json()] == [a["label"]["id"]]

    a_mid, a_task = a["meeting"]["id"], a["task"]["id"]
    # Bob, holding Alice's ids, gets what a stranger gets: "not found".
    for method, path, body in (
        ("GET", f"/v1/meetings/{a_mid}", None),
        ("PATCH", f"/v1/meetings/{a_mid}", {"title": "mine now"}),
        ("GET", f"/v1/meetings/{a_mid}/transcript", None),
        ("GET", f"/v1/meetings/{a_mid}/tasks", None),
        ("POST", f"/v1/meetings/{a_mid}/tasks", {"title": "sneaky"}),
        ("PATCH", f"/v1/tasks/{a_task}", {"status": "done"}),
        ("GET", f"/v1/projects/{a['project']['id']}", None),
        ("GET", f"/v1/chat/threads/{a['thread']['id']}", None),
        ("POST", f"/v1/meetings/{a_mid}/edits", {"meeting": {"baseRevision": 0, "title": "hijack"}}),
    ):
        res = await client.request(method, path, json=body, headers=bob)
        assert res.status_code == 404, (method, path, res.status_code)
    # Deleting is idempotent (204 whether or not it was there), but for Bob it removes nothing of Alice's...
    assert (await client.delete(f"/v1/tasks/{a_task}", headers=bob)).status_code == 204
    assert (await client.delete(f"/v1/meetings/{a_mid}", headers=bob)).status_code == 204
    assert (await client.get(f"/v1/meetings/{a_mid}", headers=alice)).json()["title"] == "Scribe A"  # ...her scribe is untouched
    assert len((await client.get(f"/v1/meetings/{a_mid}/tasks", headers=alice)).json()["items"]) == 1  # ...and so is her task


@pytest.mark.asyncio
async def test_each_person_can_record_at_the_same_time(tenant_engine):
    client, _, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    ma = (await client.post("/v1/meetings", json={}, headers=alice)).json()["id"]
    mb = (await client.post("/v1/meetings", json={}, headers=bob)).json()["id"]
    assert (await client.post(f"/v1/meetings/{ma}/start", json={"consentAcknowledged": True}, headers=alice)).status_code == 200
    # One recording at a time is a rule per person, so Bob is not told that Alice is recording.
    assert (await client.post(f"/v1/meetings/{mb}/start", json={"consentAcknowledged": True}, headers=bob)).status_code == 200


@pytest.mark.asyncio
async def test_settings_and_delete_all_are_per_person(tenant_engine):
    client, _, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    await make_things(client, alice, "A")
    await make_things(client, bob, "B")
    assert (await client.delete("/v1/data", headers={**alice, "X-Confirm": "delete-all"})).status_code in (200, 204)
    assert (await client.get("/v1/meetings", headers=alice)).json()["items"] == []
    assert (await client.get("/v1/tasks", headers=alice)).json()["items"] == []
    assert len((await client.get("/v1/meetings", headers=bob)).json()["items"]) == 1  # Bob lost nothing
    assert len((await client.get("/v1/tasks", headers=bob)).json()["items"]) == 1


@pytest.mark.asyncio
async def test_a_new_person_gets_a_complete_schema_even_when_two_requests_arrive_together(tenant_engine):
    client, _, as_user = tenant_engine
    carol = as_user("carol-uuid")
    results = await asyncio.gather(*[client.get("/v1/meetings", headers=carol) for _ in range(6)])
    assert all(r.status_code == 200 for r in results)
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        tenant = tenancy.tenant_key("carol-uuid")
        theirs = {r[0] for r in await (await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (tenant,))).fetchall()}
        shared = {r[0] for r in await (await conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name <> 'schema_migrations'")).fetchall()}
    # Every table the engine uses exists in the person's own schema, so a lookup can never fall through to a shared one.
    assert shared <= theirs


@pytest.mark.asyncio
async def test_without_a_tenant_in_scope_the_database_is_refused(tenant_engine):
    _, app, _ = tenant_engine
    pool = app.state.store.pool
    with pytest.raises(tenancy.TenantRequiredError):
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    with tenancy.system_scope():
        async with pool.connection() as conn:
            assert (await (await conn.execute("SELECT 1")).fetchone())[0] == 1


@pytest.mark.asyncio
async def test_a_connection_is_handed_back_on_the_shared_schema(tenant_engine):
    _, app, as_user = tenant_engine
    pool = app.state.store.pool
    as_user("dave-uuid")  # registers the schema for clean-up when the test ends
    key = tenancy.tenant_key("dave-uuid")
    with tenancy.tenant_scope(key):
        async with pool.connection() as conn:
            assert key in (await (await conn.execute("SHOW search_path")).fetchone())[0]
            await conn.execute("CREATE TABLE IF NOT EXISTS scratch (x int)")  # an uncommitted change commits on a clean exit
    with tenancy.system_scope():
        for _ in range(12):  # many borrows: whichever connection comes back must not still point at Dave's schema
            async with pool.connection() as conn:
                assert key not in (await (await conn.execute("SHOW search_path")).fetchone())[0]
    with tenancy.tenant_scope(key):
        async with pool.connection() as conn:
            assert (await (await conn.execute("SELECT to_regclass('scratch') IS NOT NULL")).fetchone())[0] is True


# ---------------------------------------------------- state that is not in the database

@pytest.mark.asyncio
async def test_live_events_go_only_to_the_person_they_belong_to():
    a, b = tenancy.tenant_key("a"), tenancy.tenant_key("b")

    async def listen(key, got):
        with tenancy.tenant_scope(key):
            async with events.subscribe() as queue:
                got["ready"].set()
                got["items"].append(await asyncio.wait_for(queue.get(), 1))

    ga, gb = {"ready": asyncio.Event(), "items": []}, {"ready": asyncio.Event(), "items": []}
    ta, tb = asyncio.create_task(listen(a, ga)), asyncio.create_task(listen(b, gb))
    await ga["ready"].wait()
    await gb["ready"].wait()
    with tenancy.tenant_scope(a):
        events.publish("meeting.update", {"id": "secret-title-of-a"})
    await asyncio.wait_for(ta, 1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tb, 0.2)  # B hears nothing of A's meeting
    assert ga["items"][0]["data"]["id"] == "secret-title-of-a" and gb["items"] == []


@pytest.mark.asyncio
async def test_a_broadcast_reaches_everyone_and_a_publish_with_no_tenant_reaches_no_tenant():
    a, b = tenancy.tenant_key("a"), tenancy.tenant_key("b")
    qs = {}
    stack = []
    for key in (a, b):
        with tenancy.tenant_scope(key):
            cm = events.subscribe()
            qs[key] = await cm.__aenter__()
            stack.append(cm)
    events.publish("model.progress", {"pct": 5}, broadcast=True)
    assert qs[a].qsize() == 1 and qs[b].qsize() == 1
    events.publish("meeting.update", {"id": "x"})  # no tenant in scope: nobody's
    assert qs[a].qsize() == 1 and qs[b].qsize() == 1
    for cm in stack:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_meeting_events_the_same_id_in_two_tenants_do_not_mix():
    a, b = tenancy.tenant_key("a"), tenancy.tenant_key("b")
    with tenancy.tenant_scope(a):
        await meeting_bus.publish("m1", "status", {"who": "a"})
    with tenancy.tenant_scope(b):
        await meeting_bus.publish("m1", "status", {"who": "b"})
        got = await (await meeting_bus.get_bus("m1").subscribe()).get()
    assert got["data"] == {"who": "b"}


@pytest.mark.asyncio
async def test_a_google_token_belongs_to_the_person_who_pushed_it():
    a, b = tenancy.tenant_key("a"), tenancy.tenant_key("b")
    with tenancy.tenant_scope(a):
        await connector_tokens.set_google_token("token-of-a", "2999-01-01T00:00:00Z")
    with tenancy.tenant_scope(b):
        assert await connector_tokens.get_google_access_token() is None
    with tenancy.tenant_scope(a):
        assert (await connector_tokens.get_google_access_token())[0] == "token-of-a"
        await connector_tokens.clear_google_token()


@pytest.mark.asyncio
async def test_a_background_job_is_visible_only_to_its_owner():
    runner = JobRunner()
    done = asyncio.Event()

    async def work():
        done.set()

    with tenancy.tenant_scope(tenancy.tenant_key("a")):
        job_id = await runner.submit("extract", work)
    await asyncio.wait_for(done.wait(), 1)
    with tenancy.tenant_scope(tenancy.tenant_key("a")):
        assert (await runner.get_job(job_id))["id"] == job_id
    with tenancy.tenant_scope(tenancy.tenant_key("b")):
        assert await runner.get_job(job_id) is None


@pytest.mark.asyncio
async def test_model_choices_are_kept_per_person():
    from app.local import model_service as ms

    a, b = tenancy.tenant_key("a"), tenancy.tenant_key("b")
    with tenancy.tenant_scope(a):
        ms.assign_role("main", "model-a")
    with tenancy.tenant_scope(b):
        assert ms.active_model("main") == ""
        ms.assign_role("main", "model-b")
        assert ms.active_model("main") == "model-b"
    with tenancy.tenant_scope(a):
        assert ms.active_model("main") == "model-a"
        ms._selected_roles.clear()
    with tenancy.tenant_scope(b):
        ms._selected_roles.clear()


@pytest.mark.asyncio
async def test_the_live_socket_refuses_a_caller_who_is_not_a_verified_person(tenant_engine):
    from app.local.routers.live import live_meeting

    _, app, _ = tenant_engine

    class FakeSocket:
        scope = {"app": app}
        headers: dict = {}
        closed = None

        async def close(self, code=1000, reason=""):
            self.closed = (code, reason)

    ws = FakeSocket()
    await live_meeting(ws, "any-id", token=None, identity=None)
    assert ws.closed == (1008, "identity required")


def test_a_tenant_key_is_a_safe_stable_schema_name():
    key = tenancy.tenant_key("3f2b8c1e-0000-4000-8000-123456789abc")
    assert key == tenancy.tenant_key("3f2b8c1e-0000-4000-8000-123456789abc") and key != tenancy.tenant_key("someone-else")
    assert key.startswith("t_") and len(key) == 26 and key[2:].isalnum() and key == key.lower()
    with pytest.raises(ValueError):
        tenancy.tenant_key("")


@pytest.mark.asyncio
async def test_ingestion_runs_in_the_owners_schema_and_nobody_elses(tenant_engine, tmp_path):
    client, app, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    doc = tmp_path / "pricing.txt"
    doc.write_text("The enterprise tier costs forty nine dollars per seat per month. " * 5)
    made = await client.post("/v1/sources", json={"kind": "file", "uri": str(doc)}, headers=alice)
    assert made.status_code in (200, 201, 202), made.text
    sid = made.json()["id"]
    for _ in range(100):
        status = (await client.get(f"/v1/sources/{sid}", headers=alice)).json()["status"]
        if status in ("ready", "error"):
            break
        await asyncio.sleep(0.1)
    detail = (await client.get(f"/v1/sources/{sid}", headers=alice)).json()
    assert status == "ready", detail.get("error")
    chunks = (await client.get(f"/v1/sources/{sid}/chunks", headers=alice)).json()
    assert chunks["items"], "the chunks were written into Alice's schema"

    assert (await client.get("/v1/sources", headers=bob)).json()["items"] == []
    assert (await client.get(f"/v1/sources/{sid}", headers=bob)).status_code == 404
    assert (await client.get(f"/v1/sources/{sid}/chunks", headers=bob)).status_code == 404
    # The same file registered by Bob is his own source, not a clash with Alice's.
    again = await client.post("/v1/sources", json={"kind": "file", "uri": str(doc)}, headers=bob)
    assert again.status_code in (200, 201, 202) and again.json()["id"] != sid
    # Nothing leaked into the shared schema.
    with tenancy.system_scope():
        async with app.state.store.pool.connection() as conn:
            assert (await (await conn.execute("SELECT count(*) FROM sources")).fetchone())[0] == 0


@pytest.mark.asyncio
async def test_startup_recovery_finishes_each_persons_interrupted_scribe_in_their_own_schema(tenant_engine):
    from app.local.finalize import recover_unfinished

    client, app, as_user = tenant_engine
    alice, bob = as_user("alice-uuid"), as_user("bob-uuid")
    ma = (await client.post("/v1/meetings", json={"title": "A"}, headers=alice)).json()["id"]
    mb = (await client.post("/v1/meetings", json={"title": "B"}, headers=bob)).json()["id"]
    for sub, mid in (("alice-uuid", ma), ("bob-uuid", mb)):
        with tenancy.tenant_scope(tenancy.tenant_key(sub)):
            async with app.state.store.pool.connection() as conn:
                await conn.execute("UPDATE meetings SET status = 'processing' WHERE id = %s", (mid,))
    assert await recover_unfinished(app) == 2  # found in two different schemas
    for _ in range(100):
        a = (await client.get(f"/v1/meetings/{ma}", headers=alice)).json()["status"]
        b = (await client.get(f"/v1/meetings/{mb}", headers=bob)).json()["status"]
        if a == b == "done":
            break
        await asyncio.sleep(0.1)
    assert (a, b) == ("done", "done")
