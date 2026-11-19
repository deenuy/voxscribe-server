"""Identity claims, roles, one-shot tickets and origins (PL08, gaps T4, T5, T6).

The real engine on a throwaway schema, with the engine token on and, unless a test says otherwise, tenant
isolation on. The live socket and the event stream are driven as raw ASGI connections, so the middleware stack
and the route's own authentication point are exercised exactly as a server would run them.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psycopg
import pytest
from starlette.routing import WebSocketRoute

from app.local import finalize, tenancy, tickets
from app.local.config import Config
from app.local.identity import DEVICE_CONTROL_ROLES, IdentityError, IdentityVerifier
from app.local.migrations import run_migrations
from app.local.voice.host import SessionContext, VoiceHost, WakeState
from tests.test_tenancy import AUDIENCE, DB_URL, ISSUER, Keys
from tests.voice.unit.fakes import fake_ports

TOKEN = "pl08-engine-token"
ORIGIN = "http://localhost:3000"
EVIL = "http://evil.test"
PRODUCTS = ("personal", "ideascribe", "senior_care", "classroom")
STEP_TIMEOUT_S = 20


@dataclass
class Engine:
    app: object
    client: httpx.AsyncClient
    keys: Keys
    created: list[str] = field(default_factory=list)

    def headers(self, subject: str, **claims) -> dict:
        """The engine token and a verified person, both as headers. Claims set to None are left out."""
        self.created.append(tenancy.tenant_key(subject))
        out = {"Authorization": f"Bearer {TOKEN}"}
        if self.app.state.identity is not None:
            out["X-Identity-Token"] = self.keys.token(subject, extra=claims)
        return out

    async def ticket(self, path: str, headers: dict) -> str:
        res = await self.client.post("/v1/tickets", json={"path": path}, headers=headers)
        assert res.status_code == 201, res.text
        assert res.headers["cache-control"] == "no-store"
        body = res.json()
        assert body["path"] == path and body["expiresInSeconds"] == self.app.state.tickets.ttl_s
        return body["ticket"]


@pytest.fixture
async def make_engine(monkeypatch):
    monkeypatch.setenv("VOXSCRIBE_TOKEN", TOKEN)
    from app.local.app import create_app

    made: list[tuple[Engine, object, str]] = []

    async def _make(*, isolation: bool = True, origins: str = ORIGIN) -> Engine:
        data_dir = Path("/tmp/voxscribe_test")
        data_dir.mkdir(parents=True, exist_ok=True)
        base = f"t_{uuid.uuid4().hex[:8]}"
        async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
            await conn.execute(f"CREATE SCHEMA {base}")
            await conn.commit()
        await run_migrations(DB_URL, schema=base)
        identity = dict(identity_issuer=ISSUER, identity_audience=AUDIENCE,
                        identity_jwks_url="http://jwks.test/keys") if isolation else {}
        settings = Config(database_url=DB_URL, data_dir=str(data_dir), providers="fake",
                          allowed_origins=origins, **identity)
        app = await create_app(settings=settings, auth_disabled=False, schema=base)
        keys = Keys()
        if isolation:
            async def fetch():
                return {"keys": [keys.jwk]}

            app.state.identity = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE,
                                                  jwks_url="http://jwks.test/keys", fetch=fetch)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        engine = Engine(app, client, keys)
        made.append((engine, lifespan, base))
        return engine

    yield _make
    while finalize._running:
        await asyncio.gather(*list(finalize._running), return_exceptions=True)
    for engine, lifespan, base in made:
        await engine.client.aclose()
        await lifespan.__aexit__(None, None, None)
        async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
            for key in set(engine.created):
                await conn.execute(f'DROP SCHEMA IF EXISTS "{key}" CASCADE')
            await conn.execute(f"DROP SCHEMA IF EXISTS {base} CASCADE")


def _scope(kind: str, target: str, headers: dict | None) -> dict:
    path, _, query = target.partition("?")
    return {
        "type": kind, "asgi": {"version": "3.0"}, "http_version": "1.1", "scheme": "http" if kind == "http" else "ws",
        "method": "GET", "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "server": ("test", 80), "client": ("127.0.0.1", 50000), "subprotocols": [],
    }


async def open_stream(app, target: str, headers: dict | None = None) -> tuple[int, bytes]:
    """GET a streaming address. Returns (status, body); a 200 stream is disconnected as soon as it starts."""
    started = asyncio.Event()
    out: dict = {"body": b""}

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            started.set()
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    task = asyncio.create_task(app(_scope("http", target, headers), receive, send))
    await asyncio.wait_for(started.wait(), STEP_TIMEOUT_S)
    if out["status"] == 200:
        task.cancel()
    try:
        await asyncio.wait_for(task, STEP_TIMEOUT_S)
    except (asyncio.CancelledError, Exception):
        pass
    return out["status"], out["body"]


async def open_socket(app, target: str, headers: dict | None = None) -> tuple[str, str | None]:
    """Open a WebSocket. Returns ("accept", None) or ("close", reason) for the server's first answer; an accepted
    socket is disconnected at once."""
    answered = asyncio.Event()
    out: dict = {}
    connected = False

    async def receive():
        nonlocal connected
        if not connected:
            connected = True
            return {"type": "websocket.connect"}
        await answered.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        if "kind" not in out:
            if message["type"] == "websocket.accept":
                out.update(kind="accept", reason=None)
            elif message["type"] == "websocket.close":
                out.update(kind="close", reason=message.get("reason"))
            answered.set()

    await asyncio.wait_for(app(_scope("websocket", target, headers), receive, send), STEP_TIMEOUT_S)
    return out["kind"], out["reason"]


async def recording_meeting(engine: Engine, headers: dict) -> str:
    mid = (await engine.client.post("/v1/meetings", json={"title": "pl08"}, headers=headers)).json()["id"]
    res = await engine.client.post(f"/v1/meetings/{mid}/start", json={"consentAcknowledged": True}, headers=headers)
    assert res.status_code == 200, res.text
    return mid


# ── claims ────────────────────────────────────────────────────────────────────

async def test_the_verified_claims_are_carried_and_a_malformed_claim_makes_the_token_invalid():
    keys = Keys()

    async def fetch():
        return {"keys": [keys.jwk]}

    verifier = IdentityVerifier(issuer=ISSUER, audience=AUDIENCE, jwks_url="x", fetch=fetch)
    who = await verifier.verify(keys.token("u1", extra={"org": "o1", "product": "senior_care",
                                                        "device": "hallway", "role": "admin"}))
    assert (who.subject, who.organization, who.product, who.device, who.role) == (
        "u1", "o1", "senior_care", "hallway", "admin")
    bare = await verifier.verify(keys.token("u1", extra={"role": None}))
    assert (bare.organization, bare.product, bare.device, bare.role) == (None, None, None, None)
    assert not bare.may_control_device
    for claim, bad in (("role", 7), ("org", ""), ("product", ["personal"]), ("device", {"id": "x"})):
        with pytest.raises(IdentityError):
            await verifier.verify(keys.token("u1", extra={claim: bad}))


# ── product ───────────────────────────────────────────────────────────────────

async def test_a_product_claim_that_differs_from_the_loaded_profile_is_refused_with_product_mismatch(make_engine):
    engine = await make_engine()
    loaded = engine.app.state.profile.mode
    other = next(p for p in PRODUCTS if p != loaded)

    res = await engine.client.get("/v1/meetings", headers=engine.headers("alice", product=other))
    assert res.status_code == 403 and res.json()["error"] == "product_mismatch"
    res = await engine.client.post("/v1/tickets", json={"path": "/v1/events"},
                                   headers=engine.headers("alice", product=other))
    assert res.status_code == 403 and res.json()["error"] == "product_mismatch"
    kind, reason = await open_socket(engine.app, "/v1/meetings/m1/live", engine.headers("alice", product=other))
    assert (kind, reason) == ("close", "product_mismatch")

    assert (await engine.client.get("/v1/meetings", headers=engine.headers("alice", product=loaded))).status_code == 200
    assert (await engine.client.get("/v1/meetings", headers=engine.headers("alice"))).status_code == 200  # untied


# ── tickets ───────────────────────────────────────────────────────────────────

def test_a_ticket_is_single_use_short_lived_and_bound_to_its_path():
    now = [1000.0]
    book = tickets.TicketBook(30, clock=lambda: now[0])
    value = book.issue("/v1/events", tenant="t_a", identity=None)
    grant = book.redeem(value, "/v1/events")
    assert grant is not None and grant.tenant == "t_a"
    assert book.redeem(value, "/v1/events") is None                       # used

    value = book.issue("/v1/events", tenant="t_a", identity=None)
    now[0] += 30
    assert book.redeem(value, "/v1/events") is None                       # expired

    value = book.issue("/v1/events", tenant="t_a", identity=None)
    assert book.redeem(value, "/v1/meetings/m1/live") is None             # another route
    assert book.redeem(value, "/v1/events") is None                       # and burnt by trying
    assert book.redeem("", "/v1/events") is None and book.redeem("made-up", "/v1/events") is None

    with pytest.raises(ValueError):
        book.issue("/v1/meetings", tenant="t_a", identity=None)
    for _ in range(tickets.MAX_OUTSTANDING_PER_TENANT):
        book.issue("/v1/events", tenant="t_b", identity=None)
    with pytest.raises(tickets.TooManyTickets):
        book.issue("/v1/events", tenant="t_b", identity=None)
    book.issue("/v1/events", tenant="t_c", identity=None)                 # another tenant is unaffected
    now[0] += 31
    book.issue("/v1/events", tenant="t_b", identity=None)                 # expired ones are swept
    assert len(book) == 1


async def test_a_valid_ticket_opens_the_event_stream_once(make_engine):
    engine = await make_engine()
    value = await engine.ticket("/v1/events", engine.headers("alice"))
    assert (await open_stream(engine.app, f"/v1/events?ticket={value}"))[0] == 200
    status, body = await open_stream(engine.app, f"/v1/events?ticket={value}")
    assert status == 401 and b"ticket_invalid" in body


async def test_an_expired_ticket_is_refused_on_the_event_stream_and_the_live_socket(make_engine):
    engine = await make_engine()
    now = [1000.0]
    engine.app.state.tickets._clock = lambda: now[0]
    alice = engine.headers("alice")
    mid = await recording_meeting(engine, alice)
    stream = await engine.ticket("/v1/events", alice)
    socket = await engine.ticket(f"/v1/meetings/{mid}/live", alice)
    now[0] += engine.app.state.tickets.ttl_s
    status, body = await open_stream(engine.app, f"/v1/events?ticket={stream}")
    assert status == 401 and b"ticket_invalid" in body
    assert await open_socket(engine.app, f"/v1/meetings/{mid}/live?ticket={socket}") == ("close", "ticket_invalid")


async def test_a_valid_ticket_opens_the_live_socket_once(make_engine):
    engine = await make_engine()
    alice = engine.headers("alice")
    mid = await recording_meeting(engine, alice)
    value = await engine.ticket(f"/v1/meetings/{mid}/live", alice)
    assert await open_socket(engine.app, f"/v1/meetings/{mid}/live?ticket={value}") == ("accept", None)
    assert await open_socket(engine.app, f"/v1/meetings/{mid}/live?ticket={value}") == ("close", "ticket_invalid")


async def test_a_ticket_works_only_on_its_route_and_for_the_person_it_was_issued_to(make_engine):
    engine = await make_engine()
    alice, bob = engine.headers("alice"), engine.headers("bob")
    bobs = await recording_meeting(engine, bob)

    # Issued for the event stream, presented at a live socket: refused, and burnt.
    value = await engine.ticket("/v1/events", alice)
    assert await open_socket(engine.app, f"/v1/meetings/{bobs}/live?ticket={value}") == ("close", "ticket_invalid")
    assert (await open_stream(engine.app, f"/v1/events?ticket={value}"))[0] == 401

    # Alice's ticket for Bob's meeting runs as Alice: Bob's meeting does not exist for her.
    value = await engine.ticket(f"/v1/meetings/{bobs}/live", alice)
    assert await open_socket(engine.app, f"/v1/meetings/{bobs}/live?ticket={value}") == ("close", "meeting not found")

    # A ticket cannot be had for any other route, and a ticket does not open an ordinary route.
    res = await engine.client.post("/v1/tickets", json={"path": "/v1/meetings"}, headers=alice)
    assert res.status_code == 400 and res.json()["error"] == "invalid_request"
    value = await engine.ticket("/v1/events", alice)
    assert (await engine.client.get(f"/v1/meetings?ticket={value}")).status_code == 401


async def test_every_socket_and_stream_route_accepts_a_ticket(make_engine):
    engine = await make_engine()
    def walk(routes):
        for route in routes:
            inner = getattr(route, "original_router", None)     # a router the app included
            yield from walk(inner.routes) if inner is not None else (route,)

    sockets = [r.path for r in walk(engine.app.routes) if isinstance(r, WebSocketRoute)]
    assert sockets, "the route walk found no socket"
    for path in sockets:
        concrete = path.replace("{meeting_id}", "m1")
        assert "{" not in concrete and tickets.ticketable(concrete), path
    assert tickets.ticketable("/v1/events")


async def test_with_isolation_on_no_long_lived_credential_is_read_from_the_address(make_engine):
    engine = await make_engine()
    identity = engine.keys.token("alice")
    engine.created.append(tenancy.tenant_key("alice"))
    status, body = await open_stream(engine.app, f"/v1/events?token={TOKEN}&identity={identity}")
    assert status == 401 and b"token required" in body
    status, body = await open_stream(engine.app, f"/v1/events?identity={identity}",
                                     {"Authorization": f"Bearer {TOKEN}"})
    assert status == 401 and b"identity_required" in body
    assert await open_socket(engine.app, f"/v1/meetings/m1/live?token={TOKEN}&identity={identity}") == (
        "close", "identity required")
    # Headers still work for a client that can set them.
    assert (await open_stream(engine.app, "/v1/events", engine.headers("alice")))[0] == 200


async def test_with_isolation_off_the_engine_token_in_the_address_still_opens_the_stream(make_engine):
    """Kept for the desktop app until its client asks for tickets; off when isolation is on (above)."""
    engine = await make_engine(isolation=False)
    assert (await open_stream(engine.app, f"/v1/events?token={TOKEN}"))[0] == 200
    assert (await open_stream(engine.app, "/v1/events?token=wrong"))[0] == 401
    value = await engine.ticket("/v1/events", engine.headers("nobody"))
    assert (await open_stream(engine.app, f"/v1/events?ticket={value}"))[0] == 200
    assert (await open_stream(engine.app, f"/v1/events?ticket={value}"))[0] == 401


# ── origins ───────────────────────────────────────────────────────────────────

async def test_with_isolation_on_an_origin_outside_the_configured_list_is_refused(make_engine):
    engine = await make_engine()
    res = await engine.client.get("/v1/meetings", headers={**engine.headers("alice"), "Origin": EVIL})
    assert res.status_code == 403 and res.json()["error"] == "origin_not_allowed"
    assert "access-control-allow-origin" not in res.headers
    assert (await engine.client.get("/v1/health", headers={"Origin": EVIL})).status_code == 403
    pre = await engine.client.options("/v1/meetings", headers={
        "Origin": EVIL, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"})
    assert pre.status_code != 200 and "access-control-allow-origin" not in pre.headers
    assert await open_socket(engine.app, "/v1/meetings/m1/live", {"Origin": EVIL}) == ("close", "origin_not_allowed")

    ok = await engine.client.get("/v1/meetings", headers={**engine.headers("alice"), "Origin": ORIGIN})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert (await engine.client.get("/v1/meetings", headers=engine.headers("alice"))).status_code == 200  # no Origin


async def test_with_isolation_on_and_no_origins_configured_no_origin_is_allowed(make_engine):
    engine = await make_engine(origins="")
    res = await engine.client.get("/v1/health", headers={"Origin": ORIGIN})
    assert res.status_code == 403 and res.json()["error"] == "origin_not_allowed"


async def test_with_isolation_off_cross_origin_access_is_unchanged(make_engine):
    engine = await make_engine(isolation=False, origins="")
    res = await engine.client.get("/v1/health", headers={"Origin": EVIL})
    assert res.status_code == 200 and res.headers["access-control-allow-origin"] == "*"


# ── roles: who may bind and unbind the audio device ───────────────────────────

def _host_for(app) -> VoiceHost:
    host = VoiceHost.from_app(app)
    host.ports = fake_ports()                # built, never started: no audio, no models
    app.state.voice_host = host
    return host


async def test_a_caller_without_a_device_role_cannot_arm_or_disarm_the_audio_device(make_engine):
    engine = await make_engine()
    host = _host_for(engine.app)
    device = host.binding.device

    for refused in ({"role": "viewer"}, {"role": None}, {"role": "caregiver"}):
        res = await engine.client.post("/v1/wake/arm", headers=engine.headers("vic", **refused))
        assert res.status_code == 403 and res.json()["error"] == "role_not_permitted", refused
        assert not host.bound and host.state == WakeState.OFF
    res = await engine.client.post("/v1/wake/arm", headers=engine.headers("mel", role="member", device="elsewhere"))
    assert res.status_code == 403 and res.json()["error"] == "device_mismatch"
    assert not host.bound

    res = await engine.client.post("/v1/wake/arm", headers=engine.headers("mel", role="member", org="o1",
                                                                          device=device))
    assert res.status_code == 200 and res.json() == {"state": "armed", "changed": True}
    assert (host.binding.tenant, host.binding.organization, host.binding.role) == (
        tenancy.tenant_key("mel"), "o1", "member")

    res = await engine.client.post("/v1/wake/disarm", headers=engine.headers("vic", role="viewer"))
    assert res.status_code == 403 and res.json()["error"] == "role_not_permitted"
    assert host.state == WakeState.ARMED
    res = await engine.client.post("/v1/wake/disarm", headers=engine.headers("mel", role="user"))
    assert res.status_code == 200 and host.state == WakeState.OFF
    assert DEVICE_CONTROL_ROLES == {"owner", "admin", "member", "user"}


async def test_a_permitted_role_still_cannot_take_the_device_while_a_session_runs(make_engine):
    engine = await make_engine()
    host = _host_for(engine.app)
    assert (await engine.client.post("/v1/wake/arm", headers=engine.headers("ann", role="owner"))).status_code == 200
    host._open = SessionContext(binding=host.binding)
    res = await engine.client.post("/v1/wake/arm", headers=engine.headers("ben", role="admin"))
    assert res.status_code == 409 and res.json()["error"] == "voice_session_in_progress"
    host._open = None


async def test_with_isolation_off_arming_and_disarming_are_unchanged(make_engine):
    engine = await make_engine(isolation=False)
    host = _host_for(engine.app)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert (await engine.client.post("/v1/wake/arm", headers=auth)).json() == {"state": "armed", "changed": True}
    assert host.binding.tenant is None and host.binding.role is None
    assert (await engine.client.post("/v1/wake/disarm", headers=auth)).json()["state"] == "off"
