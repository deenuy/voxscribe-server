"""Session context (PL05): a voice session is an object that says who and what it is for, and the
host does everything for it inside its scope.

Host level, every port a fake: what the session carries, that every store call and event of a
session happens with that session in scope, what an unbound host refuses when a tenant is
required, and what binding the device to someone else does. Then the boundary: no module under
app/local/voice reaches the engine's connection pool except through the scope adapter, and the
scope adapter hands it out only inside a tenant's scope. No database here; the same things
against PostgreSQL with identity on are in tests/test_voice_tenancy.py.
"""
from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.local import tenancy
from app.local.voice import host as host_mod
from app.local.voice.adapters import scope as scope_mod
from app.local.voice.adapters.scope import TenantSessionScope, profile_version, scoped_pool
from app.local.voice.host import (
    SessionBinding,
    SessionContext,
    SessionInProgressError,
    SessionScope,
    SessionTenantRequiredError,
    VoiceConfig,
    VoiceHost,
    WakeState,
    current_session,
)

from .fakes import EndlessQueue, FakeDelegate, FakeRecognizer, FakeStore, fake_ports

TURNS = ["What is one?", "And two?"]
ALICE = SessionBinding(tenant="t_alice", product="personal", device="kitchen")
BOB = SessionBinding(tenant="t_bob", product="personal", device="kitchen")
SESSION_TIMEOUT_S = 10
VOICE_DIR = Path(host_mod.__file__).parent
SCOPE_MODULE = Path(scope_mod.__file__)


class RecordingScope(SessionScope):
    """A scope that requires a tenant, as the server's does with isolation on, and keeps the
    tenant it is inside so a test can see where a call happened."""

    tenant_required = True

    def __init__(self) -> None:
        self.inside: list[str] = []          # tenants entered and not yet left, innermost last
        self.entered: list[str] = []

    def profile_version(self) -> str:
        return "personal@test"

    @contextmanager
    def bound(self, binding: SessionBinding):
        if binding.tenant is None:
            raise SessionTenantRequiredError("no tenant")
        self.inside.append(binding.tenant)
        self.entered.append(binding.tenant)
        try:
            yield
        finally:
            self.inside.pop()

    @property
    def tenant(self) -> str | None:
        return self.inside[-1] if self.inside else None


class ScopeSeeingStore(FakeStore):
    """Records, for every store call, the tenant scope and the session it was made in."""

    CALLS = ("record_wake", "create_session", "add_segment", "begin_turn", "session_tools",
             "add_turn", "close_session", "discard_session")

    def __init__(self, scope: RecordingScope) -> None:
        super().__init__()
        self.seen: list[tuple[str, str | None, SessionContext | None]] = []
        for name in self.CALLS:
            setattr(self, name, self._watch(name, getattr(self, name), scope))

    def _watch(self, name, call, scope):
        async def watched(*args, **kw):
            self.seen.append((name, scope.tenant, current_session()))
            return await call(*args, **kw)

        return watched


def _bound_host(binding: SessionBinding | None = ALICE, **ports):
    scope = RecordingScope()
    store = ScopeSeeingStore(scope)
    p = fake_ports(store=store, **ports)
    events: list[tuple[str, str | None, SessionContext | None]] = []
    host = VoiceHost(VoiceConfig(), ports=p, scope=scope, binding=binding,
                     publish=lambda kind, _data: events.append((kind, scope.tenant,
                                                                current_session())))
    return host, scope, store, events


# ── the object ────────────────────────────────────────────────────────────────


async def test_a_session_carries_tenant_product_device_and_profile_version():
    host, _scope, store, _events = _bound_host(recognizer=FakeRecognizer(TURNS))
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                           SESSION_TIMEOUT_S)

    sessions = {id(s): s for _name, _tenant, s in store.seen}
    (session,) = sessions.values()           # one object for the whole session
    assert (session.tenant, session.product, session.device) == ("t_alice", "personal", "kitchen")
    assert session.binding == ALICE and session.profile_version == "personal@test"
    assert session.session_id == "s1"
    assert current_session() is None         # the scope is left when the session ends


async def test_every_store_call_and_event_of_a_session_happens_inside_its_scope():
    host, _scope, store, events = _bound_host(recognizer=FakeRecognizer(TURNS))
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake"),
                           SESSION_TIMEOUT_S)

    names = [name for name, _tenant, _session in store.seen]
    assert {"create_session", "add_segment", "begin_turn", "add_turn", "close_session"} <= set(names)
    assert {tenant for _name, tenant, _session in store.seen} == {"t_alice"}
    assert all(session is not None for _name, _tenant, session in store.seen)

    assert {kind for kind, _tenant, _session in events} >= {"wake.transcript", "wake.reply"}
    assert {tenant for _kind, tenant, _session in events} == {"t_alice"}
    assert all(session is not None and session.tenant == "t_alice"
               for _kind, _tenant, session in events)


async def test_a_session_can_be_opened_for_a_binding_that_is_not_the_devices():
    """The door a network session uses: the binding comes with the call, from the connection's
    credential. The device's own binding is not consulted and does not change."""
    host, _scope, store, _events = _bound_host(recognizer=FakeRecognizer(TURNS))
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9, detector="fake",
                                            binding=BOB), SESSION_TIMEOUT_S)
    assert {tenant for _name, tenant, _session in store.seen} == {"t_bob"}
    assert host.binding == ALICE


async def test_without_a_scope_the_host_behaves_as_before_and_still_has_a_session_object():
    """Isolation off: nothing to bind, nothing refused, and the session is still an object."""
    seen: list[SessionContext | None] = []

    class Store(FakeStore):
        async def create_session(self, wake_score, *, detector=None, speech_ratio=None) -> str:
            seen.append(current_session())
            return await super().create_session(wake_score, detector=detector,
                                                speech_ratio=speech_ratio)

    host = VoiceHost(VoiceConfig(), ports=fake_ports(store=Store(),
                                                     recognizer=FakeRecognizer(TURNS)))
    assert host.bound and host.arm()
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9), SESSION_TIMEOUT_S)
    (session,) = seen
    assert session is not None and session.tenant is None and session.binding == SessionBinding()


# ── no tenant bound ───────────────────────────────────────────────────────────


async def test_an_unbound_host_refuses_to_arm_and_to_open_a_session_and_writes_nothing():
    host, scope, store, events = _bound_host(binding=None)
    assert not host.bound

    with pytest.raises(SessionTenantRequiredError) as refused:
        host.arm()
    assert refused.value.code == "voice_tenant_unbound"
    assert host.state == WakeState.OFF

    with pytest.raises(SessionTenantRequiredError):
        await host.run_session(EndlessQueue(), score=0.9)
    with pytest.raises(SessionTenantRequiredError):
        await host.trigger()
    with pytest.raises(SessionTenantRequiredError):       # a binding with no tenant, handed in
        await host.run_session(EndlessQueue(), score=0.9, binding=SessionBinding(product="x"))

    assert store.seen == [] and store.sessions == [] and events == [] and scope.entered == []


async def test_an_unbound_host_started_with_the_app_stays_off_until_someone_binds_it():
    host, scope, store, _events = _bound_host(binding=None)
    host.ports.audio_in.subscribe = lambda: EndlessQueue()
    host.start()
    try:
        for _ in range(5):
            await asyncio.sleep(0.01)
        assert host.state == WakeState.OFF and host.ports.audio_in.starts == 0

        assert host.bind(ALICE) and host.arm()
        for _ in range(200):
            if host.ports.audio_in.starts:
                break
            await asyncio.sleep(0.01)
        assert host.state == WakeState.ARMED and scope.inside == ["t_alice"]

        # Unbinding (a binding with no tenant) takes it out of the scope and back to off.
        assert host.bind(SessionBinding(product="personal", device="kitchen"))
        for _ in range(200):
            if host.state == WakeState.OFF:
                break
            await asyncio.sleep(0.01)
        assert host.state == WakeState.OFF and scope.inside == []
    finally:
        await host.stop()
    assert store.sessions == []


# ── binding the device to someone else ────────────────────────────────────────


async def test_the_device_cannot_be_bound_to_another_tenant_while_a_session_runs():
    host, _scope, store, _events = _bound_host(recognizer=FakeRecognizer(TURNS))
    assert host.bind(ALICE) is False                      # the same binding: nothing changes
    refusals: list[str] = []
    begin_turn = store.begin_turn

    async def bind_mid_session(*args, **kw):
        with pytest.raises(SessionInProgressError) as refused:
            host.bind(BOB)
        refusals.append(refused.value.code)
        return await begin_turn(*args, **kw)

    store.begin_turn = bind_mid_session
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9), SESSION_TIMEOUT_S)
    assert refusals == ["voice_session_in_progress"] * len(TURNS) and host.binding == ALICE
    assert {tenant for _name, tenant, _session in store.seen} == {"t_alice"}

    assert host.bind(BOB) and host.binding == BOB         # between sessions it is allowed


async def test_binding_another_tenant_drops_what_the_delegate_held_inside_the_old_scope():
    class Delegate(FakeDelegate):
        forgot_in: list[str | None]

        def forget(self) -> None:
            self.forgot_in.append(scope.tenant)

    delegate = Delegate()
    delegate.forgot_in = []
    host, scope, store, _events = _bound_host(recognizer=FakeRecognizer(TURNS), delegate=delegate)

    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9), SESSION_TIMEOUT_S)
    assert delegate.forgot_in == []                       # the same person: nothing to drop

    host.bind(BOB)
    host.ports.recognizer = FakeRecognizer(TURNS)
    await asyncio.wait_for(host.run_session(EndlessQueue(), score=0.9), SESSION_TIMEOUT_S)
    assert delegate.forgot_in == ["t_alice"]              # dropped once, in the old scope
    tenants = [tenant for name, tenant, _session in store.seen if name == "create_session"]
    assert tenants == ["t_alice", "t_bob"]


def test_the_delegate_forgets_open_tasks_waiting_results_and_the_stale_offer():
    from app.local.voice.adapters.delegate_ollama import OllamaDelegate
    from app.local.voice.delegate_lane import DelegateLane, TaskStatus

    lane = DelegateLane()                                 # no thinker: tasks stay queued
    delegate = OllamaDelegate(SimpleNamespace(state=SimpleNamespace(settings=None)), lane=lane)
    running = lane.submit("look up the tides", session_id="s1")
    waiting = lane.submit("look up the weather", session_id="s1")
    lane.complete(waiting.task_id, "It is sunny.")
    delegate._offered = waiting
    delegate._contexts[running.task_id] = [{"role": "user", "content": "earlier"}]

    delegate.forget()
    assert running.status is TaskStatus.CANCELLED and waiting.status is TaskStatus.CANCELLED
    assert lane.latest_open() is None and lane.get_pending() == [] and lane.active() == []
    assert delegate._offered is None and delegate._contexts == {}


# ── the server's scope adapter, without a database ────────────────────────────


class _Pool:
    def __init__(self, enforced: bool) -> None:
        self.enforced = enforced


def _app(enforced: bool, **state):
    return SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(pool=_Pool(enforced)),
                                                 **state))


def test_the_tenant_scope_enters_the_bindings_tenant_and_refuses_a_binding_without_one():
    scope = TenantSessionScope(_app(enforced=True))
    assert scope.tenant_required
    with scope.bound(ALICE):
        assert tenancy.current_tenant() == "t_alice"
    assert tenancy.current_tenant() is None
    with scope.session(ALICE) as session:
        assert tenancy.current_tenant() == "t_alice" and current_session() is session
    assert current_session() is None
    with pytest.raises(SessionTenantRequiredError):
        with scope.bound(SessionBinding()):
            pass
    with pytest.raises(SessionTenantRequiredError):
        with scope.session(SessionBinding()):
            pass


def test_with_isolation_off_the_tenant_scope_enters_nothing_and_refuses_nothing():
    scope = TenantSessionScope(_app(enforced=False))
    assert not scope.tenant_required
    with scope.session(SessionBinding()) as session:
        assert tenancy.current() is None and session.tenant is None
    host = VoiceHost(VoiceConfig(), ports=fake_ports(), scope=scope,
                     binding=scope.device_binding())
    assert host.bound and host.arm()


def test_the_device_binding_takes_product_from_the_profile_and_device_from_settings():
    profile = SimpleNamespace(mode="senior_care", quiet_hours_enabled=False)
    app = _app(enforced=True, profile=profile, settings=SimpleNamespace(device_id="hallway"))
    scope = TenantSessionScope(app)
    assert scope.device_binding() == SessionBinding(None, "senior_care", "hallway")
    assert scope.device_binding("t_alice") == SessionBinding("t_alice", "senior_care", "hallway")
    assert TenantSessionScope(_app(enforced=True)).device_binding().device == scope_mod.LOCAL_DEVICE

    version = scope.profile_version()
    assert version.startswith("senior_care@") and version == profile_version(profile)
    changed = SimpleNamespace(mode="senior_care", quiet_hours_enabled=True)
    assert profile_version(changed) != version            # any change to the profile shows
    assert profile_version(None) == ""


def test_the_scoped_pool_is_handed_out_only_inside_a_tenants_scope():
    app = _app(enforced=True)
    pool = app.state.store.pool
    with pytest.raises(tenancy.TenantRequiredError):      # nothing in scope
        scoped_pool(app)
    with tenancy.system_scope(), pytest.raises(tenancy.TenantRequiredError):
        scoped_pool(app)                                  # platform work is not a tenant
    with tenancy.tenant_scope("t_alice"):
        assert scoped_pool(app) is pool
    scope = TenantSessionScope(app)
    with scope.session(ALICE):
        assert scoped_pool(app) is pool
        with tenancy.tenant_scope("t_bob"), pytest.raises(tenancy.TenantRequiredError):
            scoped_pool(app)                              # not the session's tenant

    off = _app(enforced=False)
    assert scoped_pool(off) is off.state.store.pool       # isolation off: as before


async def test_the_thinkers_web_search_takes_its_pool_through_the_scope_adapter(monkeypatch):
    """reply_chain.search_answer runs on the voice delegate lane; outside the tenant's scope, or in a session of
    another tenant, it is refused before the search is sent anywhere."""
    from app.local import reply_chain
    from app.local.tools import web_search

    searched: list[str] = []

    async def search(query):
        searched.append(query)
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(web_search, "search", search)
    app = _app(enforced=True, settings=None)
    with pytest.raises(tenancy.TenantRequiredError):
        await reply_chain.search_answer(app, "m1", "look it up", "it", task_id="k1")
    with TenantSessionScope(app).session(ALICE), tenancy.tenant_scope("t_bob"):
        with pytest.raises(tenancy.TenantRequiredError):
            await reply_chain.search_answer(app, "m1", "look it up", "it", task_id="k2")
    assert searched == []


# ── the boundary ──────────────────────────────────────────────────────────────


DATABASE_MODULES = ("psycopg", "psycopg_pool", "app.local.db", "app.local.store")


def _is_database_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in DATABASE_MODULES)


def _pool_reaches(path: Path) -> list[str]:
    """Every way this module could get a database connection without the scope adapter: the
    engine's pool by name (`<x>.state.store`, `<y>.pool`) or a connection of its own."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute):
            if node.attr == "pool":
                found.append(f"{path.name}:{node.lineno} .pool")
            elif (node.attr == "store" and isinstance(node.value, ast.Attribute)
                  and node.value.attr == "state"):
                found.append(f"{path.name}:{node.lineno} .state.store")
        elif isinstance(node, ast.Import):
            found += [f"{path.name}:{node.lineno} import {a.name}" for a in node.names
                      if _is_database_module(a.name)]
        elif isinstance(node, ast.ImportFrom) and _is_database_module(node.module or ""):
            found.append(f"{path.name}:{node.lineno} from {node.module}")
    return found


def test_no_voice_module_reaches_the_pool_except_through_the_scope_adapter():
    """A connection borrowed under app/local/voice comes from scoped_pool(), which refuses
    outside a tenant's scope when isolation is on (the test above). So a module that names the
    engine's pool itself, or opens its own connection, is a borrow that can escape the scope:
    this fails on it and says where."""
    modules = sorted(VOICE_DIR.rglob("*.py"))
    assert SCOPE_MODULE in modules and len(modules) > 10
    outside = [reach for path in modules if path != SCOPE_MODULE for reach in _pool_reaches(path)]
    assert outside == []
    assert _pool_reaches(SCOPE_MODULE)       # the scan does see the one module that may


def test_the_boundary_scan_catches_a_direct_borrow(tmp_path):
    bad = tmp_path / "bad_adapter.py"
    bad.write_text("import psycopg\n\n"
                   "async def save(app):\n"
                   "    async with app.state.store.pool.connection() as conn:\n"
                   "        await conn.execute('SELECT 1')\n")
    reaches = _pool_reaches(bad)
    assert any(".pool" in r for r in reaches) and any(".state.store" in r for r in reaches)
    assert any("import psycopg" in r for r in reaches)
