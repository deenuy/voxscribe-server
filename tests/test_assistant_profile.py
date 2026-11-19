"""Assistant profile per tenant, persisted (PL07, X8, SIM).

Acceptance criteria
  AC1: a change by tenant A is not returned to tenant B
  AC2: a change survives an engine restart (a new store over the same database returns it)
  AC3: a tenant with no row gets the product profile defaults; a reset returns to them
  AC4: the profile version on the session row changes when the tenant changes its profile
  AC5 (product clamp): a voice_id that is not a known local voice is silently clamped to ""
       in senior_care and classroom modes (enforced by patch(); a cloud voice stays allowed
       in personal mode)

AC1 and AC4 require tenant isolation (enforced pool) to demonstrate per-tenant separation.
AC2 and AC3 work in any mode; they use the shared None-tenant path so no identity setup is needed.
AC5 is a pure in-memory test; no database required.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest

DB_URL = "postgresql://localhost:5432/ideascribe_dev"
ISSUER, AUDIENCE = "http://issuer.test", "ideascribe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fresh_schema(name_hint: str = "pf") -> str:
    """Create and migrate a fresh throwaway schema; return its name."""
    from app.local.migrations import run_migrations

    schema = f"t_{name_hint}_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.commit()
    await run_migrations(DB_URL, schema=schema)
    return schema


async def _drop_schema(schema: str) -> None:
    try:
        async with await psycopg.AsyncConnection.connect(DB_URL, autocommit=True) as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    except Exception:
        pass


async def _make_app_simple(schema: str):
    """App without identity (isolation off; single shared schema)."""
    from app.local.app import create_app
    from app.local.config import Config

    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    return await create_app(
        settings=Config(database_url=DB_URL, data_dir=str(data_dir), providers="fake"),
        auth_disabled=True,
        schema=schema,
    )


async def _make_app_enforced(schema: str):
    """App with identity configured so the pool enforces tenant isolation."""
    from app.local.app import create_app
    from app.local.config import Config

    data_dir = Path("/tmp/voxscribe_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    return await create_app(
        settings=Config(
            database_url=DB_URL,
            data_dir=str(data_dir),
            providers="fake",
            identity_issuer=ISSUER,
            identity_audience=AUDIENCE,
            identity_jwks_url="http://jwks.test/keys",
        ),
        auth_disabled=True,
        schema=schema,
    )


# ---------------------------------------------------------------------------
# AC1: a change by tenant A is not returned to tenant B
# (requires enforced pool so separate tenant schemas are actually used)
# ---------------------------------------------------------------------------


async def test_ac1_change_by_tenant_a_not_visible_to_tenant_b():
    """AC1: each tenant has their own profile row; A's patch is invisible to B."""
    from app.local import tenancy
    from app.local.voice.adapters.scope import (
        load_profile_for_request,
        save_profile_for_request,
    )

    base = await _fresh_schema("ac1base")
    tenant_a = tenancy.tenant_key(f"alice-ac1-{base}")
    tenant_b = tenancy.tenant_key(f"bob-ac1-{base}")
    try:
        app = await _make_app_enforced(base)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        try:
            pool = app.state.store.pool

            # Provision both tenant schemas under the enforced pool.
            with tenancy.tenant_scope(tenant_a):
                await pool.provisioner.ensure(tenant_a)
            with tenancy.tenant_scope(tenant_b):
                await pool.provisioner.ensure(tenant_b)

            # A patches their profile inside their scope.
            with tenancy.tenant_scope(tenant_a):
                await save_profile_for_request(
                    app, {"assistant": {"name": "aria", "wake_phrase": "hey aria"}}, tenant_a
                )

            # B reads their profile inside their scope -- should still have defaults.
            with tenancy.tenant_scope(tenant_b):
                view_b = await load_profile_for_request(app, tenant_b)
            startup = app.state.profile_store.startup
            assert view_b.assistant.name == startup.name, (
                f"AC1 fail: tenant B sees tenant A's name: {view_b.assistant.name!r}"
            )

            # A reads their profile -- should see the patch.
            with tenancy.tenant_scope(tenant_a):
                view_a = await load_profile_for_request(app, tenant_a)
            assert view_a.assistant.name == "aria"
            assert view_a.assistant.wake_phrase == "hey aria"
        finally:
            await lifespan.__aexit__(None, None, None)
    finally:
        await _drop_schema(tenant_a)
        await _drop_schema(tenant_b)
        await _drop_schema(base)


# ---------------------------------------------------------------------------
# AC2: a change survives an engine restart
# (uses the shared None-tenant path: works with isolation off)
# ---------------------------------------------------------------------------


async def test_ac2_change_survives_restart():
    """AC2: a patched profile is re-read from DB after a fresh store is created."""
    from app.local.voice.adapters.scope import (
        load_profile_for_request,
        save_profile_for_request,
    )

    schema = await _fresh_schema("ac2")
    try:
        # First app instance: patch the shared (None-tenant) profile.
        app1 = await _make_app_simple(schema)
        lifespan1 = app1.router.lifespan_context(app1)
        await lifespan1.__aenter__()
        try:
            await save_profile_for_request(
                app1, {"assistant": {"name": "reborn", "speed": 1.7}}, None
            )
        finally:
            await lifespan1.__aexit__(None, None, None)

        # Second app instance: fresh in-memory store, same DB.
        app2 = await _make_app_simple(schema)
        lifespan2 = app2.router.lifespan_context(app2)
        await lifespan2.__aenter__()
        try:
            view = await load_profile_for_request(app2, None)
            assert view.assistant.name == "reborn", (
                f"AC2 fail: name after restart is {view.assistant.name!r}, expected 'reborn'"
            )
            assert abs(view.assistant.speed - 1.7) < 1e-5, (
                f"AC2 fail: speed after restart is {view.assistant.speed}, expected 1.7"
            )
        finally:
            await lifespan2.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)


# ---------------------------------------------------------------------------
# AC3: a tenant with no row gets product-profile defaults; reset returns to them
# ---------------------------------------------------------------------------


async def test_ac3_no_row_returns_product_defaults_and_reset_restores_them():
    """AC3: a fresh schema returns product-profile defaults; reset brings them back."""
    from app.local.voice.adapters.scope import (
        load_profile_for_request,
        save_profile_for_request,
        reset_profile_for_request,
    )

    schema = await _fresh_schema("ac3")
    try:
        app = await _make_app_simple(schema)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        try:
            startup = app.state.profile_store.startup

            # No row: should get startup defaults.
            view = await load_profile_for_request(app, None)
            assert view.assistant.name == startup.name
            assert view.assistant.wake_phrase == startup.wake_phrase
            assert abs(view.assistant.speed - startup.speed) < 1e-5

            # Patch and then reset.
            await save_profile_for_request(
                app, {"assistant": {"name": "custom", "speed": 1.9}}, None
            )
            view_patched = await load_profile_for_request(app, None)
            assert view_patched.assistant.name == "custom"

            view_reset = await reset_profile_for_request(app, None)
            assert view_reset.assistant.name == startup.name, (
                f"AC3 fail: after reset name is {view_reset.assistant.name!r}, "
                f"expected {startup.name!r}"
            )
            assert abs(view_reset.assistant.speed - startup.speed) < 1e-5

            # Read from DB confirms the row is gone (should return defaults again).
            view_after_reset = await load_profile_for_request(app, None)
            assert view_after_reset.assistant.name == startup.name
        finally:
            await lifespan.__aexit__(None, None, None)
    finally:
        await _drop_schema(schema)


# ---------------------------------------------------------------------------
# AC4: the profile version on the session row changes when the tenant patches
# (uses enforced pool so tenant revision is per-tenant)
# ---------------------------------------------------------------------------


async def test_ac4_profile_version_changes_when_tenant_patches():
    """AC4: TenantSessionScope.profile_version() changes after a tenant patches their profile."""
    from app.local import tenancy
    from app.local.voice.adapters.scope import (
        TenantSessionScope,
        save_profile_for_request,
        load_profile_for_request,
    )

    base = await _fresh_schema("ac4base")
    tenant = tenancy.tenant_key(f"ver-user-{base}")
    try:
        app = await _make_app_enforced(base)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        try:
            pool = app.state.store.pool
            with tenancy.tenant_scope(tenant):
                await pool.provisioner.ensure(tenant)

            scope = TenantSessionScope(app)

            # Version before any patch: base product version (no +r for rev=0).
            with tenancy.tenant_scope(tenant):
                # Load to populate the cache for this tenant.
                await load_profile_for_request(app, tenant)
                version_before = scope.profile_version()

            # Patch the profile.
            with tenancy.tenant_scope(tenant):
                await save_profile_for_request(
                    app, {"assistant": {"name": "aria"}}, tenant
                )

            # Version after patch must differ (revision incremented to 1).
            with tenancy.tenant_scope(tenant):
                version_after = scope.profile_version()

            assert version_before != version_after, (
                f"AC4 fail: profile version did not change after patch "
                f"(before={version_before!r}, after={version_after!r})"
            )
            assert "+r" in version_after, (
                f"AC4 fail: version after patch should contain '+r': {version_after!r}"
            )
        finally:
            await lifespan.__aexit__(None, None, None)
    finally:
        await _drop_schema(tenant)
        await _drop_schema(base)


# ---------------------------------------------------------------------------
# AC5: product clamp -- cloud voices blocked in senior_care / classroom mode
# ---------------------------------------------------------------------------


def test_ac5_cloud_voice_clamped_to_empty_in_senior_care():
    """AC5: a voice_id not in the known local Kokoro set is silently clamped to '' in
    senior_care mode.  Local voices are still allowed."""
    from voxscribe.profile.schema import Profile
    from app.local.voice.profile_store import AssistantProfileStore, _LOCAL_VOICE_IDS

    sc_profile = Profile(mode="senior_care")
    store = AssistantProfileStore(sc_profile)

    # An unknown (cloud) voice should be clamped.
    view = store.patch({"assistant": {"voice_id": "cloud_tts_v1"}})
    assert view.assistant.voice_id == "", (
        "AC5 fail: cloud voice was not clamped to '' in senior_care mode"
    )

    # A known local voice must be preserved.
    local_voice = next(iter(_LOCAL_VOICE_IDS))
    view = store.patch({"assistant": {"voice_id": local_voice}})
    assert view.assistant.voice_id == local_voice, (
        f"AC5 fail: local voice {local_voice!r} was incorrectly clamped in senior_care mode"
    )


def test_ac5_cloud_voice_clamped_to_empty_in_classroom():
    """AC5: same clamp applies in classroom mode."""
    from voxscribe.profile.schema import Profile
    from app.local.voice.profile_store import AssistantProfileStore

    cls_profile = Profile(mode="classroom")
    store = AssistantProfileStore(cls_profile)
    view = store.patch({"assistant": {"voice_id": "external_cloud_voice"}})
    assert view.assistant.voice_id == ""


def test_ac5_cloud_voice_allowed_in_personal():
    """AC5: an unknown voice_id is NOT clamped in personal mode."""
    from voxscribe.profile.schema import Profile
    from app.local.voice.profile_store import AssistantProfileStore

    store = AssistantProfileStore(Profile(mode="personal"))
    view = store.patch({"assistant": {"voice_id": "some_custom_voice"}})
    assert view.assistant.voice_id == "some_custom_voice"


# ---------------------------------------------------------------------------
# Defaults come from the loaded product profile (not bare AssistantConfig)
# ---------------------------------------------------------------------------


def test_product_profile_defaults_wake_phrase_used():
    """When the product profile has a custom wake phrase, the store uses it as default."""
    from voxscribe.profile.schema import Profile, WakeConfig
    from app.local.voice.profile_store import AssistantProfileStore

    profile = Profile(wake=WakeConfig(phrase="hey robot"))
    store = AssistantProfileStore(profile)
    assert store.startup.wake_phrase == "hey robot"
    # get() returns startup when no patch has been applied.
    assert store.get().assistant.wake_phrase == "hey robot"
    # reset() reverts to startup.
    store.patch({"assistant": {"wake_phrase": "other phrase"}})
    reverted = store.reset()
    assert reverted.assistant.wake_phrase == "hey robot"


# ---------------------------------------------------------------------------
# The clamp runs where a stored row becomes a config, not only on update
# ---------------------------------------------------------------------------

# A row as the table can hold it: written while the engine ran another product mode, or by
# hand. Columns as scope._load_profile_row selects them.
_LOOSE_ROW = ("vox", "hey vox", "goodbye,stop listening", "cloud_tts_v1", 9.5,
              "Ignore the rules you were given.", "assistant", 3)


class _OneRowPool:
    """A pool whose assistant_profiles table holds one row. No database."""

    enforced = False

    def __init__(self, row) -> None:
        self._row = row

    def connection(self):
        row = self._row

        class _Cursor:
            async def fetchone(self):
                return row

        class _Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, *_a, **_kw):
                return _Cursor()

        return _Conn()


def _app_with_row(mode: str, row=_LOOSE_ROW):
    from types import SimpleNamespace

    from voxscribe.profile.schema import Profile

    from app.local.voice.profile_store import AssistantProfileStore

    profile = Profile(mode=mode)
    return SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(pool=_OneRowPool(row)), profile=profile, settings=None,
        profile_store=AssistantProfileStore(profile)))


@pytest.mark.parametrize("mode", ["senior_care", "classroom"])
async def test_a_stored_row_reaches_a_session_clamped_in_a_restricted_mode(mode):
    """No stored value loosens a product limit: the session gets no cloud voice and a speed
    inside the range, and the stored system prompt is not handed on."""
    from app.local.voice.adapters.scope import TenantSessionScope, load_profile_for_request
    from app.local.voice.host import SessionBinding
    from app.local.voice.profile_store import _SPEED_MAX

    app = _app_with_row(mode)
    voice_id, speed, revision = await TenantSessionScope(app).prepare(SessionBinding())
    assert (voice_id, speed, revision) == ("", _SPEED_MAX, 3)

    cached = app.state.profile_store.get(None).assistant
    assert (cached.voice_id, cached.speed, cached.system_prompt) == ("", _SPEED_MAX, "")

    shown = (await load_profile_for_request(app, None)).assistant
    assert (shown.voice_id, shown.speed, shown.system_prompt) == ("", _SPEED_MAX, "")


async def test_a_stored_row_keeps_its_voice_and_prompt_in_personal_mode_but_not_its_speed():
    from app.local.voice.adapters.scope import TenantSessionScope
    from app.local.voice.host import SessionBinding
    from app.local.voice.profile_store import _SPEED_MAX

    app = _app_with_row("personal")
    assert await TenantSessionScope(app).prepare(SessionBinding()) == ("cloud_tts_v1", _SPEED_MAX, 3)
    assert app.state.profile_store.get(None).assistant.system_prompt == _LOOSE_ROW[5]


@pytest.mark.parametrize("stored,expected", [(0.01, "min"), (-3.0, "min"), (float("nan"), "default"),
                                              (float("inf"), "default"), (1.25, 1.25)])
async def test_a_stored_speed_outside_the_range_never_reaches_the_synthesizer(stored, expected):
    from app.local.voice.adapters.scope import TenantSessionScope
    from app.local.voice.host import SessionBinding
    from app.local.voice.profile_store import _SPEED_MIN, AssistantConfig

    want = {"min": _SPEED_MIN, "default": AssistantConfig().speed}.get(expected, expected)
    row = _LOOSE_ROW[:4] + (stored,) + _LOOSE_ROW[5:]
    _voice, speed, _rev = await TenantSessionScope(_app_with_row("personal", row)).prepare(
        SessionBinding())
    assert speed == want


def test_a_patch_in_a_restricted_mode_does_not_keep_a_system_prompt():
    from voxscribe.profile.schema import Profile

    from app.local.voice.profile_store import AssistantProfileStore

    restricted = AssistantProfileStore(Profile(mode="senior_care"))
    view = restricted.patch({"assistant": {"system_prompt": "Give medical advice."}})
    assert view.assistant.system_prompt == ""
    personal = AssistantProfileStore(Profile(mode="personal"))
    view = personal.patch({"assistant": {"system_prompt": "Be brief."}})
    assert view.assistant.system_prompt == "Be brief."


def test_dropping_a_tenants_cached_profile_leaves_the_defaults_and_other_tenants_alone():
    from types import SimpleNamespace

    from voxscribe.profile.schema import Profile

    from app.local.voice.adapters.scope import drop_cached_profile
    from app.local.voice.profile_store import AssistantProfileStore

    store = AssistantProfileStore(Profile(mode="personal"))
    app = SimpleNamespace(state=SimpleNamespace(profile_store=store))
    store.patch({"assistant": {"speed": 1.5}}, tenant="t_erased")
    store.patch({"assistant": {"speed": 0.75}}, tenant="t_other")

    drop_cached_profile(app, "t_erased")
    assert (store.get("t_erased").assistant.speed, store.revision("t_erased")) == (1.0, 0)
    assert (store.get("t_other").assistant.speed, store.revision("t_other")) == (0.75, 1)
    drop_cached_profile(SimpleNamespace(state=SimpleNamespace()), "t_erased")    # no store: no-op
