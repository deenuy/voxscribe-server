"""Session scope over the engine's tenancy: where a voice session meets a tenant's schema.

The host knows a session has a binding and that a scope is entered for it; it does not know
what a tenant is. This adapter does, and it is the only voice module that does:

  TenantSessionScope   what the host enters. While armed and for the length of a session it
                       holds the binding's tenant in scope (tenancy.tenant_scope), so the pool
                       selects that tenant's schema on every borrow and the tasks the session
                       starts inherit it. With isolation on and no tenant it refuses, by name
  scoped_pool()        the one way voice code reaches the engine's connection pool. With
                       isolation on it hands the pool out only inside a tenant's scope, and
                       only the session's own tenant when a session is in scope
  profile_version()    a fingerprint of the product profile combined with the tenant's row
                       revision, recorded on the session row

Also provides DB helpers for the assistant profile table (migration 022):
  load_profile_for_request()   router GET -- load from DB, populate store cache
  save_profile_for_request()   router PATCH -- validate, write to DB, return updated view
  reset_profile_for_request()  router POST reset -- delete from DB, clear cache

With isolation off (no identity configured) there is no tenant: nothing is entered and nothing
is refused, as before.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from app.local import tenancy
from app.local.voice.host import (
    SessionBinding,
    SessionScope,
    SessionTenantRequiredError,
    current_session,
)

log = logging.getLogger("app.local.voice.adapters.scope")

LOCAL_DEVICE = "local"      # the engine's own audio devices, when no device id is configured
_FINGERPRINT_CHARS = 12


def _pool(app):
    return app.state.store.pool


def isolation_enforced(app) -> bool:
    store = getattr(app.state, "store", None)     # an app with no store has nothing to isolate
    return bool(getattr(getattr(store, "pool", None), "enforced", False))


def scoped_pool(app):
    """The engine's connection pool, for voice code. Call it at each use, not once.

    With isolation on, voice data is always some tenant's: outside a tenant's scope (none, or
    platform work) this raises instead of letting a write reach the shared schema, and inside a
    session it must be that session's tenant."""
    pool = _pool(app)
    if not getattr(pool, "enforced", False):
        return pool
    tenant = tenancy.current_tenant()
    if tenant is None:
        raise tenancy.TenantRequiredError("voice: no tenant in scope: refusing to touch the database")
    session = current_session()
    if session is not None and session.tenant != tenant:
        raise tenancy.TenantRequiredError("voice: the tenant in scope is not the session's")
    return pool


def profile_version(profile) -> str:
    """Which product profile a session ran under: mode + fingerprint of its content."""
    if profile is None:
        return ""
    content = dataclasses.asdict(profile) if dataclasses.is_dataclass(profile) else vars(profile)
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode("utf-8"))
    return f"{getattr(profile, 'mode', 'unknown')}@{digest.hexdigest()[:_FINGERPRINT_CHARS]}"


# ---- assistant profile DB helpers (migration 022) ----------------------------

_DEFAULT_END_PHRASES = "goodbye,stop listening"


def _end_phrases_to_db(phrases: tuple[str, ...]) -> str:
    return ",".join(phrases)


def _end_phrases_from_db(text: str) -> tuple[str, ...]:
    return tuple(p for p in text.split(",") if p)


async def _load_profile_row(conn):
    """Read the singleton row from assistant_profiles. Returns the row tuple or None."""
    cur = await conn.execute(
        "SELECT name, wake_phrase, end_phrases, voice_id, speed, "
        "system_prompt, persona_label, revision "
        "FROM assistant_profiles LIMIT 1"
    )
    return await cur.fetchone()


async def _upsert_profile_row(conn, cfg, revision: int) -> None:
    """Insert or update the singleton row."""
    await conn.execute(
        "INSERT INTO assistant_profiles "
        "(singleton, name, wake_phrase, end_phrases, voice_id, speed, "
        " system_prompt, persona_label, revision, updated_at) "
        "VALUES (TRUE, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (singleton) DO UPDATE SET "
        "  name = EXCLUDED.name, "
        "  wake_phrase = EXCLUDED.wake_phrase, "
        "  end_phrases = EXCLUDED.end_phrases, "
        "  voice_id = EXCLUDED.voice_id, "
        "  speed = EXCLUDED.speed, "
        "  system_prompt = EXCLUDED.system_prompt, "
        "  persona_label = EXCLUDED.persona_label, "
        "  revision = EXCLUDED.revision, "
        "  updated_at = EXCLUDED.updated_at",
        (cfg.name, cfg.wake_phrase, _end_phrases_to_db(cfg.end_phrases),
         cfg.voice_id, cfg.speed, cfg.system_prompt, cfg.persona_label,
         revision, datetime.now(UTC)),
    )


async def _delete_profile_row(conn) -> None:
    """Remove the singleton row (reset path)."""
    await conn.execute("DELETE FROM assistant_profiles")


def _row_to_config(row, startup, product_profile=None) -> "tuple[AssistantConfig, int]":
    """Convert a DB row to (AssistantConfig, revision). Falls back to startup defaults.

    This is where a stored row becomes a config, so the product clamp and the speed range run
    here: whatever the row holds (written under another product mode, or by hand), what a
    session or a response gets from it is inside the loaded product's limits."""
    from app.local.voice.profile_store import AssistantConfig, apply_product_clamp

    if row is None:
        return (apply_product_clamp(copy.deepcopy(startup), product_profile), 0)

    name, wake_phrase, end_phrases_str, voice_id, speed, system_prompt, persona_label, revision = row
    cfg = AssistantConfig(
        name=name,
        wake_phrase=wake_phrase,
        end_phrases=_end_phrases_from_db(end_phrases_str or _DEFAULT_END_PHRASES),
        voice_id=voice_id or "",
        speed=float(speed),
        system_prompt=system_prompt or "",
        persona_label=persona_label or "assistant",
    )
    return (apply_product_clamp(cfg, product_profile), int(revision))


async def load_profile_for_request(app, tenant: str | None):
    """Load the assistant profile from DB into the store cache; return the view.

    Called by the assistant router's GET handler. tenant is the caller's tenant key
    (or None with isolation off).  The DB table is in the tenant's schema (or shared).
    With no pool available (unit tests) falls back to the in-memory store.
    """
    from app.local.voice.profile_store import _ProfileView

    store = getattr(getattr(app, "state", None), "profile_store", None)
    if store is None:
        from app.local.voice.profile_store import AssistantConfig
        return _ProfileView(AssistantConfig())

    pool = getattr(getattr(getattr(app, "state", None), "store", None), "pool", None)
    if pool is None:
        return store.get(tenant)

    if tenant is not None:
        with tenancy.tenant_scope(tenant):
            async with pool.connection() as conn:
                row = await _load_profile_row(conn)
    else:
        async with pool.connection() as conn:
            row = await _load_profile_row(conn)

    cfg, revision = _row_to_config(row, store.startup, getattr(app.state, "profile", None))
    store.set_config(tenant, cfg, revision)
    return _ProfileView(copy.deepcopy(cfg))


async def save_profile_for_request(app, partial: dict, tenant: str | None):
    """Validate and persist a profile patch; update the cache; return the updated view.

    Product-profile limits: the row is clamped as it is loaded (_row_to_config) and the store's
    patch() applies the clamp again to the merged config before it is written.
    """
    from app.local.voice.profile_store import ProfileValidationError, _ProfileView

    store = getattr(getattr(app, "state", None), "profile_store", None)
    if store is None:
        raise RuntimeError("profile store not initialised")

    pool = getattr(getattr(getattr(app, "state", None), "store", None), "pool", None)
    if pool is None:
        return store.patch(partial, tenant=tenant)

    # Validate and apply patch in-memory first (raises ProfileValidationError on bad input).
    # We do this before touching the DB so an invalid request never writes.
    base_view = await load_profile_for_request(app, tenant)
    product_profile = getattr(getattr(app, "state", None), "profile", None)
    new_view = store.patch(partial, tenant=tenant,
                           product_profile=product_profile)
    new_cfg = new_view.assistant
    new_revision = store.revision(tenant)

    if tenant is not None:
        with tenancy.tenant_scope(tenant):
            async with pool.connection() as conn:
                await _upsert_profile_row(conn, new_cfg, new_revision)
    else:
        async with pool.connection() as conn:
            await _upsert_profile_row(conn, new_cfg, new_revision)

    return _ProfileView(copy.deepcopy(new_cfg))


async def reset_profile_for_request(app, tenant: str | None):
    """Delete the tenant's profile row; clear the cache; return the defaults view."""
    from app.local.voice.profile_store import _ProfileView

    store = getattr(getattr(app, "state", None), "profile_store", None)
    if store is None:
        from app.local.voice.profile_store import AssistantConfig
        return _ProfileView(AssistantConfig())

    pool = getattr(getattr(getattr(app, "state", None), "store", None), "pool", None)
    if pool is None:
        return store.reset(tenant=tenant)

    if tenant is not None:
        with tenancy.tenant_scope(tenant):
            async with pool.connection() as conn:
                await _delete_profile_row(conn)
    else:
        async with pool.connection() as conn:
            await _delete_profile_row(conn)

    store.reset(tenant=tenant)
    return _ProfileView(copy.deepcopy(store.startup))


def drop_cached_profile(app, tenant: str | None) -> None:
    """Forget the assistant profile cached for a tenant, for the erase path: the row is gone
    with the tenant's data, and the cache must not hand the old values to the next session.
    The next read loads the row again, or the product defaults. Touches no database."""
    store = getattr(getattr(app, "state", None), "profile_store", None)
    if store is not None:
        store.reset(tenant=tenant)


# ---- tenant session scope ----------------------------------------------------


class TenantSessionScope(SessionScope):
    def __init__(self, app) -> None:
        self._app = app

    @property
    def tenant_required(self) -> bool:
        return isolation_enforced(self._app)

    def profile_version(self) -> str:
        """The profile in force: product fingerprint plus the tenant's row revision.

        Sync and fast: the revision was loaded (or confirmed absent) during prepare().
        """
        base = profile_version(getattr(self._app.state, "profile", None))
        store = getattr(self._app.state, "profile_store", None)
        if store is None:
            return base
        tenant = tenancy.current_tenant()
        rev = store.revision(tenant)
        return f"{base}+r{rev}" if rev else base

    async def prepare(self, binding: SessionBinding) -> "tuple[str, float, int]":
        """Load the tenant's assistant profile from DB before the session opens.

        Returns (voice_id, speed, revision) so the host can stamp the session.
        Called once per session open, before the tenant scope is entered, so we
        enter it explicitly here.  With no pool available (unit tests) falls back
        to the in-memory store.
        """
        store = getattr(self._app.state, "profile_store", None)
        if store is None:
            return ("", 1.0, 0)

        pool = getattr(getattr(getattr(self._app, "state", None), "store", None), "pool", None)
        tenant = binding.tenant

        if pool is None:
            cfg = store.get(tenant).assistant
            return (cfg.voice_id, cfg.speed, store.revision(tenant))

        if tenant is not None:
            with tenancy.tenant_scope(tenant):
                async with pool.connection() as conn:
                    row = await _load_profile_row(conn)
        else:
            async with pool.connection() as conn:
                row = await _load_profile_row(conn)

        cfg, revision = _row_to_config(row, store.startup,
                                       getattr(self._app.state, "profile", None))
        store.set_config(tenant, cfg, revision)
        return (cfg.voice_id, cfg.speed, revision)

    def device_binding(self, tenant: str | None = None) -> SessionBinding:
        """The binding for the engine's own audio devices."""
        state = self._app.state
        return SessionBinding(
            tenant=tenant,
            product=getattr(getattr(state, "profile", None), "mode", "") or "",
            device=getattr(getattr(state, "settings", None), "device_id", "") or LOCAL_DEVICE,
        )

    @contextmanager
    def bound(self, binding: SessionBinding) -> Iterator[None]:
        if binding.tenant is None:
            if self.tenant_required:
                raise SessionTenantRequiredError(
                    "tenant isolation is on and the session has no tenant")
            yield                   # isolation off: the single shared scope, nothing to enter
            return
        with tenancy.tenant_scope(binding.tenant):
            yield
