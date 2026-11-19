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
  profile_version()    a fingerprint of the product profile, recorded on the session row

With isolation off (no identity configured) there is no tenant: nothing is entered and nothing
is refused, as before.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager

from app.local import tenancy
from app.local.voice.host import (
    SessionBinding,
    SessionScope,
    SessionTenantRequiredError,
    current_session,
)

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
    """Which profile a session ran under: the product mode and a fingerprint of the profile's
    content, so two sessions under the same settings compare equal and any change shows."""
    if profile is None:
        return ""
    content = dataclasses.asdict(profile) if dataclasses.is_dataclass(profile) else vars(profile)
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode("utf-8"))
    return f"{getattr(profile, 'mode', 'unknown')}@{digest.hexdigest()[:_FINGERPRINT_CHARS]}"


class TenantSessionScope(SessionScope):
    def __init__(self, app) -> None:
        self._app = app

    @property
    def tenant_required(self) -> bool:
        return isolation_enforced(self._app)

    def profile_version(self) -> str:
        return profile_version(getattr(self._app.state, "profile", None))

    def device_binding(self, tenant: str | None = None) -> SessionBinding:
        """The binding for the engine's own audio devices: the loaded profile's product and
        this engine's device id. The tenant is whoever binds the device, not known at startup."""
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
