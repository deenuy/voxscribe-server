"""Whose data is this code touching?

Every signed-in person is a tenant, and a tenant's rows live in a PostgreSQL schema of their own. Which schema a
piece of code works in is decided in exactly one place: the connection pool (`TenantPool`), from the tenant that is
in scope (a context variable) when a connection is borrowed. Nothing else chooses a schema, so a store or a router
that forgets about tenants still cannot read another tenant's rows: it only ever sees the tables of the tenant in scope.

Rules this module enforces:
- Fail closed. When tenant isolation is on and no tenant is in scope, borrowing a connection raises. There is no
  fallback to a shared schema.
- A tenant's schema is created and migrated the first time that tenant is seen, and is complete before it is used.
- A connection goes back to the pool with its schema restored, so the next borrower never inherits a tenant's.
- Work that runs outside a request (startup, background queues) states its tenant explicitly with `tenant_scope`,
  or says it is platform work with `system_scope` (no tenant data).

With isolation off (no identity configured) nothing changes: one shared schema, one person at a time, as before.
"""

import asyncio
import contextvars
import hashlib
import logging
from contextlib import contextmanager

import psycopg
from psycopg import sql

log = logging.getLogger(__name__)

_SYSTEM = "<system>"
_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("voxscribe_tenant", default=None)


class TenantRequiredError(RuntimeError):
    """A database connection was asked for with isolation on and no tenant in scope."""


def tenant_key(subject: str) -> str:
    """The schema name for a verified subject (the `sub` claim). Fixed length, lower case, safe as an identifier."""
    if not subject or not isinstance(subject, str):
        raise ValueError("a tenant needs a subject")
    return "t_" + hashlib.sha256(subject.encode("utf-8")).hexdigest()[:24]


def current() -> str | None:
    """The tenant key in scope, `<system>` for platform work, or None when nothing is in scope."""
    return _current.get()


def current_tenant() -> str | None:
    """The tenant key for scoping in-memory state (events, buses, tokens), or None for the single shared scope."""
    value = _current.get()
    return None if value == _SYSTEM else value


@contextmanager
def tenant_scope(key: str):
    """Everything inside works on this tenant's data (and tasks created inside inherit the scope)."""
    token = _current.set(key)
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def system_scope():
    """Platform work that touches no tenant's data: health checks, migrations of the shared schema."""
    token = _current.set(_SYSTEM)
    try:
        yield
    finally:
        _current.reset(token)


def search_path_option() -> str | None:
    """A PostgreSQL startup option that points a NEW connection at the tenant in scope, for code that opens its own
    connection instead of using the pool. None when there is no tenant in scope (single shared scope)."""
    key = _current.get()
    if key and key != _SYSTEM:
        return f"-c search_path={key},public"
    return None


class _Provisioner:
    """Creates and migrates a tenant's schema, once per process."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._ready: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    async def ensure(self, key: str) -> None:
        if key in self._ready:
            return
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._ready:
                return
            from .migrations import run_migrations

            async with await psycopg.AsyncConnection.connect(self.database_url, autocommit=True) as conn:
                await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(key)))
            await run_migrations(self.database_url, schema=key)
            self._ready.add(key)
            log.info("Tenant schema %s is ready", key)

    async def known_tenants(self) -> list[str]:
        """Every tenant schema that exists (for startup recovery of work a restart interrupted)."""
        async with await psycopg.AsyncConnection.connect(self.database_url, autocommit=True) as conn:
            cur = await conn.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name ~ '^t_[0-9a-f]{24}$' ORDER BY 1")
            return [r[0] for r in await cur.fetchall()]


class _Borrow:
    """`async with pool.connection() as conn`, with the tenant's schema set for the length of the block."""

    def __init__(self, owner: "TenantPool", kwargs: dict):
        self._owner = owner
        self._kwargs = kwargs
        self._cm = None
        self._conn = None
        self._key: str | None = None

    async def __aenter__(self):
        owner = self._owner
        scope = _current.get()
        if owner.enforced:
            if scope is None:
                raise TenantRequiredError("no tenant in scope: refusing to touch the database")
            self._key = None if scope == _SYSTEM else scope
        if self._key:
            await owner.provisioner.ensure(self._key)
        self._cm = owner.pool.connection(**self._kwargs)
        self._conn = await self._cm.__aenter__()
        if self._key:
            try:
                await self._conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self._key)))
                await self._conn.commit()
            except BaseException as exc:
                await self._discard()
                await self._cm.__aexit__(type(exc), exc, exc.__traceback__)
                raise
        return self._conn

    async def _discard(self) -> None:
        # A connection whose schema cannot be trusted is closed, so the pool drops it instead of lending it on.
        try:
            await self._conn.close()
        except Exception:  # noqa: BLE001 - closing a broken connection may itself fail
            pass

    async def __aexit__(self, exc_type, exc, tb):
        if self._key and not self._conn.closed:
            try:
                # Settle the caller's work the way the pool would (commit on success, roll back on error), then restore.
                await (self._conn.commit() if exc_type is None else self._conn.rollback())
                await self._owner.restore_schema(self._conn)
            except Exception:  # noqa: BLE001
                await self._discard()
        return await self._cm.__aexit__(exc_type, exc, tb)


class TenantPool:
    """The one object every store borrows connections from. Wraps the real pool; everything but `connection()` is
    passed straight through."""

    def __init__(self, pool, *, database_url: str, enforced: bool, base_schema: str | None = None):
        self.pool = pool
        self.enforced = enforced
        self.base_schema = base_schema
        self.database_url = database_url
        self.provisioner = _Provisioner(database_url)

    def connection(self, **kwargs):
        return _Borrow(self, kwargs)

    def __getattr__(self, name):
        return getattr(self.pool, name)

    async def restore_schema(self, conn) -> None:
        """Put a connection back on the schema it started with (called as it returns to the pool)."""
        if self.base_schema:
            await conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.base_schema)))
        else:
            await conn.execute("RESET search_path")
        await conn.commit()
