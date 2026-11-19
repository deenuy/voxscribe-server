"""Retention sweep: delete old wake candidate rows and tool-call records per tenant.

Retention is off by default (days = 0 means keep forever). When a positive number is
configured, the sweep deletes rows older than that many days. The sweep runs once at
startup and then on a fixed 24-hour interval using a plain asyncio loop so no scheduler
dependency is needed.

Usage (in app.py lifespan):
    from app.local.retention import start_retention_loop
    task = start_retention_loop(app)          # starts the loop, returns the task
    # on shutdown:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

The sweep can also be called directly:
    from app.local.retention import run_sweep_once
    await run_sweep_once(pool, wake_days=7, tool_call_days=30)
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.local.tenancy import TenantPool

log = logging.getLogger(__name__)

_SWEEP_INTERVAL_S = 24 * 60 * 60  # 24 hours


async def sweep_tenant(conn, wake_days: int, tool_call_days: int) -> dict[str, int]:
    """Delete rows older than the configured retention inside the connection's current schema.

    Returns a dict of {table: rows_deleted} for logging.
    """
    deleted: dict[str, int] = {}

    if wake_days > 0:
        cur = await conn.execute(
            "DELETE FROM wake_events WHERE triggered_at < NOW() - (%s || ' days')::interval",
            (str(wake_days),),
        )
        deleted["wake_events"] = cur.rowcount

    if tool_call_days > 0:
        # called_at is TEXT (migration 014); cast to timestamptz for comparison
        cur = await conn.execute(
            "DELETE FROM tool_calls WHERE called_at::timestamptz < NOW() - (%s || ' days')::interval",
            (str(tool_call_days),),
        )
        deleted["tool_calls"] = cur.rowcount

    return deleted


async def run_sweep_once(pool: "TenantPool", wake_days: int, tool_call_days: int) -> None:
    """Run the retention sweep across every tenant schema (or the shared schema when isolation is off)."""
    if wake_days == 0 and tool_call_days == 0:
        return

    from app.local import tenancy

    if pool.enforced:
        # Isolation on: sweep each known tenant schema in its own scope.
        for key in await pool.provisioner.known_tenants():
            with tenancy.tenant_scope(key):
                try:
                    async with pool.connection() as conn:
                        deleted = await sweep_tenant(conn, wake_days, tool_call_days)
                    if any(deleted.values()):
                        log.info("retention sweep tenant=%s deleted=%s", key, deleted)
                except Exception:
                    log.warning("retention sweep failed for tenant=%s", key, exc_info=True)
    else:
        # Isolation off: one shared schema.
        with tenancy.system_scope():
            try:
                async with pool.connection() as conn:
                    deleted = await sweep_tenant(conn, wake_days, tool_call_days)
                if any(deleted.values()):
                    log.info("retention sweep (shared) deleted=%s", deleted)
            except Exception:
                log.warning("retention sweep failed (shared schema)", exc_info=True)


def start_retention_loop(app) -> asyncio.Task:
    """Start a background task that runs the sweep at startup and every 24 hours.

    Returns the task so the caller can cancel it on shutdown.
    """
    async def _loop():
        while True:
            settings = app.state.settings
            pool = app.state.store.pool
            try:
                await run_sweep_once(
                    pool,
                    wake_days=settings.wake_candidate_retention_days,
                    tool_call_days=settings.tool_call_retention_days,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("retention loop: unexpected error in sweep", exc_info=True)
            await asyncio.sleep(_SWEEP_INTERVAL_S)

    return asyncio.create_task(_loop(), name="retention-sweep")
