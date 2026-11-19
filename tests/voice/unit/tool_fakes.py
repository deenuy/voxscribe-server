"""Fakes for the tool tests: a connection pool that keeps tool_calls and reminders rows in memory,
answers the stored privacy setting, and answers the gate's receipt lookup from what was written;
and an HTTP transport that counts every request that would leave the device."""
from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx

_INSERT_RE = re.compile(r"INSERT INTO (\w+)\s*\(([^)]*)\)", re.IGNORECASE)

SEARCH_BODY = {"results": [{"title": "Weather", "content": "Sunny, high of 20.",
                            "url": "http://example.test/weather"}]}


class _Cursor:
    def __init__(self, rows=(), rowcount: int = 0) -> None:
        self._rows = list(rows)
        self.rowcount = rowcount

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class ToolPool:
    """privacy: the stored privacyMode (None: no row). fail_reminders: a reminder insert raises."""

    def __init__(self, *, privacy: str | None = "connectors",
                 fail_reminders: bool = False) -> None:
        self.privacy = privacy
        self.fail_reminders = fail_reminders
        self.rows: dict[str, list[dict]] = {}
        self.receipt_lookups = 0
        self.settings_reads = 0

    def table(self, name: str) -> list[dict]:
        return self.rows.get(name, [])

    @property
    def tool_calls(self) -> list[dict]:
        return self.table("tool_calls")

    @asynccontextmanager
    async def connection(self):
        yield self

    async def commit(self) -> None:
        pass

    async def execute(self, sql: str, params=()):
        params = tuple(params)
        m = _INSERT_RE.match(sql.strip())
        if m:
            table = m.group(1)
            if table == "reminders" and self.fail_reminders:
                raise RuntimeError("reminder store down")
            cols = [c.strip() for c in m.group(2).split(",")]
            strict = table in ("tool_calls", "reminders")
            self.rows.setdefault(table, []).append(dict(zip(cols, params, strict=strict)))
            return _Cursor(rowcount=1)
        if "FROM settings" in sql:
            self.settings_reads += 1
            if self.privacy is None:
                return _Cursor()
            return _Cursor([(json.dumps({"value": self.privacy}),)])
        if "FROM tool_calls" in sql:
            self.receipt_lookups += 1
            return _Cursor(self._receipts(sql, params))
        return _Cursor()

    def _receipts(self, sql: str, params: tuple) -> list[tuple]:
        """The gate's lookup: executed write receipts of this session for one turn, matched on
        the turn or the turn that confirmed it."""
        session_id, turn_id = params[0], params[1]
        names = []
        for row in self.tool_calls:
            if row.get("meeting_id") != session_id or row.get("status") != "executed":
                continue
            if row.get("side_effect") not in ("write", "irreversible"):
                continue
            if turn_id not in (row.get("turn_id"), row.get("confirm_turn_id")):
                continue
            if row["tool"] not in names:
                names.append(row["tool"])
        return [(n,) for n in names]


class CountingTransport(httpx.AsyncBaseTransport):
    """Every request that would have left the device, answered with a fixed search result."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=SEARCH_BODY)


def settings(**over):
    base = dict(vox_timezone="UTC", vox_location="", vox_user_name="", voice_mode="default",
                voice_reply_model="fake-talker", privacy_mode="offline")
    base.update(over)
    return SimpleNamespace(**base)


def make_app(pool: ToolPool, *, profile=None, **settings_over):
    return SimpleNamespace(state=SimpleNamespace(
        settings=settings(**settings_over), store=SimpleNamespace(pool=pool), profile=profile))


def load_profile(mode: str):
    """A shipped product profile from the pinned wheel: personal, senior_care, classroom."""
    from voxscribe.profile.loader import load_profile as _load
    from voxscribe.resources import resolve

    try:
        path = resolve(f"profile/profiles/{mode}.yaml")
    except Exception:  # noqa: BLE001 - the profiles ship beside the loader
        from pathlib import Path

        import voxscribe.profile as vp

        path = Path(vp.__file__).parent / "profiles" / f"{mode}.yaml"
    return _load(str(path))
