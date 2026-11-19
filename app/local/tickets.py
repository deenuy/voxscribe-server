"""Short-lived, single-use tickets for the addresses a browser cannot send headers to.

A WebSocket and an EventSource cannot carry an Authorization or X-Identity-Token header, so their credential has
to travel in the address, and addresses end up in logs, proxies and history. The long-lived engine token and the
account's access token must not. Instead the client asks for a ticket over an ordinary authenticated request
(POST /v1/tickets, headers only) and presents it once as `?ticket=` on the socket or the stream.

A ticket is:
  short-lived   it expires ticket_ttl_s after it is issued (VOXSCRIBE_TICKET_TTL_S, Config.ticket_ttl_s)
  single use    presenting it consumes it, whether or not it is accepted
  bound         to the exact path it was issued for, and to the verified person (tenant and claims) who asked
                for it; presenting it elsewhere fails and burns it
  in memory     one engine process; a restart forgets every ticket, which only means asking for a new one

Only a digest of each ticket is held. A presented ticket is found by its SHA-256 digest, so no comparison ever
runs over the secret itself and timing reveals nothing usable about it. Ticket values are never logged.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.local.errors import EngineError
from app.local.identity import Identity

# The routes a ticket opens: the event stream and a meeting's live socket. Nothing else accepts one.
_TICKETABLE = re.compile(r"^/v1/events$|^/v1/meetings/[A-Za-z0-9._~-]{1,128}/live$")

# Outstanding tickets one tenant may hold at a time, so no caller can grow the book without bound.
MAX_OUTSTANDING_PER_TENANT = 64


def ticketable(path: str) -> bool:
    return bool(_TICKETABLE.match(path))


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


@dataclass(frozen=True)
class Grant:
    """What a redeemed ticket stands for: who asked for it, and where it may be used."""

    path: str
    tenant: str | None          # None only when tenant isolation is off
    identity: Identity | None   # None only when tenant isolation is off
    expires_at: float


class TooManyTickets(EngineError):
    def __init__(self):
        super().__init__("too_many_tickets", "too many unused tickets: use or let some expire first", 429)


class TicketBook:
    def __init__(self, ttl_s: int, *, clock: Callable[[], float] = time.monotonic):
        if ttl_s <= 0:
            raise ValueError("ticket lifetime must be positive")
        self.ttl_s = ttl_s
        self._clock = clock
        self._grants: dict[bytes, Grant] = {}

    def __len__(self) -> int:
        return len(self._grants)

    def _sweep(self, now: float) -> None:
        for key in [k for k, g in self._grants.items() if g.expires_at <= now]:
            del self._grants[key]

    def issue(self, path: str, *, tenant: str | None, identity: Identity | None) -> str:
        if not ticketable(path):
            raise ValueError("tickets are issued only for the live socket and the event stream")
        now = self._clock()
        self._sweep(now)
        if sum(1 for g in self._grants.values() if g.tenant == tenant) >= MAX_OUTSTANDING_PER_TENANT:
            raise TooManyTickets()
        value = secrets.token_urlsafe(32)
        self._grants[_digest(value)] = Grant(path=path, tenant=tenant, identity=identity,
                                             expires_at=now + self.ttl_s)
        return value

    def redeem(self, value: str, path: str) -> Grant | None:
        """The grant for a presented ticket, consumed. None when it is unknown, already used, expired, or was
        issued for another path; a ticket presented at the wrong path is consumed too."""
        if not value:
            return None
        grant = self._grants.pop(_digest(value), None)
        if grant is None or grant.path != path or self._clock() >= grant.expires_at:
            return None
        return grant
