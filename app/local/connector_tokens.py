"""In-memory token storage for Google connector.

Access tokens are kept in memory only and never persisted to database.
The engine receives short-lived access tokens from the Electron main process.
When a token expires, the engine publishes connector.token_needed event.
"""

import asyncio
from datetime import datetime

from . import events, tenancy

# In-memory token storage: {(tenant, connector): (token, expires_at)}. A person's Google token is theirs alone.
_tokens: dict[tuple[str | None, str], tuple[str, datetime]] = {}


def _key() -> tuple[str | None, str]:
    return (tenancy.current_tenant(), "google")
_token_lock = asyncio.Lock()


async def set_google_token(token: str, expires_at: str) -> None:
    """Store a Google access token with its expiry time.

    Args:
        token: Short-lived access token
        expires_at: ISO 8601 UTC expiry time
    """
    async with _token_lock:
        # Parse expiry time
        try:
            if expires_at.endswith("Z"):
                expires_at = expires_at[:-1]
            exp = datetime.fromisoformat(expires_at)
        except (ValueError, AttributeError):
            exp = datetime.utcnow()

        _tokens[_key()] = (token, exp)


async def get_google_access_token() -> tuple[str, datetime] | None:
    """Get the stored Google access token if it exists and is not expired.

    Returns:
        (token, expiry_time) tuple or None if absent or expired
    """
    async with _token_lock:
        if _key() not in _tokens:
            return None

        token, expires_at = _tokens[_key()]
        now = datetime.utcnow()

        # Check if token is expired (with 5-minute buffer)
        if expires_at <= now:
            # Token is expired, remove it and request a new one
            del _tokens[_key()]
            events.publish("connector.token_needed", {"connector": "google"})
            return None

        return (token, expires_at)


async def clear_google_token() -> None:
    """Clear the stored Google access token."""
    async with _token_lock:
        _tokens.pop(_key(), None)


async def request_token() -> None:
    """Publish a request for the client to refresh and push a new token."""
    events.publish("connector.token_needed", {"connector": "google"})
