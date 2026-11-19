"""Event sink for the voice host: the app event bus, broadcast.

The host publishes a session's events inside the session's scope, so the tenant they belong to
is known here (tenancy.current_tenant(), and the session itself through
app.local.voice.host.current_session()). Delivery is still broadcast: scoping, typing and
sequencing these events is PL06. Not every event has a session: arm and disarm publish
wake_state from the caller's request, and capture_health comes from the capture watchdog.
"""
from __future__ import annotations


def publish_broadcast(kind: str, data: dict) -> None:
    try:
        from app.local import events

        events.publish(kind, data, broadcast=True)
    except Exception:
        pass
