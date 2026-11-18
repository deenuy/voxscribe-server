"""Event sink for the voice host: the app event bus, broadcast.

The host runs outside any tenant scope, so a scoped event would reach no one. Typed, sequenced
session events replace this in T09.
"""
from __future__ import annotations


def publish_broadcast(kind: str, data: dict) -> None:
    try:
        from app.local import events

        events.publish(kind, data, broadcast=True)
    except Exception:
        pass
