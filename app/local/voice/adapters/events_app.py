"""Event sink for the voice host: the app event bus, scoped to the tenant in scope.

The host publishes a session's events inside the session's scope, so they go to that session's
tenant only, and each carries the session's id and its next sequence number
(SessionContext.next_seq()). Events about the device rather than a session (wake_state,
capture_health) the host publishes in the scope of the tenant the device is bound to
(VoiceHost._publish_device), so they too reach that tenant only. Nothing here broadcasts.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def publish(kind: str, data: dict) -> None:
    try:
        from app.local import events
        from app.local.voice.host import current_session

        session = current_session()
        if session is None:
            events.publish(kind, data)
        else:
            events.publish(kind, data, session_id=session.session_id or None,
                           seq=session.next_seq())
    except Exception as exc:    # an event must never break the voice path; the payload is not logged
        log.warning("voice events: failed to publish kind=%s error=%s", kind, type(exc).__name__)
