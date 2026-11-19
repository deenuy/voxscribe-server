"""SessionStore over Postgres: wake_events, meetings, segments, the reply chain's turn rows and
the gate records (app/local/gate_records.py).
Also the delegate lane's task rows (delegate_tasks), written by PostgresDelegateTaskStore.

A voice session is a meeting with origin "wake", created and stopped through the same
MeetingStore calls the mic button uses, then finalised like any other meeting. Segments are
upserted and published on meeting_bus as transcript.stable, as the live WebSocket path does.

Every connection is borrowed through scoped_pool() (adapters/scope.py): with tenant isolation on
these writes only happen inside the session's tenant scope, which the host enters. The session
row records who and what the session ran for: tenant, product, device and profile version, read
from the session in scope.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from voxscribe.session.ports import ReplyOutcome, TranscriptSegment, TurnRecord

from app.local import gate_records, reply_chain
from app.local.voice.adapters.scope import scoped_pool
from app.local.voice.barge_in import Interruption
from app.local.voice.delegate_lane import DelegateTask, TaskStatus
from app.local.voice.host import current_session

log = logging.getLogger("app.local.voice.store")

CONSENT_VERSION = "wake-v1"        # consent is captured by arming the wake word
WAKE_EVENT_PHRASE = "hey_vox"      # wake_events.phrase, as written before T02
SESSION_TITLE = "Wake session {:%H:%M}"


def _now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class PostgresSessionStore:
    def __init__(self, app) -> None:
        self._app = app

    @property
    def _pool(self):
        return scoped_pool(self._app)

    def _meetings(self):
        from app.local.store_meetings import MeetingStore

        return MeetingStore(self._pool)

    async def record_wake(self, *, score: float, accepted: bool, reject_reason: str | None,
                          detector: str | None, speech_ratio: float | None,
                          session_id: str | None = None) -> None:
        """One wake_events row per candidate, accepted or not. Never raises."""
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO wake_events (id, phrase, confidence, triggered_at, "
                    "started_meeting_id, accepted, reject_reason, detector, speech_ratio) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), WAKE_EVENT_PHRASE, float(score), _now_z(), session_id,
                     accepted, reject_reason, detector, speech_ratio),
                )
                await conn.commit()
        except Exception:
            log.exception("voice store: failed to record wake event accepted=%s", accepted)

    async def create_session(self, wake_score: float, *, detector: str | None = None,
                             speech_ratio: float | None = None) -> str:
        meeting_id = str(uuid.uuid4())
        store = self._meetings()
        await store.create_meeting(meeting_id=meeting_id,
                                   title=SESSION_TITLE.format(datetime.now(UTC)),
                                   origin="wake", client_request_id=f"wake-{meeting_id}")
        session = current_session()
        async with self._pool.connection() as conn:
            await conn.execute("UPDATE meetings SET consent_version=%s, consent_at=%s WHERE id=%s",
                               (CONSENT_VERSION, _now_z(), meeting_id))
            if session is not None:
                await conn.execute(
                    "UPDATE meetings SET tenant=%s, product=%s, device=%s, profile_version=%s"
                    " WHERE id=%s",
                    (session.tenant, session.product or None, session.device or None,
                     session.profile_version or None, meeting_id))
            await conn.commit()
        await self.record_wake(score=wake_score, accepted=True, reject_reason=None,
                               detector=detector, speech_ratio=speech_ratio,
                               session_id=meeting_id)
        await store.start_meeting(meeting_id)
        return meeting_id

    async def add_segment(self, session_id: str, segment: TranscriptSegment) -> None:
        from app.local import meeting_bus
        from app.local.models import Segment

        db = Segment(id=segment.id, startMs=segment.start_ms, endMs=segment.end_ms,
                     source=segment.source, speaker=segment.speaker, text=segment.text,
                     state=segment.state)
        try:
            await self._meetings().upsert_segment(session_id, db)
        except Exception:
            log.exception("voice store: failed to persist segment")
        await meeting_bus.publish(session_id, "transcript.stable", {
            "id": db.id, "startMs": db.startMs, "endMs": db.endMs, "source": db.source,
            "speaker": db.speaker, "text": db.text, "state": db.state,
        })

    async def begin_turn(self, session_id: str, turn_n: int, text: str) -> TurnRecord:
        model, mode = reply_chain.reply_settings(getattr(self._app.state, "settings", None))
        rec = await reply_chain.begin_reply(self._pool, session_id, turn_n, text,
                                            model=model, mode=mode)
        # PL09: the gate's heard record, with the policy version, before anything is played.
        # When it cannot be written, rec.heard_written stays False and the turn's gate decides
        # whether the reply may be spoken (TurnGate.for_turn).
        await gate_records.record_heard(self._pool, rec, self._gate_policy().stamp)
        return TurnRecord(session_id=session_id, turn_n=turn_n, turn_id=rec.turn_id, handle=rec)

    async def flush_gate_records(self, turn: TurnRecord, *, decision: str | None,
                                 category: str | None, ended: str) -> None:
        """The reply's gate journal and its outcome row, in their own transaction. This extends
        the SessionStore port in this server: the host calls it however the reply ended
        (`ended`: complete, interrupted, cancelled or failed). A failed write is logged by the
        ledger, not raised; outside the session's tenant scope the pool itself refuses."""
        await gate_records.flush_reply(self._pool, turn.handle, decision=decision,
                                       category=category, ended=ended)

    def _gate_policy(self):
        from app.local.voice.adapters.gate_policy import session_policy

        return session_policy(self._app)

    async def turn_receipts(self, session_id: str, turn_id: str | None) -> list[str]:
        """The receipts the gate's action-claim rule reads (PL43): the tools with an executed
        write receipt for this turn, or for the turn that confirmed them. This extends the
        SessionStore port in this server; a failed lookup raises and the gate fails closed."""
        from app.local.tools import audit

        return await audit.turn_receipts(self._pool, session_id, turn_id)

    async def session_tools(self, session_id: str) -> list[str]:
        """The port's session-wide lookup: every executed write receipt in the session. The
        host's gate does not read it (a receipt from another turn backs nothing)."""
        from app.local.tools import audit

        return await audit.session_receipts(self._pool, session_id)

    async def add_turn(self, turn: TurnRecord, outcome: ReplyOutcome, *, turn_end_ms: int,
                       asr_final_ms: int | None = None,
                       interruption: Interruption | None = None) -> None:
        """interruption: set when a barge-in cut the reply short (T13); outcome.text is then
        what was heard, and the heard and full texts are written beside it."""
        await reply_chain.finish_reply(
            self._pool, turn.handle, reply_text=outcome.text, decision=outcome.decision,
            category=outcome.category, turn_end_ms=turn_end_ms, asr_final_ms=asr_final_ms,
            llm_request_sent_ms=outcome.started_ms, llm_first_token_ms=outcome.first_sentence_ms,
            tts_first_audio_ms=outcome.first_audio_ms, playback_end_ms=outcome.playback_end_ms,
            played_text=interruption.played_text if interruption else None,
            full_text=interruption.full_text if interruption else None)

    async def discard_session(self, session_id: str) -> None:
        """Silent close: the session heard no genuine speech in turn 1, so it was never a real
        meeting. Its wake_events row is kept for audit with the link cleared; the meeting row
        and everything hanging off it (segments) are deleted. Never raises."""
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    "UPDATE wake_events SET started_meeting_id=NULL WHERE started_meeting_id=%s",
                    (session_id,))
                await conn.execute("DELETE FROM meetings WHERE id=%s", (session_id,))
                await conn.commit()
            log.info("voice store: silent close, meeting deleted meeting_id=%s", session_id)
        except Exception:
            log.exception("voice store: silent close cleanup failed meeting_id=%s", session_id)

    async def close_session(self, session_id: str) -> None:
        """stop_once, then finalise exactly as the WebSocket path does."""
        from app.local import meeting_bus
        from app.local.finalize import schedule_finalize

        try:
            _meeting, stopped_now = await self._meetings().stop_once(session_id)
        except Exception:
            log.exception("voice store: stop_once failed meeting_id=%s", session_id)
            stopped_now = False
        if stopped_now:
            await meeting_bus.publish(session_id, "status", {"state": "processing"})
            schedule_finalize(self._app, session_id)
            log.info("voice store: finalize scheduled meeting_id=%s", session_id)
        else:
            log.info("voice store: session already stopped meeting_id=%s", session_id)


class PostgresDelegateTaskStore:
    """One delegate_tasks row per lane task, rewritten on every state change. Never raises."""

    def __init__(self, app) -> None:
        self._app = app
        self._order = asyncio.Lock()     # writes land in the order the changes happened

    async def save(self, task: DelegateTask) -> None:
        delivered = task.status is TaskStatus.DELIVERED
        async with self._order:
            try:
                async with scoped_pool(self._app).connection() as conn:
                    await conn.execute(
                        "INSERT INTO delegate_tasks"
                        " (id, session_id, request, status, result, delivered_at, task_id)"
                        " VALUES (%s, %s, %s, %s, %s, CASE WHEN %s THEN now() END, %s)"
                        " ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status,"
                        " result = EXCLUDED.result, delivered_at ="
                        " COALESCE(delegate_tasks.delivered_at, EXCLUDED.delivered_at)",
                        (task.task_id, task.session_id, task.request, task.status.value,
                         task.result, delivered, task.task_id),
                    )
                    await conn.commit()
            except Exception:
                log.exception("voice store: failed to write delegate task=%s status=%s",
                              task.task_id, task.status.value)
