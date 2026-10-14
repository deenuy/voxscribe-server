"""Dedicated storage for meetings, notes, transcripts, and summaries."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .timefmt import iso_z
from .errors import ConflictError, EngineError, InvalidRequestError, NotFoundError
from .models import Block, Citation, EnhancedBlock, Meeting, Segment


# One lock number serialises starting a recording, so "only one meeting records at a time" holds under concurrency.
_RECORDING_LOCK = 7_283_117_002

MAX_SEGMENT_CHARS = 20_000  # a single spoken turn is a paragraph, never a document
MAX_SPEAKER_CHARS = 120

# Words in a text column, the same way everywhere (stored counts and recomputed counts must agree).
_WORDS_SQL = "COALESCE(sum(cardinality(regexp_split_to_array(btrim(text), '\\s+'))) FILTER (WHERE btrim(text) <> ''), 0)"


_UNSET = object()  # "not sent", as distinct from None ("clear it")


class MeetingStore:
    """PostgreSQL storage for meetings and related data."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    # ========================================================================
    # Meetings CRUD
    # ========================================================================

    async def create_meeting(
        self,
        meeting_id: str,
        title: str = "New meeting",
        context: str | None = None,
        calendar_event_id: str | None = None,
        scheduled_start: str | None = None,
        scheduled_end: str | None = None,
        project_id: str | None = None,
        origin: str = "recorded",
        client_request_id: str | None = None,
    ) -> Meeting:
        """Create a new meeting with status 'scheduled'."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """INSERT INTO meetings
                    (id, title, context, calendar_event_id, scheduled_start, scheduled_end, status,
                     project_id, origin, client_request_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (meeting_id, title, context, calendar_event_id, scheduled_start, scheduled_end, "scheduled",
                     project_id, origin, client_request_id, now, now),
                )
                await conn.commit()
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
        return await self.get_meeting(meeting_id)

    async def find_by_client_request_id(self, client_request_id: str) -> "Meeting | None":
        """Return an existing meeting created with this clientRequestId, or None."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                "SELECT * FROM meetings WHERE client_request_id = %s", (client_request_id,)
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return await self._row_to_meeting(row)

    async def get_meeting(self, meeting_id: str) -> Meeting:
        """Get a meeting by ID with computed fields."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                "SELECT * FROM meetings WHERE id = %s", (meeting_id,)
            )
            row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"meeting {meeting_id} not found")
        return await self._row_to_meeting(row)

    async def list_meetings(
        self,
        query: str | None = None,
        when: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[Meeting], str | None]:
        """List meetings with filtering and cursor pagination."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            sql_query = "SELECT * FROM meetings WHERE 1=1"
            params: list[Any] = []

            if query:
                sql_query += " AND (title ILIKE %s OR context ILIKE %s)"
                params.extend([f"%{query}%", f"%{query}%"])

            if when == "upcoming":
                sql_query += " AND status IN ('scheduled', 'recording', 'processing')"
                sql_query += " ORDER BY COALESCE(scheduled_start, created_at) ASC"
            elif when == "past":
                sql_query += " AND status = 'done'"
                sql_query += " ORDER BY ended_at DESC NULLS LAST"
            else:
                sql_query += " ORDER BY started_at DESC NULLS LAST, created_at DESC"

            sql_query += " LIMIT %s"
            params.append(limit + 1)

            cursor_obj = await conn.execute(sql_query, params)
            rows = await cursor_obj.fetchall()

        meetings = [await self._row_to_meeting(row) for row in rows[:limit]]
        next_cursor = str(rows[limit]["id"]) if len(rows) > limit else None
        return meetings, next_cursor

    async def update_meeting(
        self,
        meeting_id: str,
        title: str | None = None,
        context: str | None = None,
        reminder_enabled: bool | None = None,
        enhance_style: str | None = None,
        auto_stop_minutes: int | None = None,
        project_id=_UNSET,
        base_revision: int | None = None,
    ) -> Meeting:
        """Update meeting fields (only the allowed ones from PATCH). `project_id` None takes it out of its project; left
        out, the project is not touched."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        updates = {"updated_at": now}
        if title is not None:
            updates["title"] = title
        if context is not None:
            updates["context"] = context
        if reminder_enabled is not None:
            updates["reminder_enabled"] = reminder_enabled
        if enhance_style is not None:
            updates["enhance_style"] = enhance_style
        if auto_stop_minutes is not None:
            updates["auto_stop_minutes"] = auto_stop_minutes
        if project_id is not _UNSET:
            updates["project_id"] = project_id

        cols = [k for k in updates.keys()]
        vals = [updates[k] for k in cols]
        vals.append(meeting_id)

        set_clause = ", ".join(f"{k} = %s" for k in cols) + ", revision = revision + 1"
        where = "id = %s"
        if base_revision is not None:
            where += " AND revision = %s"
            vals.append(base_revision)
        try:
            async with self.pool.connection() as conn:
                cur = await conn.execute(
                    f"UPDATE meetings SET {set_clause} WHERE {where} RETURNING revision",
                    vals,
                )
                changed = await cur.fetchone()
                if not changed and base_revision is not None:
                    row = await (await conn.execute("SELECT revision FROM meetings WHERE id = %s", (meeting_id,))).fetchone()
                    if row:
                        raise EngineError(
                            "revision_conflict",
                            f"this scribe changed (now revision {row[0]}); reload and try again",
                            409,
                            {"items": [{"kind": "meeting", "id": meeting_id, "currentRevision": row[0]}]},
                        )
                await conn.commit()
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
        return await self.get_meeting(meeting_id)

    async def start_meeting(self, meeting_id: str) -> Meeting:
        """Start recording, atomically: at most one meeting records at a time, and only a scheduled meeting can start.

        Check and change happen in one transaction under an advisory lock, so two simultaneous starts cannot both
        succeed. A repeat start of the same meeting, or a start of one that already finished, is a 409."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_RECORDING_LOCK,))
                other = await (await conn.execute(
                    "SELECT 1 FROM meetings WHERE status = 'recording' AND id <> %s LIMIT 1", (meeting_id,)
                )).fetchone()
                if other:
                    raise ConflictError("another meeting is already recording")
                updated = await (await conn.execute(
                    "UPDATE meetings SET status = 'recording', started_at = %s, updated_at = %s "
                    "WHERE id = %s AND status = 'scheduled' RETURNING id",
                    (now, now, meeting_id),
                )).fetchone()
                if not updated:
                    row = await (await conn.execute("SELECT status FROM meetings WHERE id = %s", (meeting_id,))).fetchone()
                    if not row:
                        raise NotFoundError(f"meeting {meeting_id} not found")
                    if row[0] == "recording":
                        raise ConflictError("this meeting is already recording")
                    raise ConflictError(f"this meeting is {row[0]} and cannot be started")
        return await self.get_meeting(meeting_id)

    async def stop_meeting(self, meeting_id: str) -> Meeting:
        """Stop recording. Safe to call twice: see `stop_once`."""
        meeting, _ = await self.stop_once(meeting_id)
        return meeting

    async def stop_once(self, meeting_id: str) -> tuple[Meeting, bool]:
        """Move a recording meeting to 'processing' and report whether this call did it.

        The change is conditional on the meeting still recording, so concurrent or repeated stops change nothing
        the second time (the end time is not overwritten, a finished meeting does not go back to processing) and
        only the call that really stopped it should start finalisation. A meeting that was never started is a
        409; one already stopping or done is returned as it is."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            changed = await (await conn.execute(
                "UPDATE meetings SET status = 'processing', ended_at = %s, updated_at = %s "
                "WHERE id = %s AND status = 'recording' RETURNING id",
                (now, now, meeting_id),
            )).fetchone()
            await conn.commit()
        meeting = await self.get_meeting(meeting_id)  # NotFoundError for an unknown id
        if not changed and meeting.status == "scheduled":
            raise ConflictError("this meeting is not recording")
        return meeting, bool(changed)

    async def mark_done(self, meeting_id: str) -> Meeting:
        """Publish the meeting: the transcript is final and may be edited from here on.

        Any segment the recogniser left provisional is settled (nothing more is coming), the word count is
        stored, and the publish time is set once."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                # Take the row lock first: the word count below is then read after any edit that was in flight
                # has committed (a statement's snapshot is taken when it starts, not when it gets the lock).
                await conn.execute("SELECT 1 FROM meetings WHERE id = %s FOR UPDATE", (meeting_id,))
                await conn.execute(
                    "UPDATE transcript_segments SET state = 'stable' WHERE meeting_id = %s AND state = 'provisional'",
                    (meeting_id,),
                )
                await conn.execute(
                    f"""UPDATE meetings SET status = 'done', updated_at = %s,
                        published_at = COALESCE(published_at, %s),
                        word_count = (SELECT {_WORDS_SQL} FROM transcript_segments WHERE meeting_id = %s)
                        WHERE id = %s""",
                    (now, now, meeting_id, meeting_id),
                )
        return await self.get_meeting(meeting_id)

    async def mark_interrupted(self, meeting_id: str) -> None:
        """Mark a recording-stuck meeting as interrupted.

        Called at server startup for any meeting left in the 'recording' state by a prior crash.
        The transcript (if any) is preserved and segments are settled.
        """
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE transcript_segments SET state = 'stable' WHERE meeting_id = %s AND state = 'provisional'",
                    (meeting_id,),
                )
                await conn.execute(
                    f"""UPDATE meetings SET status = 'interrupted', updated_at = %s,
                        word_count = (SELECT {_WORDS_SQL} FROM transcript_segments WHERE meeting_id = %s)
                        WHERE id = %s AND status = 'recording'""",
                    (now, meeting_id, meeting_id),
                )

    async def delete_meeting(self, meeting_id: str):
        """Delete a meeting (cascade handles notes, transcript, summary, attachments)."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM meetings WHERE id = %s", (meeting_id,))
            await conn.commit()

    async def _row_to_meeting(self, row: dict) -> Meeting:
        """Convert a database row to a Meeting model with computed fields."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            # Attachment count
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM meeting_attachments WHERE meeting_id = %s",
                (row["id"],),
            )
            attach_result = await cursor.fetchone()
            attach_count = attach_result.get("count", 0) if attach_result else 0

            # Transcript presence
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM transcript_segments WHERE meeting_id = %s",
                (row["id"],),
            )
            trans_result = await cursor.fetchone()
            segment_count = int(trans_result.get("count", 0)) if trans_result else 0
            has_transcript = segment_count > 0

            # Summary presence
            cursor = await conn.execute(
                "SELECT status FROM summaries WHERE meeting_id = %s",
                (row["id"],),
            )
            sum_result = await cursor.fetchone()
            has_summary = bool(sum_result and sum_result.get("status") == "ready")

        iso = lambda v: v.isoformat().replace("+00:00", "Z") if isinstance(v, datetime) else v  # noqa: E731
        duration_ms = None
        if isinstance(row["started_at"], datetime) and isinstance(row["ended_at"], datetime):
            duration_ms = max(0, int((row["ended_at"] - row["started_at"]).total_seconds() * 1000))

        return Meeting(
            id=row["id"],
            title=row["title"],
            context=row["context"],
            status=row["status"],
            scheduledStart=iso_z(row["scheduled_start"]),
            scheduledEnd=iso_z(row["scheduled_end"]),
            startedAt=iso_z(row["started_at"]),
            endedAt=iso_z(row["ended_at"]),
            calendarEventId=row["calendar_event_id"],
            reminderEnabled=bool(row["reminder_enabled"]),
            enhanceStyle=row["enhance_style"],
            autoStopMinutes=row["auto_stop_minutes"],
            attachmentCount=attach_count,
            hasTranscript=has_transcript,
            hasSummary=has_summary,
            projectId=row["project_id"],
            origin=row["origin"],
            language=row["language"],
            segmentCount=segment_count,
            wordCount=row["word_count"],
            durationMs=duration_ms,
            publishedAt=iso(row["published_at"]),
            transcriptRevision=row["transcript_revision"],
            summaryStale=bool(row["summary_stale"]),
            revision=row["revision"],
            createdAt=iso_z(row["created_at"]),
            updatedAt=iso_z(row["updated_at"]),
        )

    # ========================================================================
    # Transcript Segments
    # ========================================================================

    async def append_segment(self, meeting_id: str, segment: Segment) -> Segment:
        """Append or upsert a transcript segment."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO transcript_segments
                (id, meeting_id, start_ms, end_ms, source, speaker, text, state, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    text = EXCLUDED.text,
                    state = EXCLUDED.state
                WHERE transcript_segments.revision = 0""",  # a person's edit is never overwritten by a late recogniser write
                (
                    segment.id,
                    meeting_id,
                    segment.startMs,
                    segment.endMs,
                    segment.source,
                    segment.speaker,
                    segment.text,
                    segment.state,
                    now,
                ),
            )
            await conn.commit()
        return segment

    async def upsert_segment(self, meeting_id: str, segment: Segment) -> Segment:
        """Alias for append_segment."""
        return await self.append_segment(meeting_id, segment)

    async def import_lines(self, meeting_id: str, lines: list[dict]) -> int:
        """Add a whole transcript at once (an import): one connection, one transaction, every line stable.

        Used for scribes that were not recorded here, so the meeting goes straight from empty to having a transcript."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        rows = [(f"imp_{i:06d}_{uuid.uuid4().hex[:8]}", meeting_id, l["startMs"], l["endMs"], "mic", l.get("speaker"), l["text"], "stable", now) for i, l in enumerate(lines)]
        async with self.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """INSERT INTO transcript_segments (id, meeting_id, start_ms, end_ms, source, speaker, text, state, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        rows,
                    )
                await conn.execute("UPDATE meetings SET status = 'done', ended_at = %s, updated_at = %s WHERE id = %s", (now, now, meeting_id))
        return len(rows)

    async def get_transcript(self, meeting_id: str, offset: int = 0, limit: int | None = None) -> list[Segment]:
        """Transcript segments in spoken order. `offset` and `limit` page through a long meeting."""
        sql = ("SELECT * FROM transcript_segments WHERE meeting_id = %s "
               "ORDER BY start_ms ASC, source ASC, id ASC")
        params: list[Any] = [meeting_id]
        if limit is not None:
            sql += " LIMIT %s OFFSET %s"
            params += [limit, max(0, offset)]
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        return [self._row_to_segment(row) for row in rows]

    @staticmethod
    def _row_to_segment(row: dict) -> Segment:
        return Segment(
            id=row["id"],
            startMs=row["start_ms"],
            endMs=row["end_ms"],
            source=row["source"],
            speaker=row["speaker"],
            text=row["text"],
            state=row["state"],
            revision=row["revision"],
            edited=row["edited_at"] is not None,
        )

    async def transcript_state(self, meeting_id: str) -> tuple[bool, int]:
        """(editable, revision) for a meeting. Editable means published (status 'done')."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                "SELECT status, transcript_revision FROM meetings WHERE id = %s", (meeting_id,)
            )
            row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"meeting {meeting_id} not found")
        return row["status"] == "done", row["transcript_revision"]

    async def _apply_segment_edit(self, conn, meeting_id: str, seg: dict, new_text, speaker_given: bool, new_speaker, now: str) -> bool:
        """Write one segment edit inside the caller's transaction (the segment row is already locked and its revision
        already checked). Keeps the previous wording, bumps the transcript revision and marks the summary stale. A change
        that changes nothing writes nothing. Returns whether anything changed."""
        text_after = new_text if new_text is not None else seg["text"]
        speaker_after = new_speaker if speaker_given else seg["speaker"]
        if text_after == seg["text"] and speaker_after == seg["speaker"]:
            return False
        revision = seg["revision"] + 1
        await conn.execute(
            """UPDATE transcript_segments
               SET text = %s, speaker = %s, revision = %s, edited_at = %s, state = 'stable'
               WHERE id = %s""",
            (text_after, speaker_after, revision, now, seg["id"]),
        )
        await conn.execute(
            """INSERT INTO transcript_edits
               (meeting_id, segment_id, revision, before_text, after_text, before_speaker, after_speaker, edited_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (meeting_id, seg["id"], revision, seg["text"], text_after, seg["speaker"], speaker_after, now),
        )
        await conn.execute(
            f"""UPDATE meetings SET transcript_revision = transcript_revision + 1, summary_stale = true,
                updated_at = %s,
                word_count = (SELECT {_WORDS_SQL} FROM transcript_segments WHERE meeting_id = %s)
                WHERE id = %s""",
            (now, meeting_id, meeting_id),
        )
        return True

    async def edit_segment(
        self,
        meeting_id: str,
        segment_id: str,
        *,
        base_revision: int,
        text: str | None = None,
        speaker: str | None = None,
        speaker_given: bool = False,
    ) -> dict:
        """Edit one segment of a published transcript.

        One transaction, one meeting row lock then one segment row lock, so two editors cannot both win:
        the second sees a newer revision and gets `revision_conflict` with nothing written. The previous
        text and speaker go to transcript_edits, so the recogniser's output is never lost. A change that
        changes nothing writes nothing."""
        new_text = None
        if text is not None:
            new_text = text.strip()
            if not new_text:
                raise InvalidRequestError("text: a segment cannot be empty; ask to remove it instead")
            if len(new_text) > MAX_SEGMENT_CHARS:
                raise InvalidRequestError(f"text: at most {MAX_SEGMENT_CHARS} characters")
        new_speaker = None
        if speaker_given:
            new_speaker = (speaker or "").strip() or None
            if new_speaker and len(new_speaker) > MAX_SPEAKER_CHARS:
                raise InvalidRequestError(f"speaker: at most {MAX_SPEAKER_CHARS} characters")
        if new_text is None and not speaker_given:
            raise InvalidRequestError("nothing to change: send text and/or speaker")

        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT status, transcript_revision FROM meetings WHERE id = %s FOR UPDATE", (meeting_id,)
                )
                meeting = await cur.fetchone()
                if not meeting:
                    raise NotFoundError(f"meeting {meeting_id} not found")
                if meeting["status"] != "done":
                    raise EngineError(
                        "transcript_not_editable",
                        "the transcript can be edited once the meeting is stopped and published",
                        409,
                    )
                cur = await conn.execute(
                    "SELECT * FROM transcript_segments WHERE id = %s AND meeting_id = %s FOR UPDATE",
                    (segment_id, meeting_id),
                )
                seg = await cur.fetchone()
                if not seg:
                    raise NotFoundError(f"segment {segment_id} not found")
                if seg["revision"] != base_revision:
                    raise EngineError(
                        "revision_conflict",
                        f"this segment changed (now revision {seg['revision']}); reload and try again",
                        409,
                    )

                await self._apply_segment_edit(conn, meeting_id, seg, new_text, speaker_given, new_speaker, now)
                cur = await conn.execute("SELECT * FROM transcript_segments WHERE id = %s", (segment_id,))
                row = await cur.fetchone()
                cur = await conn.execute(
                    "SELECT transcript_revision, summary_stale, word_count FROM meetings WHERE id = %s", (meeting_id,)
                )
                m = await cur.fetchone()
        return {
            "segment": self._row_to_segment(row),
            "transcriptRevision": m["transcript_revision"],
            "summaryStale": bool(m["summary_stale"]),
            "wordCount": m["word_count"],
        }

    # ========================================================================
    # Notes (Shorthand Blocks)
    # ========================================================================

    async def put_notes(self, meeting_id: str, blocks: list[Block]) -> int:
        """Replace the shorthand notes for a meeting, return how many were accepted.

        Blocks are matched by id. Unchanged blocks are left alone, changed ones are updated and their enhanced
        note is marked pending, and only blocks that are no longer sent are deleted. Deleting everything and
        re-inserting would cascade to every enhanced note (including ones the person edited) on each save."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ids = [b.blockId for b in blocks]
        if len(set(ids)) != len(ids):
            raise InvalidRequestError("blocks: blockId values must be unique")
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.transaction():
                cursor = await conn.execute(
                    "SELECT block_id, text FROM notes_blocks WHERE meeting_id = %s FOR UPDATE", (meeting_id,)
                )
                existing_map = {row["block_id"]: row["text"] for row in await cursor.fetchall()}

                await conn.execute(
                    "DELETE FROM notes_blocks WHERE meeting_id = %s AND NOT (block_id = ANY(%s))", (meeting_id, ids)
                )
                for block in blocks:
                    cur = await conn.execute(
                        """INSERT INTO notes_blocks (block_id, meeting_id, text, offset_ms, created_at, updated_at)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT (block_id) DO UPDATE
                               SET text = EXCLUDED.text, offset_ms = EXCLUDED.offset_ms, updated_at = EXCLUDED.updated_at
                               WHERE notes_blocks.meeting_id = EXCLUDED.meeting_id
                           RETURNING block_id""",
                        (block.blockId, meeting_id, block.text, block.offsetMs, now, now),
                    )
                    if not await cur.fetchone():
                        raise InvalidRequestError("blocks: a blockId is already used by another meeting")
                    if block.blockId in existing_map and existing_map[block.blockId] != block.text:
                        await conn.execute(
                            """UPDATE enhanced_blocks SET status = 'pending', updated_at = %s
                               WHERE block_id = %s AND meeting_id = %s AND status != 'edited'""",
                            (now, block.blockId, meeting_id),
                        )
        return len(blocks)

    async def get_notes(self, meeting_id: str) -> list[Block]:
        """Get all notes for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                """SELECT block_id, text, offset_ms FROM notes_blocks
                WHERE meeting_id = %s ORDER BY offset_ms ASC""",
                (meeting_id,),
            )
            rows = await cursor.fetchall()
        return [Block(blockId=row["block_id"], text=row["text"], offsetMs=row["offset_ms"]) for row in rows]

    # ========================================================================
    # Enhanced Blocks
    # ========================================================================

    async def get_enhanced(self, meeting_id: str) -> dict:
        """Get enhanced blocks for a meeting with the enhance style."""
        meeting = await self.get_meeting(meeting_id)
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                """SELECT * FROM enhanced_blocks WHERE meeting_id = %s ORDER BY
                (SELECT offset_ms FROM notes_blocks WHERE block_id = enhanced_blocks.block_id) ASC""",
                (meeting_id,),
            )
            rows = await cursor.fetchall()
        return {
            "style": meeting.enhanceStyle,
            "blocks": [
                EnhancedBlock(
                    blockId=row["block_id"],
                    text=row["text"],
                    status=row["status"],
                    citations=[Citation(**c) for c in (row["citations"] or [])],
                    updatedAt=iso_z(row["updated_at"]),
                )
                for row in rows
            ],
        }

    async def set_enhanced_block(
        self,
        meeting_id: str,
        block_id: str,
        text: str,
        citations: list[Citation],
        status: str = "stable",
    ):
        """Create or update an enhanced block."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        citations_list = [c.model_dump(exclude_none=True) for c in citations]
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO enhanced_blocks
                (block_id, meeting_id, text, status, citations, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(block_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    status = EXCLUDED.status,
                    citations = EXCLUDED.citations,
                    updated_at = EXCLUDED.updated_at""",
                (block_id, meeting_id, text, status, json.dumps(citations_list), now, now),
            )
            await conn.commit()

    async def edit_enhanced_block(self, meeting_id: str, block_id: str, text: str):
        """Mark an enhanced block as edited (never overwritten by auto-enrichment)."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE enhanced_blocks SET text = %s, status = 'edited', updated_at = %s
                WHERE block_id = %s AND meeting_id = %s""",
                (text, now, block_id, meeting_id),
            )
            await conn.commit()

    async def get_non_edited_blocks(self, meeting_id: str) -> list[dict]:
        """Get all non-edited enhanced blocks for a meeting (for enrichment)."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                """SELECT eb.block_id, nb.text, nb.offset_ms, eb.status FROM enhanced_blocks eb
                JOIN notes_blocks nb ON eb.block_id = nb.block_id
                WHERE eb.meeting_id = %s AND eb.status != 'edited'
                ORDER BY nb.offset_ms ASC""",
                (meeting_id,),
            )
            rows = await cursor.fetchall()
        return rows or []

    # ========================================================================
    # Summaries
    # ========================================================================

    async def get_summary(self, meeting_id: str) -> dict:
        """Get summary for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                "SELECT * FROM summaries WHERE meeting_id = %s", (meeting_id,)
            )
            row = await cursor.fetchone()
        if row:
            generated_at = row["generated_at"]
            # Convert datetime to ISO string if needed
            if hasattr(generated_at, 'isoformat'):
                generated_at = generated_at.isoformat().replace("+00:00", "Z")
            return {
                "status": row["status"],
                "overview": row["overview"],
                "details": row["details"] or [],
                "generatedAt": generated_at,
                "revision": row["revision"],
                "edited": bool(row["edited"]),
                "promptVersion": row.get("prompt_version"),
            }
        return {"status": "none", "overview": None, "details": [], "generatedAt": None, "revision": 0, "edited": False, "promptVersion": None}

    async def set_summary_pending(self, meeting_id: str):
        """Mark summary as pending (during processing)."""
        await self.set_summary_status(meeting_id, "pending")

    async def set_summary_status(self, meeting_id: str, status: str):
        """Set the summary status without changing the content.

        Inserts a row if none exists. Used to advance from pending through
        running to ready/skipped/unavailable/failed.
        """
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO summaries
                (meeting_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at""",
                (meeting_id, status, now, now),
            )
            await conn.commit()

    async def set_summary(
        self,
        meeting_id: str,
        overview: str,
        details: list[dict],
        generated_at: str,
        prompt_version: str | None = None,
    ):
        """Set summary for a meeting as ready."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO summaries
                (meeting_id, status, overview, details, generated_at, prompt_version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    status = 'ready',
                    overview = EXCLUDED.overview,
                    details = EXCLUDED.details,
                    generated_at = EXCLUDED.generated_at,
                    prompt_version = EXCLUDED.prompt_version,
                    revision = summaries.revision + 1,
                    edited = false,
                    edited_at = NULL,
                    updated_at = EXCLUDED.updated_at""",
                (meeting_id, "ready", overview, json.dumps(details), generated_at, prompt_version, now, now),
            )
            await conn.execute("UPDATE meetings SET summary_stale = false WHERE id = %s", (meeting_id,))
            await conn.commit()

    # ========================================================================
    # Attachments
    # ========================================================================

    async def attach_source(self, meeting_id: str, source_id: str):
        """Attach a source to a meeting."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO meeting_attachments (meeting_id, source_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (meeting_id, source_id),
            )
            await conn.commit()

    async def detach_source(self, meeting_id: str, source_id: str):
        """Detach a source from a meeting."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM meeting_attachments WHERE meeting_id = %s AND source_id = %s",
                (meeting_id, source_id),
            )
            await conn.commit()

    async def get_attachments(self, meeting_id: str) -> list[str]:
        """Get all attached source IDs for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                "SELECT source_id FROM meeting_attachments WHERE meeting_id = %s",
                (meeting_id,),
            )
            rows = await cursor.fetchall()
        return [row["source_id"] for row in rows]
