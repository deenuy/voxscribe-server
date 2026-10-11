"""Dedicated storage for meetings, notes, transcripts, and summaries."""

import json
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .errors import EngineError, InvalidRequestError, NotFoundError
from .models import Block, Citation, EnhancedBlock, Meeting, Segment


MAX_SEGMENT_CHARS = 20_000  # a single spoken turn is a paragraph, never a document
MAX_SPEAKER_CHARS = 120

# Words in a text column, the same way everywhere (stored counts and recomputed counts must agree).
_WORDS_SQL = "COALESCE(sum(cardinality(regexp_split_to_array(btrim(text), '\\s+'))) FILTER (WHERE btrim(text) <> ''), 0)"


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
    ) -> Meeting:
        """Create a new meeting with status 'scheduled'."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """INSERT INTO meetings
                    (id, title, context, calendar_event_id, scheduled_start, scheduled_end, status,
                     project_id, origin, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (meeting_id, title, context, calendar_event_id, scheduled_start, scheduled_end, "scheduled",
                     project_id, origin, now, now),
                )
                await conn.commit()
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
        return await self.get_meeting(meeting_id)

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
    ) -> Meeting:
        """Update meeting fields (only the allowed ones from PATCH)."""
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

        cols = [k for k in updates.keys()]
        vals = [updates[k] for k in cols]
        vals.append(meeting_id)

        set_clause = ", ".join(f"{k} = %s" for k in cols)
        async with self.pool.connection() as conn:
            await conn.execute(
                f"UPDATE meetings SET {set_clause} WHERE id = %s",
                vals,
            )
            await conn.commit()
        return await self.get_meeting(meeting_id)

    async def start_meeting(self, meeting_id: str) -> Meeting:
        """Start recording: set status to 'recording' and startedAt."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE meetings SET status = %s, started_at = %s, updated_at = %s WHERE id = %s",
                ("recording", now, now, meeting_id),
            )
            await conn.commit()
        return await self.get_meeting(meeting_id)

    async def stop_meeting(self, meeting_id: str) -> Meeting:
        """Stop recording: set status to 'processing' and endedAt."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE meetings SET status = %s, ended_at = %s, updated_at = %s WHERE id = %s",
                ("processing", now, now, meeting_id),
            )
            await conn.commit()
        return await self.get_meeting(meeting_id)

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
            scheduledStart=row["scheduled_start"].isoformat() + "Z" if isinstance(row["scheduled_start"], datetime) else row["scheduled_start"],
            scheduledEnd=row["scheduled_end"].isoformat() + "Z" if isinstance(row["scheduled_end"], datetime) else row["scheduled_end"],
            startedAt=row["started_at"].isoformat() + "Z" if isinstance(row["started_at"], datetime) else row["started_at"],
            endedAt=row["ended_at"].isoformat() + "Z" if isinstance(row["ended_at"], datetime) else row["ended_at"],
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
            createdAt=row["created_at"].isoformat() + "Z" if isinstance(row["created_at"], datetime) else row["created_at"],
            updatedAt=row["updated_at"].isoformat() + "Z" if isinstance(row["updated_at"], datetime) else row["updated_at"],
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
                    state = EXCLUDED.state""",
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

                text_after = new_text if new_text is not None else seg["text"]
                speaker_after = new_speaker if speaker_given else seg["speaker"]
                if text_after != seg["text"] or speaker_after != seg["speaker"]:
                    revision = seg["revision"] + 1
                    await conn.execute(
                        """UPDATE transcript_segments
                           SET text = %s, speaker = %s, revision = %s, edited_at = %s, state = 'stable'
                           WHERE id = %s""",
                        (text_after, speaker_after, revision, now, segment_id),
                    )
                    await conn.execute(
                        """INSERT INTO transcript_edits
                           (meeting_id, segment_id, revision, before_text, after_text, before_speaker, after_speaker, edited_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (meeting_id, segment_id, revision, seg["text"], text_after, seg["speaker"], speaker_after, now),
                    )
                    await conn.execute(
                        f"""UPDATE meetings SET transcript_revision = transcript_revision + 1, summary_stale = true,
                            updated_at = %s,
                            word_count = (SELECT {_WORDS_SQL} FROM transcript_segments WHERE meeting_id = %s)
                            WHERE id = %s""",
                        (now, meeting_id, meeting_id),
                    )
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
        """Replace all notes for a meeting, return count accepted."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            # Get existing blocks to check what changed
            cursor = await conn.execute(
                "SELECT block_id, text FROM notes_blocks WHERE meeting_id = %s",
                (meeting_id,),
            )
            existing = await cursor.fetchall()
            existing_map = {row["block_id"]: row["text"] for row in existing}

            # Clear old blocks
            await conn.execute("DELETE FROM notes_blocks WHERE meeting_id = %s", (meeting_id,))

            # Insert new blocks
            for block in blocks:
                await conn.execute(
                    """INSERT INTO notes_blocks (block_id, meeting_id, text, offset_ms, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (block.blockId, meeting_id, block.text, block.offsetMs, now, now),
                )

                # If this block changed, mark its enhanced block as pending
                if block.blockId in existing_map and existing_map[block.blockId] != block.text:
                    await conn.execute(
                        """UPDATE enhanced_blocks SET status = %s, updated_at = %s
                        WHERE block_id = %s AND meeting_id = %s AND status != 'edited'""",
                        ("pending", now, block.blockId, meeting_id),
                    )

            await conn.commit()
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
                    updatedAt=row["updated_at"].isoformat() + "Z" if isinstance(row["updated_at"], datetime) else row["updated_at"],
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
            }
        return {"status": "none", "overview": None, "details": [], "generatedAt": None}

    async def set_summary_pending(self, meeting_id: str):
        """Mark summary as pending (during processing)."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO summaries
                (meeting_id, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    status = 'pending',
                    updated_at = EXCLUDED.updated_at""",
                (meeting_id, "pending", now, now),
            )
            await conn.commit()

    async def set_summary(
        self, meeting_id: str, overview: str, details: list[dict], generated_at: str
    ):
        """Set summary for a meeting as ready."""
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO summaries
                (meeting_id, status, overview, details, generated_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    status = 'ready',
                    overview = EXCLUDED.overview,
                    details = EXCLUDED.details,
                    generated_at = EXCLUDED.generated_at,
                    updated_at = EXCLUDED.updated_at""",
                (meeting_id, "ready", overview, json.dumps(details), generated_at, now, now),
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
