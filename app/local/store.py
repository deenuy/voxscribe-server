"""PostgreSQL storage for the engine."""

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .db import reset_connection
from .tenancy import TenantPool
from .errors import DatabaseUnavailableError, NotFoundError
from .models import (
    Block,
    Citation,
    EnhancedBlock,
    Label,
    Meeting,
    Message,
    Segment,
    Source,
    Thread,
)


def _iso(value):
    """Timestamps leave the engine as ISO strings."""
    if value is None or isinstance(value, str):
        return value
    return value.isoformat()


class Store:
    """PostgreSQL-backed persistent storage."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    @classmethod
    async def create(cls, database_url: str, schema: str | None = None, enforced: bool = False) -> "Store":
        """Create a store with a connection pool.

        Args:
            database_url: PostgreSQL connection URL
            schema: Optional schema name to set search_path (for tests)
            enforced: tenant isolation on: every connection is borrowed for the tenant in scope (see tenancy.py)
        """
        async def configure_conn(conn):
            """Configure each connection with the test schema if provided."""
            if schema:
                await conn.execute(f"SET search_path TO {schema}, public")
                # The pool rejects connections left inside a transaction.
                await conn.commit()

        pool = AsyncConnectionPool(
            database_url,
            min_size=2,
            max_size=10,
            configure=configure_conn if schema else None,
            reset=reset_connection,
        )
        await pool.open()
        return cls(TenantPool(pool, database_url=database_url, enforced=enforced, base_schema=schema))

    async def close(self):
        """Close the connection pool."""
        await self.pool.close()

    async def health_check(self) -> tuple[bool, int, str]:
        """Check database health and return (ok, version, db_name)."""
        try:
            async with self.pool.connection() as conn:
                # Test connectivity
                await conn.execute("SELECT 1")
                # Get schema version
                cursor = await conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
                )
                row = await cursor.fetchone()
                version = row[0] if row else 0
                # Get database name
                cursor = await conn.execute("SELECT current_database()")
                db_row = await cursor.fetchone()
                db_name = db_row[0] if db_row else "unknown"
                return True, version, db_name
        except Exception as e:
            raise DatabaseUnavailableError(f"database health check failed: {e}") from e

    # ========================================================================
    # Meetings
    # ========================================================================

    async def create_meeting(
        self,
        meeting_id: str,
        title: str,
        context: str | None = None,
        calendar_event_id: str | None = None,
        scheduled_start: str | None = None,
        scheduled_end: str | None = None,
    ) -> Meeting:
        """Create a new meeting."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO meetings
                (id, title, context, calendar_event_id, scheduled_start, scheduled_end, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (meeting_id, title, context, calendar_event_id, scheduled_start, scheduled_end, now, now),
            )
            await conn.commit()
        return await self.get_meeting(meeting_id)

    async def get_meeting(self, meeting_id: str) -> Meeting:
        """Get a meeting by ID."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            row = await conn.execute(
                "SELECT * FROM meetings WHERE id = %s", (meeting_id,)
            ).fetchone()
        if not row:
            raise NotFoundError(f"meeting {meeting_id} not found")
        return self._row_to_meeting(row)

    async def list_meetings(
        self,
        query: str | None = None,
        when: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[Meeting], str | None]:
        """List meetings with optional filtering and pagination."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            sql_query = "SELECT * FROM meetings WHERE 1=1"
            params: list[Any] = []

            if when == "upcoming":
                sql_query += " AND status IN ('scheduled', 'recording', 'processing')"
                sql_query += " ORDER BY scheduled_start ASC"
            elif when == "past":
                sql_query += " AND status IN ('done')"
                sql_query += " ORDER BY ended_at DESC"
            else:
                sql_query += " ORDER BY created_at DESC"

            sql_query += " LIMIT %s"
            params.append(limit + 1)

            cursor = await conn.execute(sql_query, params)
            rows = await cursor.fetchall()

        meetings = [self._row_to_meeting(row) for row in rows[:limit]]
        next_cursor = str(meetings[-1]["id"]) if len(rows) > limit else None
        return meetings, next_cursor

    async def update_meeting(self, meeting_id: str, updates: dict) -> Meeting:
        """Update meeting fields."""
        now = datetime.utcnow().isoformat() + "Z"
        updates["updated_at"] = now
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

    async def delete_meeting(self, meeting_id: str):
        """Delete a meeting and all related data."""
        async with self.pool.connection() as conn:
            # Cascade deletes are handled by foreign keys ON DELETE CASCADE
            await conn.execute("DELETE FROM meetings WHERE id = %s", (meeting_id,))
            await conn.commit()

    async def _row_to_meeting(self, row: dict) -> Meeting:
        """Convert a database row to a Meeting model."""
        async with self.pool.connection() as conn:
            attach_count = await conn.execute(
                "SELECT COUNT(*) FROM meeting_attachments WHERE meeting_id = %s",
                (row["id"],),
            ).fetchone()
            has_transcript = await conn.execute(
                "SELECT COUNT(*) FROM transcript_segments WHERE meeting_id = %s",
                (row["id"],),
            ).fetchone()
            has_summary = await conn.execute(
                "SELECT status FROM summaries WHERE meeting_id = %s",
                (row["id"],),
            ).fetchone()

        return Meeting(
            id=row["id"],
            title=row["title"],
            context=row["context"],
            status=row["status"],
            scheduledStart=row["scheduled_start"],
            scheduledEnd=row["scheduled_end"],
            startedAt=row["started_at"],
            endedAt=row["ended_at"],
            calendarEventId=row["calendar_event_id"],
            reminderEnabled=bool(row["reminder_enabled"]),
            enhanceStyle=row["enhance_style"],
            autoStopMinutes=row["auto_stop_minutes"],
            attachmentCount=attach_count[0] if attach_count else 0,
            hasTranscript=bool(has_transcript and has_transcript[0] > 0),
            hasSummary=bool(has_summary and has_summary[0] == "ready"),
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
        )

    def _row_to_meeting_sync(self, row: dict) -> Meeting:
        """Sync version (for internal use)."""
        return Meeting(
            id=row["id"],
            title=row["title"],
            context=row["context"],
            status=row["status"],
            scheduledStart=row["scheduled_start"],
            scheduledEnd=row["scheduled_end"],
            startedAt=row["started_at"],
            endedAt=row["ended_at"],
            calendarEventId=row["calendar_event_id"],
            reminderEnabled=bool(row["reminder_enabled"]),
            enhanceStyle=row["enhance_style"],
            autoStopMinutes=row["auto_stop_minutes"],
            attachmentCount=0,
            hasTranscript=False,
            hasSummary=False,
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
        )

    # ========================================================================
    # Transcript Segments
    # ========================================================================

    async def add_segment(self, meeting_id: str, segment: Segment) -> Segment:
        """Add or update a transcript segment."""
        now = datetime.utcnow().isoformat() + "Z"
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

    async def get_transcript(self, meeting_id: str) -> list[Segment]:
        """Get all transcript segments for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await conn.execute(
                """SELECT * FROM transcript_segments WHERE meeting_id = %s
                ORDER BY start_ms ASC""",
                (meeting_id,),
            ).fetchall()
        return [
            Segment(
                id=row["id"],
                startMs=row["start_ms"],
                endMs=row["end_ms"],
                source=row["source"],
                speaker=row["speaker"],
                text=row["text"],
                state=row["state"],
            )
            for row in rows
        ]

    # ========================================================================
    # Notes (Shorthand Blocks)
    # ========================================================================

    async def put_notes(self, meeting_id: str, blocks: list[Block]) -> int:
        """Replace all notes for a meeting."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM notes_blocks WHERE meeting_id = %s", (meeting_id,))
            for block in blocks:
                await conn.execute(
                    """INSERT INTO notes_blocks (block_id, meeting_id, text, offset_ms, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (block.blockId, meeting_id, block.text, block.offsetMs, now, now),
                )
            await conn.commit()
        return len(blocks)

    async def get_notes(self, meeting_id: str) -> list[Block]:
        """Get all notes for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await conn.execute(
                """SELECT block_id, text, offset_ms FROM notes_blocks
                WHERE meeting_id = %s ORDER BY offset_ms ASC""",
                (meeting_id,),
            ).fetchall()
        return [Block(blockId=row["block_id"], text=row["text"], offsetMs=row["offset_ms"]) for row in rows]

    # ========================================================================
    # Enhanced Blocks
    # ========================================================================

    async def get_enhanced(self, meeting_id: str) -> dict:
        """Get enhanced blocks for a meeting."""
        meeting = await self.get_meeting(meeting_id)
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await conn.execute(
                """SELECT * FROM enhanced_blocks WHERE meeting_id = %s ORDER BY
                (SELECT offset_ms FROM notes_blocks WHERE block_id = enhanced_blocks.block_id) ASC""",
                (meeting_id,),
            ).fetchall()
        return {
            "style": meeting.enhanceStyle,
            "blocks": [
                EnhancedBlock(
                    blockId=row["block_id"],
                    text=row["text"],
                    status=row["status"],
                    citations=[Citation(**c) for c in (row["citations"] or [])],
                    updatedAt=row["updated_at"],
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
        """Set an enhanced block."""
        now = datetime.utcnow().isoformat() + "Z"
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
                (block_id, meeting_id, text, status, citations_list, now, now),
            )
            await conn.commit()

    async def edit_enhanced_block(self, meeting_id: str, block_id: str, text: str):
        """Mark an enhanced block as edited."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE enhanced_blocks SET text = %s, status = 'edited', updated_at = %s
                WHERE block_id = %s AND meeting_id = %s""",
                (text, now, block_id, meeting_id),
            )
            await conn.commit()

    # ========================================================================
    # Summaries
    # ========================================================================

    async def get_summary(self, meeting_id: str) -> dict:
        """Get summary for a meeting."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            row = await conn.execute(
                "SELECT * FROM summaries WHERE meeting_id = %s", (meeting_id,)
            ).fetchone()
        if row:
            return {
                "status": row["status"],
                "overview": row["overview"],
                "details": row["details"] or [],
                "generatedAt": row["generated_at"],
            }
        return {"status": "none", "overview": None, "details": [], "generatedAt": None}

    async def set_summary(
        self, meeting_id: str, overview: str, details: list[dict], generated_at: str
    ):
        """Set summary for a meeting."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO summaries
                (meeting_id, status, overview, details, generated_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    overview = EXCLUDED.overview,
                    details = EXCLUDED.details,
                    generated_at = EXCLUDED.generated_at,
                    updated_at = EXCLUDED.updated_at""",
                (meeting_id, "ready", overview, details, generated_at, now, now),
            )
            await conn.commit()

    # ========================================================================
    # Sources
    # ========================================================================

    async def create_source(
        self,
        source_id: str,
        kind: str,
        uri: str,
        name: str,
        mime: str | None = None,
        parent_id: str | None = None,
    ) -> Source:
        """Create a new source."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO sources (id, kind, uri, name, mime, parent_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (source_id, kind, uri, name, mime, parent_id, now, now),
            )
            await conn.commit()
        return await self.get_source(source_id)

    async def get_source(self, source_id: str) -> Source:
        """Get a source by ID."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute("SELECT * FROM sources WHERE id = %s", (source_id,))
            row = await cursor.fetchone()
        if not row:
            raise NotFoundError(f"source {source_id} not found")
        return await self._row_to_source(row)

    async def list_sources(
        self,
        query: str | None = None,
        label_id: str | None = None,
        kind: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[Source], str | None, int, dict]:
        """List sources with filtering and pagination."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            sql_query = "SELECT * FROM sources WHERE parent_id IS NULL"
            params: list[Any] = []

            if kind:
                sql_query += " AND kind = %s"
                params.append(kind)

            if label_id:
                sql_query += " AND id IN (SELECT source_id FROM source_labels WHERE label_id = %s)"
                params.append(label_id)

            sql_query += " ORDER BY imported_at DESC, created_at DESC LIMIT %s"
            params.append(limit + 1)

            cursor = await conn.execute(sql_query, params)
            rows = await cursor.fetchall()

            # Count by kind
            cursor2 = await conn.execute(
                "SELECT kind, COUNT(*) as count FROM sources WHERE parent_id IS NULL GROUP BY kind"
            )
            count_rows = await cursor2.fetchall()

        sources = [await self._row_to_source(row) for row in rows[:limit]]
        next_cursor = str(sources[-1].id) if len(rows) > limit else None
        counts = {row["kind"]: row["count"] for row in count_rows}
        total = sum(counts.values())

        return sources, next_cursor, total, counts

    async def delete_source(self, source_id: str):
        """Delete a source and its chunks/vectors."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
            await conn.commit()

    async def update_source_status(self, source_id: str, status: str, error: str | None = None):
        """Update source status."""
        now = datetime.utcnow().isoformat() + "Z"
        imported_at = now if status == "ready" else None
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE sources SET status = %s, error = %s, imported_at = %s, updated_at = %s
                WHERE id = %s""",
                (status, error, imported_at, now, source_id),
            )
            await conn.commit()

    async def _row_to_source(self, row: dict) -> Source:
        """Convert a database row to a Source model."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute(
                """SELECT l.* FROM labels l
                JOIN source_labels sl ON l.id = sl.label_id
                WHERE sl.source_id = %s""",
                (row["id"],),
            )
            label_rows = await cursor.fetchall()
        labels = [Label(id=lr["id"], name=lr["name"], color=lr.get("color")) for lr in label_rows]
        return Source(
            id=row["id"],
            kind=row["kind"],
            uri=row["uri"],
            name=row["name"],
            mime=row["mime"],
            parentId=row["parent_id"],
            status=row["status"],
            error=row["error"],
            importedAt=_iso(row["imported_at"]),
            chunkCount=row["chunk_count"],
            labels=labels,
        )

    # ========================================================================
    # Source Chunks
    # ========================================================================

    async def add_chunk(
        self,
        source_id: str,
        chunk_id: str,
        ordinal: int,
        text: str,
        locator: str | None = None,
        embedding: bytes | None = None,
    ):
        """Add a chunk to a source."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO source_chunks (id, source_id, ordinal, text, locator, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (chunk_id, source_id, ordinal, text, locator, embedding, now),
            )
            # Update chunk count
            count = await conn.execute(
                "SELECT COUNT(*) FROM source_chunks WHERE source_id = %s", (source_id,)
            ).fetchone()
            await conn.execute(
                "UPDATE sources SET chunk_count = %s WHERE id = %s", (count[0], source_id)
            )
            await conn.commit()

    async def get_chunks(
        self, source_id: str, limit: int = 25, cursor: str | None = None
    ) -> tuple[list[dict], str | None]:
        """Get chunks for a source."""
        async with self.pool.connection() as conn:
            rows = await conn.execute(
                """SELECT id, ordinal, text, locator FROM source_chunks
                WHERE source_id = %s ORDER BY ordinal ASC LIMIT %s""",
                (source_id, limit + 1),
            ).fetchall()
        chunks = [
            {"id": row[0], "ordinal": row[1], "text": row[2], "locator": row[3]}
            for row in rows[:limit]
        ]
        next_cursor = str(rows[limit][0]) if len(rows) > limit else None
        return chunks, next_cursor

    # ========================================================================
    # Labels
    # ========================================================================

    async def create_label(self, label_id: str, name: str, color: str | None = None) -> Label:
        """Create a label."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO labels (id, name, color, created_at) VALUES (%s, %s, %s, %s)",
                (label_id, name, color, now),
            )
            await conn.commit()
        return Label(id=label_id, name=name, color=color)

    async def list_labels(self) -> list[Label]:
        """List all labels."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute("SELECT * FROM labels ORDER BY name ASC")
            rows = await cursor.fetchall()
        return [Label(id=row["id"], name=row["name"], color=row.get("color")) for row in rows]

    async def delete_label(self, label_id: str):
        """Delete a label."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM labels WHERE id = %s", (label_id,))
            await conn.commit()

    # ========================================================================
    # Chat
    # ========================================================================

    async def create_thread(
        self,
        thread_id: str,
        label_ids: list[str] | None = None,
        meeting_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> Thread:
        """Create a chat thread."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO chat_threads (id, label_ids, meeting_id, source_ids, created_at)
                VALUES (%s, %s, %s, %s, %s)""",
                (thread_id, label_ids or [], meeting_id, source_ids or [], now),
            )
            await conn.commit()
        return Thread(
            id=thread_id,
            scope={
                "labelIds": label_ids,
                "meetingId": meeting_id,
                "sourceIds": source_ids,
            },
            createdAt=now,
        )

    async def get_thread(self, thread_id: str) -> Thread | None:
        """Get a thread with its messages."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute("SELECT * FROM chat_threads WHERE id = %s", (thread_id,))
            row = await cursor.fetchone()
        if not row:
            return None
        return Thread(
            id=row["id"],
            scope={
                "labelIds": row["label_ids"],
                "meetingId": row["meeting_id"],
                "sourceIds": row["source_ids"],
            },
            createdAt=row["created_at"],
        )

    async def list_threads(self) -> list[Thread]:
        """List all chat threads."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cursor = await conn.execute("SELECT * FROM chat_threads ORDER BY created_at DESC")
            rows = await cursor.fetchall()
        return [
            Thread(
                id=row["id"],
                scope={
                    "labelIds": row["label_ids"],
                    "meetingId": row["meeting_id"],
                    "sourceIds": row["source_ids"],
                },
                createdAt=row["created_at"],
            )
            for row in rows
        ]

    async def add_message(
        self, message_id: str, thread_id: str, role: str, text: str, citations: list[Citation]
    ) -> Message:
        """Add a message to a thread."""
        now = datetime.utcnow().isoformat() + "Z"
        citations_list = [c.model_dump(exclude_none=True) for c in citations]
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO chat_messages (id, thread_id, role, text, citations, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (message_id, thread_id, role, text, citations_list, now),
            )
            await conn.commit()
        return Message(
            id=message_id,
            role=role,
            text=text,
            citations=citations,
            createdAt=now,
        )

    async def get_messages(self, thread_id: str) -> list[Message]:
        """Get all messages in a thread."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await conn.execute(
                """SELECT * FROM chat_messages WHERE thread_id = %s ORDER BY created_at ASC""",
                (thread_id,),
            ).fetchall()
        return [
            Message(
                id=row["id"],
                role=row["role"],
                text=row["text"],
                citations=[Citation(**c) for c in (row["citations"] or [])],
                createdAt=row["created_at"],
            )
            for row in rows
        ]

    # ========================================================================
    # Settings
    # ========================================================================

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = await cursor.fetchone()
        if row:
            return row[0]
        return default

    async def set_setting(self, key: str, value: Any):
        """Set a setting."""
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )
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

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def delete_all_data(self):
        """Delete all data (for test cleanup)."""
        async with self.pool.connection() as conn:
            await conn.execute("""
DELETE FROM chat_messages;
DELETE FROM chat_threads;
DELETE FROM meeting_attachments;
DELETE FROM enhanced_blocks;
DELETE FROM notes_blocks;
DELETE FROM transcript_segments;
DELETE FROM summaries;
DELETE FROM source_chunks;
DELETE FROM source_labels;
DELETE FROM sources;
DELETE FROM meetings;
DELETE FROM calendar_events;
DELETE FROM settings;
            """)
            await conn.commit()
