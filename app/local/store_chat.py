"""Store module for chat threads and messages.

Tables owned by this module:
- chat_threads
- chat_messages
"""

from .timefmt import iso_z
from datetime import datetime

from psycopg.rows import dict_row

from .models import ChatScope, Citation, Message, Thread


def _to_iso_string(dt):
    """Convert datetime to ISO string if needed."""
    return iso_z(dt)


class ChatStore:
    """Storage operations for chat threads and messages."""

    def __init__(self, pool):
        self.pool = pool

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
            scope=ChatScope(
                labelIds=label_ids,
                meetingId=meeting_id,
                sourceIds=source_ids,
            ),
            createdAt=now,
        )

    async def delete_thread(self, thread_id: str) -> None:
        """Delete a thread; its messages go with it (ON DELETE CASCADE)."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM chat_threads WHERE id = %s", (thread_id,))
            await conn.commit()

    async def get_thread(self, thread_id: str) -> Thread | None:
        """Get a thread by ID."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cur = await conn.execute("SELECT * FROM chat_threads WHERE id = %s", (thread_id,))
            row = await cur.fetchone()
        if not row:
            return None
        return Thread(
            id=row["id"],
            scope=ChatScope(
                labelIds=row["label_ids"],
                meetingId=row["meeting_id"],
                sourceIds=row["source_ids"],
            ),
            createdAt=_to_iso_string(row["created_at"]),
        )

    async def list_threads(self) -> list[Thread]:
        """List all chat threads."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cur = await conn.execute("SELECT * FROM chat_threads ORDER BY created_at DESC")
            rows = await cur.fetchall()
        return [
            Thread(
                id=row["id"],
                scope=ChatScope(
                    labelIds=row["label_ids"],
                    meetingId=row["meeting_id"],
                    sourceIds=row["source_ids"],
                ),
                createdAt=_to_iso_string(row["created_at"]),
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
            cur = await conn.execute(
                """SELECT * FROM chat_messages WHERE thread_id = %s ORDER BY created_at ASC""",
                (thread_id,),
            )
            rows = await cur.fetchall()
        return [
            Message(
                id=row["id"],
                role=row["role"],
                text=row["text"],
                citations=[Citation(**c) for c in (row["citations"] or [])],
                createdAt=_to_iso_string(row["created_at"]),
            )
            for row in rows
        ]
