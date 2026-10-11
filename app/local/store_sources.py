"""Store module for sources and knowledge base.

Tables owned by this module:
- sources
- labels
- source_labels
- source_chunks
"""

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from .models import Label, Source


def _iso(value):
    """Timestamps leave the engine as ISO strings."""
    if value is None or isinstance(value, str):
        return value
    return value.isoformat()


class SourceStore:
    """Storage operations for sources and labels."""

    def __init__(self, pool):
        self.pool = pool

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

    async def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cur = await conn.execute("SELECT * FROM sources WHERE id = %s", (source_id,))
            row = await cur.fetchone()
        if not row:
            return None
        return await self._row_to_source(row)

    async def list_sources(
        self,
        query: str | None = None,
        label_id: str | None = None,
        kind: str | None = None,
        sort: str = "imported",
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[Source], str | None, int, dict]:
        """List sources with filtering and pagination."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            sql_query = "SELECT * FROM sources WHERE parent_id IS NULL"
            params: list[Any] = []

            # Full-text search
            if query:
                sql_query += """
                AND (name ILIKE %s OR id IN (
                    SELECT source_id FROM source_chunks
                    WHERE text_tsvector @@ websearch_to_tsquery('english', %s)
                ))
                """
                params.extend([f"%{query}%", query])

            # Label filter
            if label_id:
                sql_query += " AND id IN (SELECT source_id FROM source_labels WHERE label_id = %s)"
                params.append(label_id)

            # Kind filter
            if kind:
                sql_query += " AND kind = %s"
                params.append(kind)

            # Sorting
            if sort == "imported":
                sql_query += " ORDER BY imported_at DESC NULLS LAST, created_at DESC"
            else:
                sql_query += " ORDER BY name ASC"

            sql_query += " LIMIT %s"
            params.append(limit + 1)

            cur = await conn.execute(sql_query, params)
            rows = await cur.fetchall()

            # Count by kind
            count_cur = await conn.execute(
                "SELECT kind, COUNT(*) as count FROM sources WHERE parent_id IS NULL GROUP BY kind"
            )
            count_rows = await count_cur.fetchall()

        sources = [await self._row_to_source(row) for row in rows[:limit]]
        next_cursor = str(sources[-1].id) if len(rows) > limit else None
        # count_rows are dicts due to row_factory
        counts = {row["kind"]: row["count"] for row in count_rows}
        total = sum(counts.values())

        return sources, next_cursor, total, counts

    async def find_child_id(self, parent_id: str, uri: str) -> str | None:
        """Id of the child source of a folder that points at this file, if any."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id FROM sources WHERE parent_id = %s AND uri = %s", (parent_id, uri)
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def delete_source(self, source_id: str):
        """Delete a source and its chunks/vectors (cascade)."""
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

    async def update_source_chunk_count(self, source_id: str, count: int):
        """Update the chunk count for a source."""
        now = datetime.utcnow().isoformat() + "Z"
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE sources SET chunk_count = %s, updated_at = %s WHERE id = %s",
                (count, now, source_id),
            )
            await conn.commit()

    async def update_source_labels(self, source_id: str, label_ids: list[str]):
        """Update labels for a source."""
        async with self.pool.connection() as conn:
            async with conn.transaction():
                # Delete existing associations
                await conn.execute(
                    "DELETE FROM source_labels WHERE source_id = %s", (source_id,)
                )
                # Add new associations
                for label_id in label_ids:
                    await conn.execute(
                        "INSERT INTO source_labels (source_id, label_id) VALUES (%s, %s)",
                        (source_id, label_id),
                    )
            await conn.commit()

    async def _row_to_source(self, row: dict) -> Source:
        """Convert a database row to a Source model."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            label_cur = await conn.execute(
                """SELECT l.* FROM labels l
                JOIN source_labels sl ON l.id = sl.label_id
                WHERE sl.source_id = %s""",
                (row["id"],),
            )
            label_rows = await label_cur.fetchall()
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
            cur = await conn.execute("SELECT * FROM labels ORDER BY name ASC")
            rows = await cur.fetchall()
        return [Label(id=row["id"], name=row["name"], color=row.get("color")) for row in rows]

    async def update_label(self, label_id: str, name: str | None, color: str | None) -> Label | None:
        """Change a label's name and/or colour; None when there is no such label."""
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            cur = await conn.execute(
                "UPDATE labels SET name = COALESCE(%s, name), color = COALESCE(%s, color) WHERE id = %s RETURNING id, name, color",
                (name, color, label_id),
            )
            row = await cur.fetchone()
            await conn.commit()
        return Label(id=row["id"], name=row["name"], color=row["color"]) if row else None

    async def delete_label(self, label_id: str):
        """Delete a label (cascade removes associations)."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM labels WHERE id = %s", (label_id,))
            await conn.commit()

    # ========================================================================
    # Chunks
    # ========================================================================

    async def get_chunks(
        self, source_id: str, limit: int = 25, cursor: str | None = None
    ) -> tuple[list[dict], str | None]:
        """Get chunks for a source with pagination."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id, ordinal, text, locator FROM source_chunks
                WHERE source_id = %s ORDER BY ordinal ASC LIMIT %s""",
                (source_id, limit + 1),
            )
            rows = await cur.fetchall()
        chunks = [
            {"id": row[0], "ordinal": row[1], "text": row[2], "locator": row[3]}
            for row in rows[:limit]
        ]
        next_cursor = str(rows[limit][0]) if len(rows) > limit else None
        return chunks, next_cursor

    # ========================================================================
    # Privacy Mode
    # ========================================================================

    async def get_privacy_mode(self) -> str:
        """Get current privacy mode from settings."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT value FROM settings WHERE key = %s", ("privacyMode",)
            )
            row = await cur.fetchone()
        if row:
            import json
            value = row[0]
            if isinstance(value, str):
                try:
                    value = json.loads(value)  # a stored JSON string, or a bare word such as "offline"
                except ValueError:
                    pass
            return value.get("value", "offline") if isinstance(value, dict) else str(value)
        return "offline"
