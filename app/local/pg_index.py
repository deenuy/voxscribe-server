"""PostgreSQL-backed knowledge index with hybrid retrieval (vector + text search)."""

import hashlib
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool


@dataclass
class Hit:
    """One retrieval result."""

    chunk_id: str
    source_id: str
    text: str
    locator: str
    score: float  # composite score


class PostgresKnowledgeIndex:
    """PostgreSQL index with vector and keyword search via pgvector and tsvector."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def add_source(
        self,
        source_id: str,
        filename: str,
        mime_type: str,
        chunks_with_vecs: list[tuple[str, str, list[float]]],
    ) -> None:
        """Add a source and its chunks with embeddings.

        chunks_with_vecs: list of (text, locator, embedding_vec)
        """
        text = "".join(text for text, _, _ in chunks_with_vecs)
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        async with self.pool.connection() as conn:
            async with conn.transaction():
                # Check if source already exists with this hash
                existing_cur = await conn.execute(
                    "SELECT id FROM sources WHERE id = %s AND content_hash = %s",
                    (source_id, content_hash),
                )
                existing = await existing_cur.fetchone()
                if existing:
                    return

                # Insert source
                try:
                    await conn.execute(
                        """INSERT INTO sources (id, filename, mime_type, content_hash, created_at)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                        (source_id, filename, mime_type, content_hash),
                    )
                except Exception:
                    # Source might already exist with different content
                    return

                # Insert chunks with embeddings
                for ordinal, (text, locator, vec) in enumerate(chunks_with_vecs):
                    chunk_id = f"{source_id}#{ordinal}"
                    embedding = vec if vec else None
                    await conn.execute(
                        """INSERT INTO source_chunks
                        (id, source_id, ordinal, text, locator, embedding, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                        (chunk_id, source_id, ordinal, text, locator, embedding),
                    )

    async def replace_source(
        self,
        source_id: str,
        filename: str,
        mime_type: str,
        chunks_with_vecs: list[tuple[str, str, list[float]]],
    ) -> None:
        """Replace a source (idempotent by content hash)."""
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM source_chunks WHERE source_id = %s", (source_id,)
                )
                await conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        await self.add_source(source_id, filename, mime_type, chunks_with_vecs)

    async def remove_source(self, source_id: str) -> None:
        """Remove a source and its chunks/vectors."""
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM source_chunks WHERE source_id = %s", (source_id,)
                )
                await conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))

    async def list_chunks(
        self, source_id: str | None = None
    ) -> list[tuple[str, str, str, str]]:
        """List chunks: (chunk_id, text, locator, source_id)."""
        async with self.pool.connection() as conn:
            if source_id:
                cur = await conn.execute(
                    """SELECT id, text, locator, source_id FROM source_chunks
                    WHERE source_id = %s ORDER BY ordinal ASC""",
                    (source_id,),
                )
                rows = await cur.fetchall()
            else:
                cur = await conn.execute(
                    """SELECT id, text, locator, source_id FROM source_chunks
                    ORDER BY source_id, ordinal ASC"""
                )
                rows = await cur.fetchall()
        return rows

    async def search(
        self,
        query_vec: list[float],
        query_text: str,
        k: int = 10,
        source_ids: list[str] | None = None,
    ) -> list[Hit]:
        """Hybrid search: cosine similarity + websearch_to_tsquery, merged by RRF.

        Returns top-k results scored by reciprocal rank fusion.
        """
        vec_results = await self._vector_search(query_vec, k * 3, source_ids)
        fts_results = await self._fts_search(query_text, k * 3, source_ids)

        scores = {}
        for rank, (chunk_id, _score) in enumerate(vec_results):
            scores.setdefault(chunk_id, 0)
            scores[chunk_id] += 1.0 / (rank + 1)

        for rank, (chunk_id, _score) in enumerate(fts_results):
            scores.setdefault(chunk_id, 0)
            scores[chunk_id] += 1.0 / (rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:k]

        async with self.pool.connection() as conn:
            hits = []
            for chunk_id, score in sorted_results:
                cur = await conn.execute(
                    """SELECT text, locator, source_id FROM source_chunks
                    WHERE id = %s""",
                    (chunk_id,),
                )
                row = await cur.fetchone()
                if row:
                    text, locator, source_id = row
                    hits.append(
                        Hit(
                            chunk_id=chunk_id,
                            source_id=source_id,
                            text=text,
                            locator=locator,
                            score=score,
                        )
                    )
        return hits

    async def _vector_search(
        self,
        query_vec: list[float],
        limit: int,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Vector similarity search using cosine distance."""
        if not query_vec:
            return []

        async with self.pool.connection() as conn:
            if source_ids:
                placeholders = ",".join("%s" for _ in source_ids)
                sql = f"""
                    SELECT id, 1 - (embedding <=> %s::vector) as score
                    FROM source_chunks
                    WHERE embedding IS NOT NULL AND source_id IN ({placeholders})
                    ORDER BY embedding <=> %s::vector LIMIT %s
                """
                params = [query_vec] + source_ids + [query_vec, limit]
            else:
                sql = """
                    SELECT id, 1 - (embedding <=> %s::vector) as score
                    FROM source_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector LIMIT %s
                """
                params = [query_vec, query_vec, limit]

            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def _fts_search(
        self,
        query_text: str,
        limit: int,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search using tsvector and websearch_to_tsquery."""
        if not query_text:
            return []

        async with self.pool.connection() as conn:
            if source_ids:
                placeholders = ",".join("%s" for _ in source_ids)
                sql = f"""
                    SELECT id, ts_rank(text_tsvector, websearch_to_tsquery('english', %s)) as score
                    FROM source_chunks
                    WHERE text_tsvector @@ websearch_to_tsquery('english', %s)
                      AND source_id IN ({placeholders})
                    ORDER BY score DESC LIMIT %s
                """
                params = [query_text, query_text] + source_ids + [limit]
            else:
                sql = """
                    SELECT id, ts_rank(text_tsvector, websearch_to_tsquery('english', %s)) as score
                    FROM source_chunks
                    WHERE text_tsvector @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC LIMIT %s
                """
                params = [query_text, query_text, limit]

            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def get_source_filename(self, source_id: str) -> str | None:
        """Get source filename for display."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT filename FROM sources WHERE id = %s", (source_id,)
            )
            row = await cur.fetchone()
        return row[0] if row else None
