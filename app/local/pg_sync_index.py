"""Synchronous PostgreSQL knowledge index for ingestion worker.

This is a wrapper around PostgreSQL that implements the KnowledgeIndexProtocol
with synchronous methods suitable for use with voxscribe-core's ingest_path.
"""

import psycopg

from . import tenancy


class PostgresKnowledgeIndexSync:
    """Synchronous PostgreSQL index for ingestion."""

    def __init__(self, database_url: str, base_schema: str | None = None):
        self.database_url = database_url
        # A connection of its own, so it must be pointed at the right schema here: the pool cannot do it for us. The
        # tenant in scope decides; without one, the shared schema the pool itself was built on (a test's throwaway one).
        base = f"-c search_path={base_schema},public" if base_schema else None
        self._get_connection = lambda: psycopg.connect(database_url, options=tenancy.search_path_option() or base)

    def add_source(
        self,
        source_id: str,
        filename: str,
        mime_type: str,
        chunks_with_vecs: list[tuple[str, str, list[float]]],
    ) -> None:
        """Add a source and its chunks with embeddings."""
        with self._get_connection() as conn:
            with conn.transaction():
                # Check if source already exists
                existing = conn.execute(
                    "SELECT id FROM sources WHERE id = %s",
                    (source_id,),
                ).fetchone()
                if existing:
                    return

                # Insert source (match actual schema)
                try:
                    conn.execute(
                        """INSERT INTO sources (id, kind, uri, name, mime, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'ready', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                        (source_id, "file", f"ingested:{source_id}", filename, mime_type),
                    )
                except Exception:
                    # Source might already exist
                    return

                # Insert chunks with embeddings
                for ordinal, (text, locator, vec) in enumerate(chunks_with_vecs):
                    chunk_id = f"{source_id}#{ordinal}"
                    embedding = vec if vec else None
                    conn.execute(
                        """INSERT INTO source_chunks
                        (id, source_id, ordinal, text, locator, embedding, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                        (chunk_id, source_id, ordinal, text, locator, embedding),
                    )

    def source_names(self, source_ids) -> dict[str, str]:
        """Map source id to its display name."""
        ids = list(source_ids)
        if not ids:
            return {}
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, name FROM sources WHERE id = ANY(%s)", (ids,)
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def write_chunks(
        self,
        source_id: str,
        mime_type: str,
        chunks_with_vecs: list[tuple[str, str, list[float]]],
    ) -> None:
        """Replace the chunks of an existing source row."""
        with self._get_connection() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM source_chunks WHERE source_id = %s", (source_id,))
                if mime_type:
                    conn.execute(
                        "UPDATE sources SET mime = %s WHERE id = %s", (mime_type, source_id)
                    )
                for ordinal, (text, locator, vec) in enumerate(chunks_with_vecs):
                    conn.execute(
                        """INSERT INTO source_chunks
                        (id, source_id, ordinal, text, locator, embedding, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, CURRENT_TIMESTAMP)""",
                        (f"{source_id}#{ordinal}", source_id, ordinal, text, locator, vec or None),
                    )

    def replace_source(
        self,
        source_id: str,
        filename: str,
        mime_type: str,
        chunks_with_vecs: list[tuple[str, str, list[float]]],
    ) -> None:
        """Replace a source (idempotent by content hash)."""
        with self._get_connection() as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM source_chunks WHERE source_id = %s", (source_id,)
                )
                conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        self.add_source(source_id, filename, mime_type, chunks_with_vecs)

    def remove_source(self, source_id: str) -> None:
        """Remove a source and its chunks/vectors."""
        with self._get_connection() as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM source_chunks WHERE source_id = %s", (source_id,)
                )
                conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))

    def list_chunks(
        self, source_id: str | None = None
    ) -> list[tuple[str, str, str, str]]:
        """List chunks: (chunk_id, text, locator, source_id)."""
        with self._get_connection() as conn:
            if source_id:
                rows = conn.execute(
                    """SELECT id, text, locator, source_id FROM source_chunks
                    WHERE source_id = %s ORDER BY ordinal ASC""",
                    (source_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, text, locator, source_id FROM source_chunks
                    ORDER BY source_id, ordinal ASC"""
                ).fetchall()
        return rows

    def search(
        self,
        query_vec: list[float],
        query_text: str,
        k: int = 10,
        source_ids: list[str] | None = None,
    ) -> list:
        """Hybrid search: cosine similarity + websearch_to_tsquery, merged by RRF."""
        vec_results = self._vector_search(query_vec, k * 3, source_ids)
        fts_results = self._fts_search(query_text, k * 3, source_ids)

        scores = {}
        for rank, (chunk_id, _score) in enumerate(vec_results):
            scores.setdefault(chunk_id, 0)
            scores[chunk_id] += 1.0 / (rank + 1)

        for rank, (chunk_id, _score) in enumerate(fts_results):
            scores.setdefault(chunk_id, 0)
            scores[chunk_id] += 1.0 / (rank + 1)

        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:k]

        with self._get_connection() as conn:
            hits = []
            for chunk_id, score in sorted_results:
                row = conn.execute(
                    """SELECT text, locator, source_id FROM source_chunks
                    WHERE id = %s""",
                    (chunk_id,),
                ).fetchone()
                if row:
                    from app.local.pg_index import Hit
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

    def _vector_search(
        self,
        query_vec: list[float],
        limit: int,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Vector similarity search using cosine distance."""
        if not query_vec:
            return []

        with self._get_connection() as conn:
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

            rows = conn.execute(sql, params).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _fts_search(
        self,
        query_text: str,
        limit: int,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search using tsvector and websearch_to_tsquery."""
        if not query_text:
            return []

        with self._get_connection() as conn:
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

            rows = conn.execute(sql, params).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_source_filename(self, source_id: str) -> str | None:
        """Get source filename for display."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT filename FROM sources WHERE id = %s", (source_id,)
            ).fetchone()
        return row[0] if row else None


class SourceBoundIndex:
    """Index view that stores everything under one existing source row.

    voxscribe-core derives its own source id from the file path. The engine already
    owns the source row (and its id) that the API, meetings and chats refer to, so
    ingestion writes the chunks under that id instead of creating a second source.
    """

    def __init__(self, base: PostgresKnowledgeIndexSync, source_id: str):
        self.base = base
        self.source_id = source_id

    def add_source(self, source_id, filename, mime_type, chunks_with_vecs) -> None:
        self.base.write_chunks(self.source_id, mime_type, chunks_with_vecs)

    replace_source = add_source


class PostgresSyncRetriever:
    """Synchronous retriever over the Postgres index (same interface as core's Retriever).

    Core's Retriever looks titles up through a SQLite connection, which the Postgres
    index does not have, so this one reads the source name from the sources table.
    """

    def __init__(self, index: PostgresKnowledgeIndexSync, embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, scope_source_ids: list[str] | None = None) -> list:
        from voxscribe.knowledge import RetrievedChunk

        query_vec = self.embedder.embed([query])[0]
        hits = self.index.search(
            query_vec=query_vec, query_text=query, k=k, source_ids=scope_source_ids
        )
        names = self.index.source_names({h.source_id for h in hits})
        return [
            RetrievedChunk(
                chunk_id=h.chunk_id,
                source_id=h.source_id,
                title=names.get(h.source_id, "unknown"),
                text=h.text,
                locator=h.locator,
                score=h.score,
            )
            for h in hits
        ]
