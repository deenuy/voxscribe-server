"""Async retriever for PostgreSQL knowledge index."""


from voxscribe.knowledge import RetrievedChunk


class PostgresRetriever:
    """Async retriever for PostgreSQL knowledge index."""

    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        scope_source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: user query text
            k: number of results
            scope_source_ids: if provided, search only these sources

        Returns:
            list of RetrievedChunk sorted by score descending
        """
        query_vec = self.embedder.embed([query])[0]
        hits = await self.index.search(
            query_vec=query_vec,
            query_text=query,
            k=k,
            source_ids=scope_source_ids,
        )

        results = []
        for hit in hits:
            source_filename = await self.index.get_source_filename(hit.source_id)
            title = source_filename if source_filename else "unknown"

            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    source_id=hit.source_id,
                    title=title,
                    text=hit.text,
                    locator=hit.locator,
                    score=hit.score,
                )
            )

        return results
