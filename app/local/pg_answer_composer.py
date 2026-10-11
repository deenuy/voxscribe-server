"""Async answer composer for PostgreSQL retriever."""

from voxscribe.knowledge import Citation as CoreCitation


class PostgresAnswerComposer:
    """Async grounded Q&A over retrieved context."""

    def __init__(
        self,
        llm,
        retriever,
        min_score_threshold: float = 0.3,
        max_context_tokens: int = 2000,
        min_overlap: float = 0.3,
    ):
        self.llm = llm
        self.retriever = retriever
        self.min_score_threshold = min_score_threshold
        self.max_context_tokens = max_context_tokens
        self.min_overlap = min_overlap

    def _has_query_overlap(self, query: str, text: str) -> float:
        """Check fraction of query terms that appear in text."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        return overlap / len(query_words) if query_words else 0.0

    async def stream(
        self,
        query: str,
        scope_source_ids: list[str] | None = None,
    ) -> tuple:
        """Stream answer tokens and return citations.

        Returns: (token_iterator, citations)

        If retrieval is empty or best score is below threshold, returns
        a "no source found" message without calling the LLM.
        """
        chunks = await self.retriever.retrieve(
            query, k=5, scope_source_ids=scope_source_ids
        )

        if not chunks or chunks[0].score < self.min_score_threshold:
            citations = []

            def no_source_gen():
                yield "I could not find relevant sources for that question."

            return no_source_gen(), citations

        best_overlap = max(
            (self._has_query_overlap(query, c.text) for c in chunks),
            default=0.0,
        )
        if best_overlap < self.min_overlap:
            citations = []

            def no_source_gen():
                yield "I could not find relevant sources for that question."

            return no_source_gen(), citations

        context_text = ""
        context_tokens = 0
        used_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_tokens = len(chunk.text.split())
            if context_tokens + chunk_tokens > self.max_context_tokens:
                break
            context_text += f"\n[{i}] {chunk.title} ({chunk.locator}):\n{chunk.text}\n"
            context_tokens += chunk_tokens
            used_chunks.append(chunk)

        system_prompt = (
            "You are a helpful assistant answering questions based on provided context. "
            "Cite your sources by referencing the [1], [2], etc. numbers in the text. "
            "If the context does not contain relevant information, say so."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        citations = [
            CoreCitation(
                kind="source",
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                locator=chunk.locator,
            )
            for chunk in used_chunks
        ]

        token_stream = self.llm.stream(system_prompt, user_prompt)

        return token_stream, citations
