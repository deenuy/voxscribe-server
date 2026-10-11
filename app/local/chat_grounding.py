"""Chat grounding using voxscribe-core's Retriever and AnswerComposer."""

import asyncio

from voxscribe.knowledge import AnswerComposer, Retriever


class AsyncRetriever:
    """Async wrapper for voxscribe-core's Retriever."""

    def __init__(self, index, embedder):
        """Initialize with sync index and embedder."""
        self.retriever = Retriever(index, embedder)

    async def retrieve(self, query: str, k: int = 5, scope_source_ids=None):
        """Retrieve chunks asynchronously."""
        return await asyncio.to_thread(
            self.retriever.retrieve,
            query,
            k,
            scope_source_ids,
        )


class AsyncAnswerComposer:
    """Async wrapper for voxscribe-core's AnswerComposer."""

    def __init__(
        self,
        llm,
        retriever,
        min_score_threshold: float = 0.3,
        max_context_tokens: int = 2000,
        min_overlap: float = 0.3,
        min_query_terms_covered: int = 1,
        cosine_rel_threshold: float = 0.2,
        use_fake_embedder: bool = False,
    ):
        """Initialize with sync AnswerComposer."""
        self.composer = AnswerComposer(
            llm,
            retriever,
            min_score_threshold=min_score_threshold,
            max_context_tokens=max_context_tokens,
            min_overlap=min_overlap,
            min_query_terms_covered=min_query_terms_covered,
            cosine_rel_threshold=cosine_rel_threshold,
            use_fake_embedder=use_fake_embedder,
        )

    async def stream(self, query: str, scope_source_ids=None):
        """Stream answer with citations asynchronously."""
        return await asyncio.to_thread(
            self.composer.stream,
            query,
            scope_source_ids,
        )
