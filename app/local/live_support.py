"""Builders for the retrieval and assist pieces used during a live meeting."""

from voxscribe.knowledge import AnswerComposer
from voxscribe.meeting.assist import LiveAssist
from voxscribe.meeting.qdetect import QuestionDetector

from .pg_sync_index import PostgresKnowledgeIndexSync, PostgresSyncRetriever


def build_retriever(app) -> PostgresSyncRetriever | None:
    """Core retriever over the Postgres index, or None when embeddings are unavailable."""
    embedder = app.state.providers.get("embedding_provider")
    if embedder is None:
        return None
    return PostgresSyncRetriever(
        PostgresKnowledgeIndexSync(app.state.settings.database_url, getattr(app.state.store.pool, "base_schema", None)), embedder
    )


class _ScopedComposer:
    """Fixes the source scope so LiveAssist can call stream(question) alone."""

    def __init__(self, composer: AnswerComposer, scope_source_ids: list[str] | None):
        self.composer = composer
        self.scope = scope_source_ids

    def stream(self, query: str):
        return self.composer.stream(query, self.scope)


def build_live_assist(app, scope_source_ids: list[str] | None) -> LiveAssist | None:
    """Live assist on the live model, answering only from the given sources (all if None)."""
    retriever = build_retriever(app)
    llm = app.state.providers.get("live_llm")
    if retriever is None or llm is None:
        return None
    composer = AnswerComposer(
        llm,
        retriever,
        use_fake_embedder=app.state.settings.providers == "fake",
    )
    return LiveAssist(QuestionDetector(), _ScopedComposer(composer, scope_source_ids))
