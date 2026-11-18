"""Delegate over the reply chain (Ollama reply model, pre-router clock answers, web search).

`speculate` starts the reply model on the turn's text before the final recognizer flush, so
token generation overlaps it; the host keeps or cancels it with its drift rule. `reply` routes
the turn and returns its sentences; the reply stage gates and speaks them. After a barge-in the
host passes `system_note` (voice/barge_in.py), which reaches the reply model's system prompt.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence

from voxscribe.session.ports import ReplyStream

from app.local import reply_chain

log = logging.getLogger("app.local.voice.delegate")


class OllamaDelegate:
    def __init__(self, app) -> None:
        self._app = app

    @property
    def model(self) -> str:
        return reply_chain.reply_settings(getattr(self._app.state, "settings", None))[0]

    async def warm(self) -> None:
        """One-token request so Ollama loads the reply model and caches the prompt prefix."""
        from app.local.llm_adapter import stream_reply

        try:
            log.info("voice delegate: pre-warming reply model=%s", self.model)
            async for _ in stream_reply("hi", model=self.model):
                break
            log.info("voice delegate: reply model warmed model=%s", self.model)
        except Exception:
            log.debug("voice delegate: prewarm failed (non-critical) model=%s", self.model)

    def speculate(self, partial_text: str, history: Sequence[dict] | None) -> _Speculation:
        from app.local.llm_adapter import stream_reply

        return _Speculation(stream_reply(partial_text, model=self.model,
                                         history=list(history) if history else None))

    async def reply(self, turn_text: str, session_id: str, *, turn_id: str | None = None,
                    history: Sequence[dict] | None = None, speculation=None,
                    system_note: str | None = None) -> ReplyStream:
        """system_note is appended to the system prompt for this turn only (the barge-in note)."""
        src = reply_chain.reply_sentences(
            self._app, session_id, turn_text, turn_id=turn_id,
            history=list(history) if history else None, pre_stream=speculation,
            system_note=system_note)
        return ReplyStream(sentences=src.sentences, source=src.source,
                           cue=reply_chain.THINKING_PHRASE if turn_text.strip() else None,
                           lead_in=src.lead_in, fallback=reply_chain.FALLBACK_PHRASE)


class _Speculation:
    """Reads a token stream into a buffer in the background; iterate to replay, or cancel."""

    def __init__(self, source: AsyncIterator[str]) -> None:
        self._q: asyncio.Queue[str | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._drain(source), name="reply-speculation")

    async def _drain(self, source: AsyncIterator[str]) -> None:
        with contextlib.suppress(Exception):
            async for token in source:
                await self._q.put(token)
        await self._q.put(None)

    def cancel(self) -> None:
        self._task.cancel()
        while not self._q.empty():
            self._q.get_nowait()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        token = await self._q.get()
        if token is None:
            raise StopAsyncIteration
        return token
