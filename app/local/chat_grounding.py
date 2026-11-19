"""What a chat answer rests on, and the reply stream kept off the event loop.

Every database read here borrows from the pool the caller passes in (`app.state.store.pool`, the TenantPool), so
with tenant isolation on it reads only the tables of the tenant in scope, and fails closed with none. Nothing here
opens a connection of its own.

Two kinds of thread:

  a scribe thread   (scope.meetingId, the "Ask about this scribe" panel) is answered from that scribe's transcript,
                    read from the database at the moment of the question. Transcripts are not indexed as sources,
                    and are not indexed for this either: segments are stored as they are recognised, so a question
                    asked the second after Stop already has the whole transcript, with no indexing step to wait for
                    and no second copy of the words to keep in step with edits, deletion and erase. The transcript
                    goes to the model bounded (TRANSCRIPT_CONTEXT_CHARS); a longer one is cut down to the segments
                    that share words with the question and their neighbours. The model is told to answer only from
                    it and to say NOT_IN_TRANSCRIPT, in those words, when it does not hold the answer. A scribe that
                    is not the caller's (not in the tenant's schema), or that has no words, is answered with a fixed
                    sentence and no model call.
  any other thread  hybrid retrieval over the tenant's sources (pg_index.PostgresKnowledgeIndex on the pool), within
                    scope.sourceIds when the thread has them, then core's AnswerComposer decides whether the hits
                    ground an answer, exactly as before.

The model's stream is a blocking iterator. stream_off_loop() runs it on a worker thread and hands the tokens to the
event loop, which also runs the wake word and live transcription and must never wait for a model.

Nothing here logs a question, a transcript, a prompt or an answer: lengths and counts only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

from voxscribe.knowledge import AnswerComposer, RetrievedChunk
from voxscribe.knowledge import Citation as CoreCitation

from .errors import NotFoundError
from .pg_index import PostgresKnowledgeIndex
from .store_meetings import MeetingStore

log = logging.getLogger(__name__)

# The most transcript text one question sends to the model. About 6000 words: inside an 8k-token context with room
# for the instructions and the answer.
TRANSCRIPT_CONTEXT_CHARS = int(os.getenv("VOXSCRIBE_CHAT_TRANSCRIPT_CHARS", "24000"))
CITED_SEGMENTS = 3          # transcript citations on one answer: the segments that share most words with the question

NOT_IN_TRANSCRIPT = "The transcript of this scribe does not say."
NO_TRANSCRIPT = "This scribe has no transcript yet, so there is nothing to answer from."
SCRIBE_NOT_FOUND = "I could not find this scribe."

SCRIBE_SYSTEM = (
    "You answer questions about one recorded conversation, using only its transcript below. "
    "Each line starts with the time it was said. Answer briefly and plainly. "
    "Do not use anything you know from elsewhere, and do not guess. "
    f'If the transcript does not contain the answer, reply with exactly: "{NOT_IN_TRANSCRIPT}"'
)

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "he", "her",
    "his", "how", "i", "in", "is", "it", "me", "my", "no", "not", "of", "on", "or", "she", "so", "that", "the",
    "their", "them", "they", "this", "to", "up", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "about", "said", "say", "tell",
})
_WORD = re.compile(r"[a-z0-9']+")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP_WORDS and len(w) > 1}


@dataclass
class Grounding:
    """What one answer rests on: a factory for the model's token stream (called on a worker thread) and the
    citations that go with it. `model` is False when the answer is a fixed sentence and no model is asked."""

    tokens: Callable[[], Iterator[str]]
    citations: list[CoreCitation] = field(default_factory=list)
    model: bool = True
    context_chars: int = 0


def _fixed(sentence: str) -> Grounding:
    return Grounding(tokens=lambda: iter([sentence]), model=False)


def _clock(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


def _line(seg) -> str:
    who = f"{seg.speaker}: " if seg.speaker else ""
    return f"[{_clock(seg.startMs)}] {who}{seg.text.strip()}"


def select_segments(segments: list, question: str, limit: int = TRANSCRIPT_CONTEXT_CHARS) -> list:
    """The segments that go to the model, in spoken order, at most `limit` characters of lines.

    The whole transcript when it fits. Otherwise the segments that share most words with the question, each with
    its neighbours (an answer often sits in the line after the one that names the subject), best first until the
    limit. A question that shares no word with a long transcript ("what was decided?") gets its start and its end."""
    lines = [_line(s) for s in segments]
    if sum(len(x) + 1 for x in lines) <= limit:
        return list(segments)
    wanted = _terms(question)
    scores = [len(wanted & _terms(s.text)) for s in segments]
    order: list[int] = []
    if any(scores):
        for i in sorted(range(len(segments)), key=lambda i: (-scores[i], i)):
            if scores[i] == 0:
                break
            order += [j for j in (i, i - 1, i + 1) if 0 <= j < len(segments)]
    else:
        head, tail = 0, len(segments) - 1
        while head <= tail:
            order += [head, tail]
            head, tail = head + 1, tail - 1
    chosen: set[int] = set()
    used = 0
    for i in order:
        if i in chosen:
            continue
        cost = len(lines[i]) + 1
        if used + cost > limit:
            continue
        chosen.add(i)
        used += cost
    return [segments[i] for i in sorted(chosen)]


def _transcript_citations(segments: list, question: str) -> list[CoreCitation]:
    wanted = _terms(question)
    scored = sorted(((len(wanted & _terms(s.text)), i) for i, s in enumerate(segments)), key=lambda x: (-x[0], x[1]))
    best = sorted(i for score, i in scored[:CITED_SEGMENTS] if score > 0)
    return [CoreCitation(kind="transcript", segment_id=segments[i].id, start_ms=segments[i].startMs,
                         end_ms=segments[i].endMs) for i in best]


async def ground_in_scribe(pool, llm, meeting_id: str, question: str) -> Grounding:
    """The grounding for a question in a scribe thread: that scribe's transcript, read through the pool."""
    store = MeetingStore(pool)
    try:
        await store.get_meeting(meeting_id)
    except NotFoundError:
        return _fixed(SCRIBE_NOT_FOUND)
    spoken = [s for s in await store.get_transcript(meeting_id) if s.text and s.text.strip()]
    if not spoken:
        return _fixed(NO_TRANSCRIPT)
    chosen = select_segments(spoken, question)
    context = "\n".join(_line(s) for s in chosen)
    prompt = f"Transcript:\n{context}\n\nQuestion: {question}"
    log.info("chat: scribe grounding segments=%d of %d context_chars=%d question_chars=%d",
             len(chosen), len(spoken), len(context), len(question))
    return Grounding(tokens=lambda: llm.stream(SCRIBE_SYSTEM, prompt),
                     citations=_transcript_citations(chosen, question), context_chars=len(context))


class _Prefetched:
    """A retriever that returns chunks already read through the pool, so core's AnswerComposer applies its
    grounding rules without touching a database itself."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def retrieve(self, query, k=5, scope_source_ids=None):
        return self.chunks[:k]


async def ground_in_sources(pool, llm, embedder, question: str, source_ids: list[str] | None, *,
                            fake_embedder: bool = False) -> Grounding:
    """The grounding for any other thread: the tenant's sources (within `source_ids` when given), through the pool."""
    index = PostgresKnowledgeIndex(pool)
    vectors = await asyncio.to_thread(embedder.embed, [question])
    hits = await index.search(query_vec=vectors[0], query_text=question, k=5, source_ids=source_ids or None)
    names = await index.source_names({h.source_id for h in hits})
    chunks = [RetrievedChunk(chunk_id=h.chunk_id, source_id=h.source_id, title=names.get(h.source_id, "unknown"),
                             text=h.text, locator=h.locator, score=h.score) for h in hits]
    composer = AnswerComposer(llm, _Prefetched(chunks), use_fake_embedder=fake_embedder)
    # stream() only decides and builds the prompt here; the model is asked when the iterator is read.
    tokens, citations = await asyncio.to_thread(composer.stream, question, source_ids or None)
    log.info("chat: source grounding hits=%d cited=%d question_chars=%d", len(hits), len(citations), len(question))
    return Grounding(tokens=lambda: tokens, citations=citations, model=bool(citations))


_workers: set[asyncio.Future] = set()       # generation threads still finishing after their reader left


async def stream_off_loop(make_tokens: Callable[[], Iterator[str]]) -> AsyncIterator[str]:
    """The tokens of a blocking iterator, produced on a worker thread and read on the event loop.

    The event loop only ever waits on a queue. When the reader stops early (the client went away, an error), the
    worker is told to stop at its next token; a model call already in flight finishes on its thread."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def hand(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:        # the loop has closed: nobody is reading
            stop.set()

    def work() -> None:
        try:
            for token in make_tokens():
                if stop.is_set():
                    return
                hand(("token", token))
        except BaseException as exc:  # noqa: BLE001 - handed to the reader, which raises it
            hand(("error", exc))
            return
        hand(("end", None))

    worker = asyncio.ensure_future(asyncio.to_thread(work))
    _workers.add(worker)
    worker.add_done_callback(_workers.discard)
    try:
        while True:
            kind, value = await queue.get()
            if kind == "token":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        stop.set()
