"""Reply chain: the reply model, the thinker's search and the reply bookkeeping for wake sessions.

The voice host's reply stage (voxscribe.session.stages.reply) does the speaking: it gates every
sentence before synthesis and plays what passed. This module supplies the two ends of it,
through the Delegate and SessionStore adapters in app/local/voice/adapters:

  begin_reply()       write the turns row and pre-write the replies row. The store then writes
                      the gate's heard record (gate_records.record_heard), so a gate record
                      exists before anything is played (G1)
  talker_sentences()  the talker: the reply model's sentences for one turn, split at
                      punctuation. It has no tools
  search_answer()     the thinker: a web search on the delegate lane (app/local/voice/
                      delegate_lane.py); writes its tool_calls row with the task id
  finish_reply()      append the gate records for the reply (one per evaluation, then the
                      outcome), write the behaviour log (G6), played_at and the latency
                      ledger, then publish the reply event over meeting_bus

Latency budget (idle MacBook): turn_end to first_audio under 1.5 s.
TTS: Kokoro-ONNX primary, macOS say fallback (see tts_engine.py).
LLM: Ollama streaming (see llm_adapter.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_OLLAMA_TIMEOUT_S  = 30.0   # llama3.2 first-token <3s warm, ~8s cold; 30s covers first-time load
_FALLBACK_PHRASE   = "I'm sorry, I didn't catch that. Could you say it again?"
_THINKING_PHRASE   = "hm"
_SEARCH_UNAVAILABLE_PHRASE = "I can't look that up right now."
THINKING_PHRASE    = _THINKING_PHRASE   # spoken as the reply's cue
FALLBACK_PHRASE    = _FALLBACK_PHRASE   # recorded when the reply model yields nothing
_SENTENCE_RE       = re.compile(r'[.!?]["\')\]]*\s+')


async def _sentence_stream(
    user_text: str,
    model: str,
    timeout_s: float,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream LLM tokens and yield complete sentences at punctuation boundaries."""
    from app.local.llm_adapter import stream_reply

    buf = ""
    try:
        async with asyncio.timeout(timeout_s):
            async for token in stream_reply(user_text, model=model, history=history,
                                            system_prompt=system_prompt):
                buf += token
                while True:
                    m = _SENTENCE_RE.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if sentence:
                        yield sentence
    except TimeoutError:
        log.warning("reply_chain: LLM stream timed out")
    if buf.strip():
        yield buf.strip()


async def _sentence_stream_from_tokens(
    token_source: AsyncIterator[str],
    timeout_s: float,
) -> AsyncIterator[str]:
    """Yield sentences from a pre-started token stream (speculative generation)."""
    buf = ""
    try:
        async with asyncio.timeout(timeout_s):
            async for token in token_source:
                buf += token
                while True:
                    m = _SENTENCE_RE.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if sentence:
                        yield sentence
    except TimeoutError:
        log.warning("reply_chain: pre-stream timed out")
    if buf.strip():
        yield buf.strip()


async def warm_phrase_cache(extra: tuple[str, ...] = ()) -> None:
    """Pre-render all fixed gate phrases at startup for zero-latency refusals (V1-P03).
    `extra`: the loaded policy pack's scripted fallback line."""
    from app.local.tts_engine import prerender_phrase
    from app.local.gate import (
        ACTION_CLAIM_PHRASE,
        DEFAULT_REFUSAL_PHRASE,
        CLASSROOM_REFUSAL_PHRASE,
        SENIORCARE_ESCALATION_PHRASE,
    )
    for phrase in (_THINKING_PHRASE, DEFAULT_REFUSAL_PHRASE,
                   CLASSROOM_REFUSAL_PHRASE, SENIORCARE_ESCALATION_PHRASE,
                   ACTION_CLAIM_PHRASE, *extra):
        if phrase:
            await prerender_phrase(phrase)


def _now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def session_tools(pool, meeting_id: str) -> list[str]:
    """Tool names recorded in tool_calls for this session. Fails closed to an empty list."""
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT DISTINCT tool FROM tool_calls WHERE meeting_id=%s", (meeting_id,),
            )
            rows = await cur.fetchall()
        return [r[0] for r in rows]
    except Exception:
        log.exception("reply_chain: tool_calls lookup failed meeting_id=%s", meeting_id)
        return []


@dataclass
class ReplyRecord:
    """The rows one turn's reply writes, created by begin_reply()."""

    meeting_id: str
    turn_n: int
    user_text: str
    mode: str
    model: str
    turn_id: str
    reply_id: str
    ledger_id: str
    reply_pre_written: bool = False
    # Gate records (app/local/gate_records.py). `policy` is the PolicyStamp the session runs
    # under; `gate_events` is the reply's journal of gate evaluations, filled by TurnGate.
    policy: object | None = None
    gate_events: list = field(default_factory=list)
    gate_events_written: int = 0


def reply_settings(settings) -> tuple[str, str]:
    """(reply model, gate mode) from settings."""
    model = getattr(settings, "voice_reply_model", "llama3.2")
    mode = getattr(settings, "voice_mode", "default") if settings is not None else "default"
    return model, mode


async def begin_reply(pool, meeting_id: str, turn_n: int, user_text: str, *,
                      model: str, mode: str) -> ReplyRecord:
    """Write the turns row and pre-write the replies row before anything is played."""
    rec = ReplyRecord(meeting_id=meeting_id, turn_n=turn_n, user_text=user_text or "",
                      mode=mode, model=model, turn_id=str(uuid.uuid4()),
                      reply_id=str(uuid.uuid4()), ledger_id=str(uuid.uuid4()))
    started_z = _now_z()
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO turns (id, meeting_id, turn_n, user_text, started_at)"
                " VALUES (%s, %s, %s, %s, %s)",
                (rec.turn_id, meeting_id, turn_n, rec.user_text, started_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write turn row meeting_id=%s", meeting_id)

    # Placeholder text now; finish_reply() updates the row with the real content. The gate's
    # records are not written here: they are append-only and never hold a placeholder.
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (rec.reply_id, meeting_id, turn_n, "", model, "pending", started_z),
            )
            await conn.commit()
        rec.reply_pre_written = True
    except Exception:
        log.exception("reply_chain: failed to pre-write reply row meeting_id=%s", meeting_id)
    return rec


@dataclass
class ReplySource:
    """What talker_sentences() returns; the Delegate adapter turns it into a ReplyStream."""

    sentences: AsyncIterator[str]
    source: str


def route_turn(settings, user_text: str) -> tuple[str, str] | None:
    """The pre-router's verdict: ("clock", answer), ("search", query) or None."""
    from app.local import pre_router

    return pre_router.route(user_text or "", settings) if settings else None


def _voice_system_prompt(settings) -> str:
    """The context header from the OS clock at turn start (not session start), then the voice
    rules."""
    from app.local.context import current_context_header
    from app.local.llm_adapter import _build_system_prompt

    ctx_header = current_context_header(settings) if settings else ""
    if not ctx_header:
        return _build_system_prompt()
    return (
        f"{ctx_header} "
        "You are a helpful voice assistant. "
        "Reply in 1-2 short sentences only. "
        "Be direct and conversational. No markdown, no lists, no headers. "
        "Keep replies under 30 words."
    )

def talker_system_prompt(settings) -> str:
    """The talker's prompt: the voice prompt plus the instruction to hand anything that needs
    a tool to the delegate lane. The talker itself has no tools."""
    from app.local.voice.delegate_lane import TALKER_DELEGATE_INSTRUCTION

    return f"{_voice_system_prompt(settings)} {TALKER_DELEGATE_INSTRUCTION}"


def talker_sentences(
    app,
    meeting_id: str,
    user_text: str,
    *,
    history: list[dict] | None = None,
    pre_stream: AsyncIterator[str] | None = None,
    system_note: str | None = None,
    system_prefix: str | None = None,
) -> ReplySource:
    """The talker's reply to one turn, as sentences. No tools, nothing is spoken here.

    The speculative stream when the host started one, else the reply model. Routing (clock
    answers, delegation to the thinker) is the Delegate adapter's job and happens before this.
    system_note  appended to the system prompt for this turn (the barge-in note, T13).
    system_prefix  prepended to the system prompt (memory block T19 + control suffix T15).
    """
    settings = getattr(app.state, "settings", None)
    model, _mode = reply_settings(settings)
    log.info("reply_chain: querying %s model=%s history_turns=%d meeting_id=%s",
             "pre_stream" if pre_stream is not None else "llm", model,
             len(history) // 2 if history else 0, meeting_id)
    if pre_stream is not None:
        return ReplySource(_sentence_stream_from_tokens(pre_stream, _OLLAMA_TIMEOUT_S),
                           source="pre_stream")
    prompt = talker_system_prompt(settings)
    if system_prefix:
        prompt = f"{system_prefix}\n\n{prompt}"
    if system_note:
        prompt = f"{prompt} {system_note}"
    return ReplySource(_sentence_stream(user_text, model, _OLLAMA_TIMEOUT_S, history=history,
                                        system_prompt=prompt),
                       source="llm")


def split_sentences(text: str) -> list[str]:
    """A finished text as the sentences the reply stage gates one by one."""
    parts, buf = [], text.strip()
    while True:
        m = _SENTENCE_RE.search(buf)
        if not m:
            break
        parts.append(buf[: m.end()].strip())
        buf = buf[m.end():]
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


async def record_tool_call(pool, *, task_id: str, turn_id: str | None, meeting_id: str | None,
                           tool: str, query: str, engine: str | None, result_count: int,
                           duration_ms: int, error: str | None) -> None:
    """One tool_calls row. Every row carries the delegate task that made the call (A5)."""
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO tool_calls"
                " (id, turn_id, meeting_id, tool, query, engine, result_count,"
                "  duration_ms, error, called_at, task_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (str(uuid.uuid4()), turn_id, meeting_id, tool, query, engine, result_count,
                 duration_ms, error, _now_z(), task_id),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: tool_calls insert failed task_id=%s", task_id)


def _count_results(search_result: str) -> int:
    return len([ln for ln in search_result.splitlines()
                if ln.startswith("1.") or ln[:2] in ("2.", "3.", "4.", "5.")])


async def search_answer(app, meeting_id: str | None, request: str, query: str, *,
                        task_id: str, turn_id: str | None = None,
                        history: list[dict] | None = None) -> str:
    """The thinker's web search: search, write the tool_calls row, then have the reply model
    summarise the results in one or two sentences. Runs on the delegate lane, never in a turn.

    Returns the text to deliver. The lane bounds the whole call with THINKER_TIMEOUT_S; the
    search tool bounds its own request.
    """
    from app.local.tools.web_search import search as web_search
    from app.local.voice.adapters.scope import scoped_pool

    settings = getattr(app.state, "settings", None)
    model, _mode = reply_settings(settings)
    # Called from the voice delegate lane: the pool comes through the voice scope adapter, which refuses (before the
    # search runs) outside a tenant's scope with isolation on, or in a session that is not the tenant's.
    pool = scoped_pool(app)
    log.info("reply_chain: thinker search task_id=%s meeting_id=%s", task_id, meeting_id)
    t_search = time.monotonic()
    search_result = ""
    search_error: str | None = None
    cancelled = False
    try:
        search_result = await web_search(query)
    except asyncio.CancelledError:
        search_error, cancelled = "cancelled", True
    except Exception as exc:
        search_error = str(exc)
        log.warning("reply_chain: web_search failed: %s", type(exc).__name__)

    row = record_tool_call(
        pool, task_id=task_id, turn_id=turn_id, meeting_id=meeting_id, tool="web_search",
        query=query, engine="searxng", result_count=_count_results(search_result),
        duration_ms=int((time.monotonic() - t_search) * 1000), error=search_error)
    if cancelled:
        # The call was made, so the row is still written; then the cancel carries on.
        await asyncio.shield(row)
        raise asyncio.CancelledError
    await row

    if search_error or "[web_search: unavailable" in search_result:
        return _SEARCH_UNAVAILABLE_PHRASE
    search_system = (
        f"{_voice_system_prompt(settings)}\n\n"
        f"Use the following search results to answer the user's question. "
        f"Do not reproduce URLs. Summarise in 1-2 sentences.\n\n"
        f"{search_result}"
    ).strip()
    sentences = [s async for s in _sentence_stream(request, model, _OLLAMA_TIMEOUT_S,
                                                   history=history,
                                                   system_prompt=search_system)]
    return " ".join(sentences)


async def finish_reply(
    pool,
    rec: ReplyRecord,
    *,
    reply_text: str,
    decision: str,
    category: str,
    turn_end_ms: int,
    asr_final_ms: int | None = None,
    llm_request_sent_ms: int | None = None,
    llm_first_token_ms: int | None = None,
    tts_first_audio_ms: int | None = None,
    playback_end_ms: int | None = None,
    played_text: str | None = None,
    full_text: str | None = None,
) -> None:
    """Record what the reply stage decided and spoke. Errors are logged, never raised.

    After a barge-in (T13) reply_text is what was heard, and played_text and full_text are
    written beside it with replies.barge_in set; both are None for a reply that played out.
    """
    from app.local import gate, gate_records, meeting_bus

    meeting_id = rec.meeting_id
    decided_z = _now_z()
    barge_in = played_text is not None
    log.info("reply_chain: reply_len=%d decision=%s barge_in=%s meeting_id=%s",
             len(reply_text), decision, barge_in, meeting_id)
    try:
        async with pool.connection() as conn:
            if rec.reply_pre_written:
                await conn.execute(
                    "UPDATE replies SET text=%s, model=%s, gate_category=%s, created_at=%s,"
                    " barge_in=%s, played_text=%s, full_text=%s WHERE id=%s",
                    (reply_text, rec.model, category, decided_z, barge_in, played_text,
                     full_text, rec.reply_id),
                )
            else:
                await conn.execute(
                    "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category,"
                    " created_at, barge_in, played_text, full_text)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (rec.reply_id, meeting_id, rec.turn_n, reply_text, rec.model, category,
                     decided_z, barge_in, played_text, full_text),
                )
            await gate_records.append_reply_events(conn, rec, decision=decision,
                                                   category=category)
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write reply/gate rows meeting_id=%s", meeting_id)

    gate.write_behaviour_log(meeting_id=meeting_id, turn_n=rec.turn_n, mode=rec.mode,
                             decision=decision, category=category)

    try:
        async with pool.connection() as conn:
            played_z = _now_z()
            await conn.execute("UPDATE replies SET played_at=%s WHERE id=%s",
                               (played_z, rec.reply_id))
            await conn.execute("UPDATE turns SET ended_at=%s WHERE id=%s",
                               (played_z, rec.turn_id))
            await conn.execute(
                "INSERT INTO latency_ledger"
                " (id, meeting_id, turn_n, turn_end_ms, asr_final_ms,"
                "  llm_request_sent_ms, llm_first_token_ms,"
                "  tts_first_audio_ms, playback_end_ms, recorded_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (rec.ledger_id, meeting_id, rec.turn_n, turn_end_ms, asr_final_ms,
                 llm_request_sent_ms, llm_first_token_ms,
                 tts_first_audio_ms, playback_end_ms, played_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write ledger meeting_id=%s", meeting_id)

    try:
        await meeting_bus.publish(meeting_id, "reply", {
            "id": rec.reply_id,
            "text": reply_text,
            "turn_n": rec.turn_n,
            "gate": decision,
            "category": category,
            "bargeIn": barge_in,
        })
    except Exception:
        log.warning("reply_chain: failed to publish reply event meeting_id=%s", meeting_id)
