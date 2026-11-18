"""Reply chain: the reply model, the search route and the reply bookkeeping for wake sessions.

The voice host's reply stage (voxscribe.session.stages.reply) does the speaking: it gates every
sentence before synthesis and plays what passed. This module supplies the two ends of it,
through the Delegate and SessionStore adapters in app/local/voice/adapters:

  begin_reply()      write the turns row, pre-write replies + gate_decisions (G1: the gate row
                     exists before anything is played)
  reply_sentences()  route the turn (clock answer, web search, or the reply model) and stream
                     its sentences, split at punctuation; a search writes its tool_calls row
  finish_reply()     write the gate outcome, the behaviour log (G6), played_at and the latency
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
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_OLLAMA_TIMEOUT_S  = 30.0   # llama3.2 first-token <3s warm, ~8s cold; 30s covers first-time load
_FALLBACK_PHRASE   = "I'm sorry, I didn't catch that. Could you say it again?"
_THINKING_PHRASE   = "hm"
_FILLER_PHRASE     = "Let me check."
_SEARCH_UNAVAILABLE_PHRASE = "I can't look that up right now."
_SEARCH_TIMEOUT_S  = 4.0
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


async def warm_phrase_cache() -> None:
    """Pre-render all fixed gate phrases at startup for zero-latency refusals (V1-P03)."""
    from app.local.tts_engine import prerender_phrase
    from app.local.gate import (
        ACTION_CLAIM_PHRASE,
        DEFAULT_REFUSAL_PHRASE,
        CLASSROOM_REFUSAL_PHRASE,
        SENIORCARE_ESCALATION_PHRASE,
    )
    for phrase in (_THINKING_PHRASE, _FILLER_PHRASE, DEFAULT_REFUSAL_PHRASE,
                   CLASSROOM_REFUSAL_PHRASE, SENIORCARE_ESCALATION_PHRASE,
                   ACTION_CLAIM_PHRASE):
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
    gate_id: str
    ledger_id: str
    reply_pre_written: bool = False
    gate_pre_written: bool = False


def reply_settings(settings) -> tuple[str, str]:
    """(reply model, gate mode) from settings."""
    model = getattr(settings, "voice_reply_model", "llama3.2")
    mode = getattr(settings, "voice_mode", "default") if settings is not None else "default"
    return model, mode


async def begin_reply(pool, meeting_id: str, turn_n: int, user_text: str, *,
                      model: str, mode: str) -> ReplyRecord:
    """Write the turns row and pre-write replies + gate_decisions before anything is played."""
    rec = ReplyRecord(meeting_id=meeting_id, turn_n=turn_n, user_text=user_text or "",
                      mode=mode, model=model, turn_id=str(uuid.uuid4()),
                      reply_id=str(uuid.uuid4()), gate_id=str(uuid.uuid4()),
                      ledger_id=str(uuid.uuid4()))
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

    # gate_decisions has an FK on replies(id), so both are inserted together. Placeholder text
    # now; finish_reply() updates them with the real content.
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (rec.reply_id, meeting_id, turn_n, "", model, "pending", started_z),
            )
            await conn.execute(
                "INSERT INTO gate_decisions"
                " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (rec.gate_id, rec.reply_id, meeting_id, mode,
                 rec.user_text[:1000], "pending", "pending", started_z),
            )
            await conn.commit()
        rec.reply_pre_written = True
        rec.gate_pre_written = True
    except Exception:
        log.exception("reply_chain: failed to pre-write gate_decisions meeting_id=%s", meeting_id)
    return rec


@dataclass
class ReplySource:
    """What reply_sentences() returns; the Delegate adapter turns it into a ReplyStream."""

    sentences: AsyncIterator[str]
    source: str
    lead_in: str | None = None


def reply_sentences(
    app,
    meeting_id: str,
    user_text: str,
    *,
    turn_id: str | None = None,
    history: list[dict] | None = None,
    pre_stream: AsyncIterator[str] | None = None,
    system_note: str | None = None,
) -> ReplySource:
    """Route one turn and return its sentence source. Nothing is spoken here.

    clock   the pre-router's answer, one sentence
    search  "Let me check." as the lead-in, then a web search (tool_calls row written) and the
            reply model grounded on the results, or a fixed line when search is unavailable
    llm     the speculative stream when the host started one, else the reply model

    system_note is appended to the system prompt for this turn (the barge-in note, T13). The
    host never starts a speculative stream on such a turn, since that stream has no prompt.
    """
    from app.local import pre_router
    from app.local.context import current_context_header

    settings = getattr(app.state, "settings", None)
    model, _mode = reply_settings(settings)

    # Context header from the OS clock at turn start (not session start).
    ctx_header = current_context_header(settings) if settings else ""
    system_prompt = (
        f"{ctx_header} "
        "You are a helpful voice assistant. "
        "Reply in 1-2 short sentences only. "
        "Be direct and conversational. No markdown, no lists, no headers. "
        "Keep replies under 30 words."
    ).strip() if ctx_header else None
    if system_note:
        from app.local.llm_adapter import _build_system_prompt

        system_prompt = f"{system_prompt or _build_system_prompt()} {system_note}"

    route = pre_router.route(user_text or "", settings) if settings else None
    if route and route[0] == "clock":
        log.info("reply_chain: pre-router clock answer meeting_id=%s", meeting_id)
        return ReplySource(_one(route[1]), source="clock")
    if route and route[0] == "search":
        return ReplySource(
            _search_sentences(app, meeting_id, user_text, route[1], turn_id=turn_id,
                              model=model, history=history, system_prompt=system_prompt),
            source="search", lead_in=_FILLER_PHRASE)

    log.info("reply_chain: querying %s model=%s history_turns=%d meeting_id=%s",
             "pre_stream" if pre_stream is not None else "llm", model,
             len(history) // 2 if history else 0, meeting_id)
    if pre_stream is not None:
        return ReplySource(_sentence_stream_from_tokens(pre_stream, _OLLAMA_TIMEOUT_S),
                           source="pre_stream")
    return ReplySource(_sentence_stream(user_text, model, _OLLAMA_TIMEOUT_S, history=history,
                                        system_prompt=system_prompt), source="llm")


async def _one(sentence: str) -> AsyncIterator[str]:
    yield sentence


async def _search_sentences(app, meeting_id: str, user_text: str, query: str, *,
                            turn_id: str | None, model: str, history: list[dict] | None,
                            system_prompt: str | None) -> AsyncIterator[str]:
    from app.local.tools.web_search import search as web_search

    pool = app.state.store.pool
    log.info("reply_chain: search route meeting_id=%s", meeting_id)
    t_search = time.monotonic()
    search_result = ""
    search_error: str | None = None
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_S):
            search_result = await web_search(query)
    except Exception as exc:
        search_error = str(exc)
        log.warning("reply_chain: web_search failed: %s", type(exc).__name__)
    search_dur_ms = int((time.monotonic() - t_search) * 1000)

    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO tool_calls"
                " (id, turn_id, meeting_id, tool, query, engine, result_count,"
                "  duration_ms, error, called_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (str(uuid.uuid4()), turn_id, meeting_id, "web_search", query,
                 "searxng", len([ln for ln in search_result.splitlines() if ln.startswith("1.")
                                 or ln[:2] in ("2.", "3.", "4.", "5.")]),
                 search_dur_ms, search_error, _now_z()),
            )
            await conn.commit()
    except Exception:
        log.debug("reply_chain: tool_calls insert failed (non-critical)")

    if search_error or "[web_search: unavailable" in search_result:
        yield _SEARCH_UNAVAILABLE_PHRASE
        return
    search_system = (
        f"{system_prompt or ''}\n\n"
        f"Use the following search results to answer the user's question. "
        f"Do not reproduce URLs. Summarise in 1-2 sentences.\n\n"
        f"{search_result}"
    ).strip()
    async for sentence in _sentence_stream(user_text, model, _OLLAMA_TIMEOUT_S,
                                           history=history, system_prompt=search_system):
        yield sentence


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
    from app.local import gate, meeting_bus

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
            if rec.gate_pre_written:
                await conn.execute(
                    "UPDATE gate_decisions"
                    " SET decision=%s, category=%s, decided_at=%s WHERE id=%s",
                    (decision, category, decided_z, rec.gate_id),
                )
            else:
                await conn.execute(
                    "INSERT INTO gate_decisions"
                    " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (rec.gate_id, rec.reply_id, meeting_id, rec.mode,
                     rec.user_text[:1000], decision, category, decided_z),
                )
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
