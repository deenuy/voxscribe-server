"""Reply chain: the reply model, the thinker's search and the reply bookkeeping for wake sessions.

The voice host's reply stage (voxscribe.session.stages.reply) does the speaking: it gates every
sentence before synthesis and plays what passed. This module supplies the two ends of it,
through the Delegate and SessionStore adapters in app/local/voice/adapters:

  begin_reply()       write the turns row and pre-write the replies row. The store then writes
                      the gate's heard record (gate_records.record_heard), so a gate record
                      exists before anything is played (G1)
  talker_sentences()  the talker: the reply model's sentences for one turn, split at
                      punctuation. It has no tools
  search_answer()     the thinker: `web.search` through the tool runtime on the delegate lane
                      (app/local/voice/delegate_lane.py); its audit row carries the task id
  finish_reply()      update the replies row, write the behaviour log (G6), played_at and
                      the latency ledger, then publish the reply event over meeting_bus. The
                      gate's records for the reply are not written here: they have their own
                      transaction (gate_records.flush_reply), so a failed reply row update
                      does not take them with it

Latency budget (idle MacBook): turn_end to first_audio under 1.5 s.
TTS: Kokoro-ONNX primary, macOS say fallback (see tts_engine.py).
LLM: Ollama streaming (see llm_adapter.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_OLLAMA_TIMEOUT_S  = 30.0   # llama3.2 first-token <3s warm, ~8s cold; 30s covers first-time load
_FALLBACK_PHRASE   = "I'm sorry, I didn't catch that. Could you say it again?"
_THINKING_PHRASE   = "hm"
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
    heard_written: bool = False       # the heard row is in the ledger
    outcome_written: bool = False     # the journal and the outcome row are in the ledger


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


_ENGINE_IDENTITY = "You are a helpful voice assistant."


def _identity(persona) -> str:
    """The prompt's first sentence. `persona` is the session's SessionSettings (or anything
    with assistant_name and persona_label); both are validated labels, never free text."""
    name = getattr(persona, "assistant_name", "") or ""
    if not name:
        return _ENGINE_IDENTITY
    label = getattr(persona, "persona_label", "") or "assistant"
    return f"You are {name}, a helpful voice {label}."


def _owner_block(persona) -> str:
    """The owner's system prompt, after the engine's instructions (profile_store bounds and
    frames it). Empty in senior care and classroom, where the stored prompt is clamped."""
    from app.local.voice.profile_store import owner_preferences_block

    return owner_preferences_block(getattr(persona, "system_prompt", "") or "")


# The spoken register (VOXSCRIBE_VOICE_EXPRESSIVENESS=natural, PL66): short rules and a few
# examples, good and bad, for a small model. The length bound is the plain register's. What the
# model still gets wrong is corrected after it, deterministically (app/local/voice/expression.py).
SPOKEN_REGISTER = """You are talking out loud with a person. Sound like a person in conversation:
- Answer first. One or two short sentences, under 30 words.
- Use contractions: I'm, it's, you're, don't.
- When it fits, react briefly to what they said before the answer: "Oh, nice." "Ah, got it." \
"Hmm, good question." Not every time, and never the same way twice in a row.
- Don't repeat their question. Don't thank them for asking, don't say you're here to help, \
don't offer more help at the end.
- No markdown, no lists, no headers.
- You are an AI. If asked whether you are a person, say you're an AI. Never claim a body, a \
past or feelings as facts.
Examples:
Person: How are you? Can you hear me?
Good: Doing well, thanks. And yes, I can hear you fine.
Bad: I'm functioning properly, thank you for asking. Yes, I can hear you, and I'm ready to \
assist you with any questions or topics you'd like to discuss.
Person: I just got back from a run.
Good: Oh, nice. How far did you go?
Bad: That's great to hear! Running is an excellent form of exercise. Is there anything else I \
can help you with?
Person: What's the capital of Australia?
Good: Canberra. A lot of people guess Sydney.
Bad: That's a great question! The capital of Australia is Canberra.
Person: Are you a real person?
Good: No, I'm an AI. Ask me anything, though.
Bad: Yes, I'm right here with you."""


def _voice_system_prompt(settings, persona=None, expressiveness: str = "plain") -> str:
    """The context header from the OS clock at turn start (not session start), then the voice
    rules. `persona` names the assistant (the session's settings; None: the engine's wording).
    `expressiveness` natural gives the spoken register in place of the plain rules."""
    from app.local.context import current_context_header
    from app.local.llm_adapter import _build_system_prompt
    from app.local.voice.expression import NATURAL

    ctx_header = current_context_header(settings) if settings else ""
    if expressiveness == NATURAL:
        if not ctx_header:
            # The plain prompt's own date line stands in for the header, as it does in plain.
            ctx_header = _build_system_prompt().split(". ", 2)[1].strip() + "."
        return f"{ctx_header} {_identity(persona)}\n{SPOKEN_REGISTER}"
    if not ctx_header:
        return _build_system_prompt().replace(_ENGINE_IDENTITY, _identity(persona), 1)
    return (
        f"{ctx_header} "
        f"{_identity(persona)} "
        "Reply in 1-2 short sentences only. "
        "Be direct and conversational. No markdown, no lists, no headers. "
        "Keep replies under 30 words."
    )

def talker_system_prompt(settings, persona=None, expressiveness: str = "plain") -> str:
    """The talker's prompt: the voice prompt plus the instruction to hand anything that needs
    a tool to the delegate lane, then the owner's preferences when there are any. The talker
    itself has no tools."""
    from app.local.voice.delegate_lane import TALKER_DELEGATE_INSTRUCTION

    voice = _voice_system_prompt(settings, persona, expressiveness)
    prompt = f"{voice}{chr(10) if expressiveness == 'natural' else ' '}{TALKER_DELEGATE_INSTRUCTION}"
    owner = _owner_block(persona)
    return f"{prompt}\n\n{owner}" if owner else prompt


def talker_sentences(
    app,
    meeting_id: str,
    user_text: str,
    *,
    history: list[dict] | None = None,
    pre_stream: AsyncIterator[str] | None = None,
    system_note: str | None = None,
    system_prefix: str | None = None,
    persona=None,
    expressiveness: str = "plain",
) -> ReplySource:
    """The talker's reply to one turn, as sentences. No tools, nothing is spoken here.

    The speculative stream when the host started one, else the reply model. Routing (clock
    answers, delegation to the thinker) is the Delegate adapter's job and happens before this.
    system_note  appended to the system prompt for this turn (the barge-in note, T13).
    system_prefix  prepended to the system prompt (memory block T19 + control suffix T15).
    persona  the session's assistant settings (name, persona label, system prompt).
    expressiveness  plain (the engine's prompt) or natural (the spoken register, PL66).
    """
    settings = getattr(app.state, "settings", None)
    model, _mode = reply_settings(settings)
    log.info("reply_chain: querying %s model=%s history_turns=%d meeting_id=%s",
             "pre_stream" if pre_stream is not None else "llm", model,
             len(history) // 2 if history else 0, meeting_id)
    if pre_stream is not None:
        return ReplySource(_sentence_stream_from_tokens(pre_stream, _OLLAMA_TIMEOUT_S),
                           source="pre_stream")
    prompt = talker_system_prompt(settings, persona, expressiveness)
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


async def search_answer(app, meeting_id: str | None, request: str, query: str, *,
                        task_id: str, turn_id: str | None = None,
                        history: list[dict] | None = None, persona=None,
                        expressiveness: str = "plain") -> str:
    """The thinker's web search: `web.search` through the tool runtime, which checks the registry
    and the network decision (PL41) before anything is sent and writes the call's audit row
    with the task id; then the reply model summarises the results in one or two sentences. Runs
    on the delegate lane, never in a turn.

    Returns the text to deliver: the summary, or the record's scripted line when the search was
    refused, failed or timed out. The lane bounds the whole call with THINKER_TIMEOUT_S; the
    record bounds the search.
    """
    from app.local.tools import builtin, runtime, web_search
    from app.local.voice.adapters.scope import scoped_pool

    settings = getattr(app.state, "settings", None)
    model, _mode = reply_settings(settings)
    # Called from the voice delegate lane: the pool comes through the voice scope adapter, which
    # refuses (before the search runs) outside a tenant's scope with isolation on, or in a session
    # that is not the tenant's.
    scoped_pool(app)
    log.info("reply_chain: thinker search task_id=%s meeting_id=%s", task_id, meeting_id)
    outcome = await runtime.call(
        app, builtin.WEB_SEARCH, {"query": query},
        runtime.CallContext(session_id=meeting_id, turn_id=turn_id, task_id=task_id),
        web_search.execute)
    if not outcome.executed:
        return outcome.line
    search_system = (
        f"{_voice_system_prompt(settings, persona, expressiveness)}\n\n"
        f"Use the following search results to answer the user's question. "
        f"Do not reproduce URLs. Summarise in 1-2 sentences.\n\n"
        f"{outcome.result['text']}"
    ).strip()
    owner = _owner_block(persona)
    if owner:
        search_system = f"{search_system}\n\n{owner}"
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
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write reply row meeting_id=%s", meeting_id)

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
