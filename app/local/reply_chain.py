"""Reply chain: LLM -> gate -> TTS -> playback for wake sessions.

Called as a background task after turn close in wake_loop._run_session.

Pipeline per turn:
  1. LLM streams tokens; split into sentences at punctuation boundaries.
  2. Gate each sentence as it arrives (per-sentence check, mode-aware).
  3. If gate passes: synthesize with Kokoro + play. First audio starts
     before LLM finishes generating (V1-P01 clause-level streaming).
  4. If gate refuses any sentence: cancel playback, play fixed refusal phrase.
  5. Write rows: turns, replies, gate_decisions, latency_ledger. Behaviour log.
  6. Publish reply event over meeting_bus.

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
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_OLLAMA_TIMEOUT_S  = 30.0   # llama3.2 first-token <3s warm, ~8s cold; 30s covers first-time load
_FALLBACK_PHRASE   = "I'm sorry, I didn't catch that. Could you say it again?"
_THINKING_PHRASE   = "hm"
_SENTENCE_RE       = re.compile(r'[.!?]["\')\]]*\s+')


async def _sentence_stream(
    user_text: str,
    model: str,
    timeout_s: float,
) -> AsyncIterator[str]:
    """Stream LLM tokens and yield complete sentences at punctuation boundaries."""
    from app.local.llm_adapter import stream_reply

    buf = ""
    try:
        async with asyncio.timeout(timeout_s):
            async for token in stream_reply(user_text, model=model):
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


async def warm_phrase_cache() -> None:
    """Pre-render all fixed gate phrases at startup for zero-latency refusals (V1-P03)."""
    from app.local.tts_engine import prerender_phrase
    from app.local.gate import (
        DEFAULT_REFUSAL_PHRASE,
        CLASSROOM_REFUSAL_PHRASE,
        SENIORCARE_ESCALATION_PHRASE,
    )
    for phrase in (_THINKING_PHRASE, DEFAULT_REFUSAL_PHRASE, CLASSROOM_REFUSAL_PHRASE, SENIORCARE_ESCALATION_PHRASE):
        await prerender_phrase(phrase)


def _now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def run_reply(
    app,
    meeting_id: str,
    user_text: str,
    turn_end_ms: int,
    turn_n: int = 1,
    mode: str = "default",
    asr_final_ms: int | None = None,
) -> None:
    """Background task: generate and play spoken reply for one wake turn.

    Writes turns, replies, gate_decisions, latency_ledger rows.
    Publishes reply event on meeting_bus.
    Errors are logged and suppressed; the wake loop does not depend on this completing.
    """
    from app.local.tts_engine import synthesize_and_play, play_prerendered, prerender_phrase
    from app.local import meeting_bus, gate
    from app.local.audio_engine import monotonic_ms

    pool = app.state.store.pool
    settings = getattr(app.state, "settings", None)
    audio_engine = getattr(app.state, "audio_engine", None)
    model = getattr(settings, "voice_reply_model", "llama3.2")
    if mode == "default":
        mode = getattr(settings, "voice_mode", "default")

    turn_id   = str(uuid.uuid4())
    reply_id  = str(uuid.uuid4())
    ledger_id = str(uuid.uuid4())
    gate_id   = str(uuid.uuid4())
    started_z = _now_z()

    # ── 1. Write turn row ────────────────────────────────────────────────────
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO turns (id, meeting_id, turn_n, user_text, started_at)"
                " VALUES (%s, %s, %s, %s, %s)",
                (turn_id, meeting_id, turn_n, user_text or "", started_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write turn row meeting_id=%s", meeting_id)

    # ── 1b. Thinking cue: play immediately so speaker isn't silent while LLM warms up ─
    if audio_engine is not None and user_text.strip():
        try:
            thinking_path = await prerender_phrase(_THINKING_PHRASE)
            if thinking_path is not None:
                await play_prerendered(thinking_path, audio_engine)
        except Exception:
            log.debug("reply_chain: thinking cue failed (non-critical)")

    # ── 1c. Write preliminary gate_decisions row before any playback (SC-G3) ──
    # SC-G3 requires a gate_decisions row to exist BEFORE playback starts.
    # Write 'pending' now; UPDATE to the real decision after the LLM loop.
    gate_pre_written = False
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO gate_decisions"
                " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (gate_id, reply_id, meeting_id, mode,
                 (user_text or "")[:1000], "pending", "pending", started_z),
            )
            await conn.commit()
        gate_pre_written = True
    except Exception:
        log.exception("reply_chain: failed to pre-write gate_decisions meeting_id=%s", meeting_id)

    # ── 2-5. LLM stream → gate per sentence → Kokoro → play ─────────────────
    # Gate each sentence as it arrives. First audio starts before LLM finishes
    # generating (V1-P01 clause-level streaming). If any sentence is refused,
    # cancel playback and play the fixed refusal phrase.

    llm_first_token_ms: int | None = None
    llm_request_sent_ms: int | None = None
    tts_first_audio_ms: int | None = None
    playback_end_ms: int | None = None
    reply_text = _FALLBACK_PHRASE
    decision = "pass"
    gate_category = "ok"
    all_sentences: list[str] = []

    if not user_text.strip():
        log.info("reply_chain: empty user_text, using fallback meeting_id=%s", meeting_id)
        decision = "refuse"
        gate_category = "empty"
        reply_text = gate.refusal_phrase("empty", mode)
    else:
        llm_request_sent_ms = monotonic_ms()
        try:
            async for sentence in _sentence_stream(user_text, model, _OLLAMA_TIMEOUT_S):
                if llm_first_token_ms is None:
                    llm_first_token_ms = monotonic_ms()

                # Gate each sentence before speaking it
                s_decision, s_category = gate.check(user_text, sentence, mode=mode)
                if s_decision != "pass":
                    decision = "refuse"
                    gate_category = s_category
                    reply_text = gate.refusal_phrase(s_category, mode)
                    log.info("reply_chain: gate refused sentence category=%s mode=%s",
                             s_category, mode)
                    if audio_engine is not None and all_sentences:
                        audio_engine.cancel_playback()
                    break

                all_sentences.append(sentence)

                if audio_engine is not None:
                    try:
                        s_dur = await synthesize_and_play(sentence, audio_engine)
                        if tts_first_audio_ms is None:
                            tts_first_audio_ms = monotonic_ms()
                            log.info("reply_chain: first audio at +%d ms from turn_end",
                                     tts_first_audio_ms - turn_end_ms)
                    except Exception:
                        log.exception("reply_chain: TTS failed for sentence meeting_id=%s",
                                      meeting_id)
        except Exception:
            log.exception("reply_chain: LLM stream failed meeting_id=%s", meeting_id)

        if not all_sentences and decision == "pass":
            log.warning("reply_chain: LLM produced no sentences, using fallback")
            reply_text = _FALLBACK_PHRASE
        elif decision == "pass":
            reply_text = " ".join(all_sentences)

    decided_z = _now_z()

    # Gate refusal: play fixed phrase (pre-rendered, zero synthesis latency)
    if decision == "refuse" and audio_engine is not None:
        try:
            phrase_path = await prerender_phrase(reply_text)
            if phrase_path is not None:
                if tts_first_audio_ms is None:
                    tts_first_audio_ms = monotonic_ms()
                await play_prerendered(phrase_path, audio_engine)
            else:
                if tts_first_audio_ms is None:
                    tts_first_audio_ms = monotonic_ms()
                await synthesize_and_play(reply_text, audio_engine)
        except Exception:
            log.exception("reply_chain: refusal playback failed meeting_id=%s", meeting_id)

    if audio_engine is not None:
        playback_end_ms = monotonic_ms()

    log.info("reply_chain: reply_len=%d decision=%s meeting_id=%s",
             len(reply_text), decision, meeting_id)

    # ── 4. Write replies row; UPDATE gate_decisions with final decision ──────
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (reply_id, meeting_id, turn_n, reply_text, model, gate_category, decided_z),
            )
            if gate_pre_written:
                await conn.execute(
                    "UPDATE gate_decisions"
                    " SET decision=%s, category=%s, decided_at=%s WHERE id=%s",
                    (decision, gate_category, decided_z, gate_id),
                )
            else:
                await conn.execute(
                    "INSERT INTO gate_decisions"
                    " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (gate_id, reply_id, meeting_id, mode,
                     (user_text or "")[:1000], decision, gate_category, decided_z),
                )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write reply/gate rows meeting_id=%s", meeting_id)

    gate.write_behaviour_log(
        meeting_id=meeting_id,
        turn_n=turn_n,
        mode=mode,
        decision=decision,
        category=gate_category,
    )

    # ── 6. Write played_at + latency ledger ──────────────────────────────────
    try:
        async with pool.connection() as conn:
            played_z = _now_z()
            await conn.execute(
                "UPDATE replies SET played_at=%s WHERE id=%s",
                (played_z, reply_id),
            )
            await conn.execute(
                "UPDATE turns SET ended_at=%s WHERE id=%s",
                (played_z, turn_id),
            )
            await conn.execute(
                "INSERT INTO latency_ledger"
                " (id, meeting_id, turn_n, turn_end_ms, asr_final_ms,"
                "  llm_request_sent_ms, llm_first_token_ms,"
                "  tts_first_audio_ms, playback_end_ms, recorded_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (ledger_id, meeting_id, turn_n, turn_end_ms, asr_final_ms,
                 llm_request_sent_ms, llm_first_token_ms,
                 tts_first_audio_ms, playback_end_ms, played_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write ledger meeting_id=%s", meeting_id)

    # ── 7. Publish reply event ────────────────────────────────────────────────
    try:
        await meeting_bus.publish(meeting_id, "reply", {
            "id": reply_id,
            "text": reply_text,
            "turn_n": turn_n,
            "gate": decision,
            "category": gate_category,
        })
    except Exception:
        log.warning("reply_chain: failed to publish reply event meeting_id=%s", meeting_id)
