"""Reply chain: LLM -> gate -> TTS -> playback for wake sessions.

Called as a background task after turn close in wake_loop._run_session.

Pipeline per turn:
  1. Call LLM (Ollama streaming) with the user's transcript text.
  2. Collect the full reply (gate needs the complete text before playback).
  3. Gate check (gate.check): mode-aware, injection-resistant regex. G1-G5 satisfied.
  4. Write rows: turns, replies, gate_decisions, latency_ledger. Behaviour log (G6).
  5. TTS and play via tts_engine.synthesize_and_play.
  6. Publish reply event over meeting_bus.

Latency budget (idle MacBook): turn_end to first_audio under 1.5 s.
This is P1 bar (adult); seniorcare bar is 1.5 s.

TTS is macOS 'say' stub (see tts_engine.py).
LLM is Ollama streaming (see llm_adapter.py).

LIMITS:
- Streaming TTS (first clause while LLM generates rest) deferred to P4.
  R6 collects the full reply before synthesizing.
- Playback reference for echo cancellation wired in R7 / P4.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_OLLAMA_TIMEOUT_S = 10.0


async def warm_phrase_cache() -> None:
    """Pre-render all fixed gate phrases at startup for zero-latency refusals (V1-P03)."""
    from app.local.tts_engine import prerender_phrase
    from app.local.gate import (
        DEFAULT_REFUSAL_PHRASE,
        CLASSROOM_REFUSAL_PHRASE,
        SENIORCARE_ESCALATION_PHRASE,
    )
    for phrase in (DEFAULT_REFUSAL_PHRASE, CLASSROOM_REFUSAL_PHRASE, SENIORCARE_ESCALATION_PHRASE):
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
) -> None:
    """Background task: generate and play spoken reply for one wake turn.

    Writes turns, replies, gate_decisions, latency_ledger rows.
    Publishes reply event on meeting_bus.
    Errors are logged and suppressed; the wake loop does not depend on this completing.
    """
    from app.local.llm_adapter import collect_reply
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

    # ── 2. LLM ───────────────────────────────────────────────────────────────
    llm_first_token_ms: int | None = None
    reply_text = _FALLBACK_PHRASE

    if user_text.strip():
        try:
            llm_start_ms = monotonic_ms()
            reply_text = await collect_reply(user_text, model=model,
                                             timeout_s=_OLLAMA_TIMEOUT_S)
            llm_first_token_ms = llm_start_ms  # best approximation (collected, not streamed)
            log.info("reply_chain: LLM reply len=%d model=%s", len(reply_text), model)
        except Exception:
            log.exception("reply_chain: LLM failed, using fallback meeting_id=%s", meeting_id)
            reply_text = _FALLBACK_PHRASE
    else:
        log.info("reply_chain: empty user_text, using fallback meeting_id=%s", meeting_id)

    # ── 3. Gate (G2-G5) ──────────────────────────────────────────────────────
    decision, category = gate.check(user_text or "", reply_text, mode=mode)
    if decision != "pass":
        log.info("reply_chain: gate refused category=%s mode=%s meeting_id=%s",
                 category, mode, meeting_id)
        reply_text = gate.refusal_phrase(category, mode)
        decision = "refuse"

    gate_category = category
    decided_z = _now_z()

    # ── 4. Write replies + gate_decisions rows + behaviour log (G1, G6) ───────
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO replies (id, meeting_id, turn_n, text, model, gate_category, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (reply_id, meeting_id, turn_n, reply_text, model, gate_category, decided_z),
            )
            await conn.execute(
                "INSERT INTO gate_decisions"
                " (id, reply_id, meeting_id, mode, input_text, decision, category, decided_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (gate_id, reply_id, meeting_id, mode,
                 (user_text or "")[:1000], decision, category, decided_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write reply/gate rows meeting_id=%s", meeting_id)

    gate.write_behaviour_log(
        meeting_id=meeting_id,
        turn_n=turn_n,
        mode=mode,
        decision=decision,
        category=category,
    )

    # ── 5. TTS + playback (pre-rendered for gate refusals) ──────────────────
    tts_first_audio_ms: int | None = None
    playback_end_ms: int | None = None

    if audio_engine is not None:
        try:
            tts_start_ms = monotonic_ms()
            tts_first_audio_ms = tts_start_ms
            if decision == "refuse":
                # Use pre-rendered WAV: zero synthesis time for fixed phrases (V1-P03)
                phrase_path = await prerender_phrase(reply_text)
                if phrase_path is not None:
                    duration_s = await play_prerendered(phrase_path, audio_engine)
                else:
                    duration_s = await synthesize_and_play(reply_text, audio_engine)
            else:
                duration_s = await synthesize_and_play(reply_text, audio_engine)
            playback_end_ms = monotonic_ms()
            log.info("reply_chain: played duration=%.2f s meeting_id=%s", duration_s, meeting_id)
        except Exception:
            log.exception("reply_chain: TTS/playback failed meeting_id=%s", meeting_id)
    else:
        log.warning("reply_chain: no audio_engine, skipping TTS meeting_id=%s", meeting_id)

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
                " (id, meeting_id, turn_n, turn_end_ms, llm_first_token_ms,"
                "  tts_first_audio_ms, playback_end_ms, recorded_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (ledger_id, meeting_id, turn_n, turn_end_ms, llm_first_token_ms,
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
            "category": category,
        })
    except Exception:
        log.warning("reply_chain: failed to publish reply event meeting_id=%s", meeting_id)
