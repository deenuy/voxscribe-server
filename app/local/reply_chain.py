"""Reply chain: LLM -> gate -> TTS -> playback for wake sessions.

Called as a background task after turn close in wake_loop._run_session.

Pipeline per turn:
  1. Call LLM (Ollama streaming) with the user's transcript text.
  2. Collect the full reply (gate needs the complete text before playback).
  3. Gate check (v0 stub: always pass unless empty; G1 satisfied by writing gate_decisions row).
  4. Write rows: turns, replies, gate_decisions, latency_ledger.
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
- Gate v0 always passes; full gate policy wired in R9 (C9).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

log = logging.getLogger("app.local.reply_chain")

_FALLBACK_PHRASE = "I didn't catch that. Could you say it again?"
_OLLAMA_TIMEOUT_S = 10.0


def _now_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _gate_v0(text: str) -> tuple[str, str]:
    """Gate v0 stub. Returns (decision, category). Always passes non-empty text."""
    if not text.strip():
        return "refuse", "empty"
    return "pass", "ok"


async def run_reply(
    app,
    meeting_id: str,
    user_text: str,
    turn_end_ms: int,
    turn_n: int = 1,
) -> None:
    """Background task: generate and play spoken reply for one wake turn.

    Writes turns, replies, gate_decisions, latency_ledger rows.
    Publishes reply event on meeting_bus.
    Errors are logged and suppressed; the wake loop does not depend on this completing.
    """
    from app.local.llm_adapter import collect_reply
    from app.local.tts_engine import synthesize_and_play
    from app.local import meeting_bus
    from app.local.audio_engine import monotonic_ms

    pool = app.state.store.pool
    settings = getattr(app.state, "settings", None)
    audio_engine = getattr(app.state, "audio_engine", None)
    model = getattr(settings, "voice_reply_model", "llama3.2")

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

    # ── 3. Gate ──────────────────────────────────────────────────────────────
    decision, category = _gate_v0(reply_text)
    if decision != "pass":
        log.info("reply_chain: gate refused category=%s meeting_id=%s", category, meeting_id)
        reply_text = _FALLBACK_PHRASE
        decision = "refuse"

    gate_category = category if decision == "pass" else decision
    decided_z = _now_z()

    # ── 4. Write replies + gate_decisions rows ───────────────────────────────
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
                (gate_id, reply_id, meeting_id, "default",
                 (user_text or "")[:1000], decision, category, decided_z),
            )
            await conn.commit()
    except Exception:
        log.exception("reply_chain: failed to write reply/gate rows meeting_id=%s", meeting_id)

    # ── 5. TTS + playback ────────────────────────────────────────────────────
    tts_first_audio_ms: int | None = None
    playback_end_ms: int | None = None

    if audio_engine is not None:
        try:
            tts_start_ms = monotonic_ms()
            tts_first_audio_ms = tts_start_ms
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
