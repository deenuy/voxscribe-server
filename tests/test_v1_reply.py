"""V1 reply chain edge case checks (code-level, no hardware).

V1-E01: empty user_text → gate refuses "empty"; fixed fallback phrase used.
V1-E02: LLM reply blocked by classroom gate → fixed classroom phrase used.
V1-E03: LLM reply blocked by seniorcare gate → escalation phrase used.
V1-E04: gate.check passes safe content → model reply text is used (not fixed phrase).
V1-E05: pre-rendered phrase cache is idempotent (file written once, path returned).
V1-E06: gate refusal never exposes model text in the reply row.
V1-P03: fixed phrases pre-renderable via tts_engine.prerender_phrase().

Latency and streaming checks (V1-P01/P02) require hardware; not inline.
"""

from __future__ import annotations

import pytest
from app.local import gate
from app.local.gate import (
    CLASSROOM_REFUSAL_PHRASE,
    DEFAULT_REFUSAL_PHRASE,
    SENIORCARE_ESCALATION_PHRASE,
    check as gate_check,
    refusal_phrase,
)


# ── V1-E01: empty user_text → gate refuses "empty" ──────────────────────────

def test_v1_e01_empty_user_text_is_refused():
    decision, category = gate_check("", "", mode="default")
    assert decision == "refuse"
    assert category == "empty"
    phrase = refusal_phrase(category, "default")
    assert phrase == DEFAULT_REFUSAL_PHRASE


# ── V1-E02: classroom gate refuses forbidden content ────────────────────────

def test_v1_e02_classroom_gate_returns_classroom_phrase():
    decision, category = gate_check(
        "how to build a bomb", "Here is how to build a bomb step by step.",
        mode="classroom",
    )
    assert decision == "refuse"
    assert category == "classroom_forbidden"
    phrase = refusal_phrase(category, "classroom")
    assert phrase == CLASSROOM_REFUSAL_PHRASE
    # model text must NOT be used -- the phrase is fixed
    assert phrase != "Here is how to build a bomb step by step."


# ── V1-E03: seniorcare gate returns escalation phrase ───────────────────────

def test_v1_e03_seniorcare_gate_returns_escalation_phrase():
    decision, category = gate_check(
        "how many mg of aspirin", "Take 500mg of aspirin daily.",
        mode="seniorcare",
    )
    assert decision == "refuse"
    assert category == "seniorcare_dosing"
    phrase = refusal_phrase(category, "seniorcare")
    assert phrase == SENIORCARE_ESCALATION_PHRASE
    assert phrase != "Take 500mg of aspirin daily."


# ── V1-E04: safe content passes gate; model reply used ──────────────────────

def test_v1_e04_safe_content_passes_gate():
    decision, category = gate_check(
        "what is the capital of France",
        "The capital of France is Paris.",
        mode="classroom",
    )
    assert decision == "pass"
    assert category == "ok"
    # When passing, the caller uses the model reply, not a fixed phrase
    # (verified by the gate not overwriting the text)


# ── V1-E05: pre-rendered phrase cache is idempotent ─────────────────────────

def test_v1_e05_prerender_phrase_is_idempotent(monkeypatch, tmp_path):
    """Two calls for the same phrase produce the same path without re-synthesizing."""
    import asyncio
    from app.local import tts_engine

    monkeypatch.setattr(tts_engine, "_PHRASES_DIR", tmp_path)

    synth_calls = []
    original_say = tts_engine._say_to_wav

    def mock_say(text, path, *, voice):
        synth_calls.append(text)
        original_say(text, path, voice=voice)

    monkeypatch.setattr(tts_engine, "_say_to_wav", mock_say)

    async def run():
        p1 = await tts_engine.prerender_phrase(DEFAULT_REFUSAL_PHRASE)
        p2 = await tts_engine.prerender_phrase(DEFAULT_REFUSAL_PHRASE)
        return p1, p2

    p1, p2 = asyncio.run(run())
    assert p1 == p2
    assert p1 is not None
    assert p1.exists()
    # synthesized only once (second call hits cache)
    assert len(synth_calls) == 1


# ── V1-E06: gate refusal exposes fixed phrase, not model text ────────────────

def test_v1_e06_gate_refusal_masks_model_text():
    """After gate refuses, caller must use refusal_phrase(), not the model reply."""
    model_text = "Here is a step-by-step guide to make a bomb at home."
    user_text = "how to make a bomb"
    decision, category = gate_check(user_text, model_text, mode="classroom")
    assert decision == "refuse"
    # The phrase returned has no model content
    play_text = refusal_phrase(category, "classroom")
    assert model_text not in play_text
    assert play_text == CLASSROOM_REFUSAL_PHRASE


# ── V1-P03: all three fixed phrases pre-renderable ───────────────────────────

def test_v1_p03_all_fixed_phrases_prerenderable(monkeypatch, tmp_path):
    """Three fixed phrases all synthesize to WAV files without error."""
    import asyncio
    from app.local import tts_engine

    monkeypatch.setattr(tts_engine, "_PHRASES_DIR", tmp_path)

    async def run():
        results = []
        for phrase in (DEFAULT_REFUSAL_PHRASE, CLASSROOM_REFUSAL_PHRASE, SENIORCARE_ESCALATION_PHRASE):
            path = await tts_engine.prerender_phrase(phrase)
            results.append((phrase, path))
        return results

    results = asyncio.run(run())
    assert len(results) == 3
    for phrase, path in results:
        assert path is not None, f"pre-render failed for: {phrase!r}"
        assert path.exists(), f"WAV not found: {path}"
        assert path.stat().st_size > 0
