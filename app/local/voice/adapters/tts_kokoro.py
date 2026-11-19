"""Talker over Kokoro-ONNX, with the macOS say fallback and the fixed-phrase cache (tts_engine)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.local import tts_engine

THINKING_PHRASE = "hm"     # the reply chain's thinking cue; rendering it warms the phrase cache


def _pcm(samples) -> np.ndarray:
    if samples is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


class KokoroTalker:
    sample_rate = tts_engine.PLAYBACK_RATE_HZ

    def __init__(self, get_voice_speed: Callable[[], tuple[str, float]] | None = None,
                 pronounce: Callable[[str], str] | None = None) -> None:
        # get_voice_speed() returns (voice_id, speed); empty voice_id uses the engine default.
        # pronounce(text) is the text the voice is given (voice/expression.py): it runs here,
        # after the gate and immediately before synthesis, so what was shown and stored keeps
        # the original spelling.
        self._get_voice_speed = get_voice_speed or (lambda: ("", 1.0))
        self._pronounce = pronounce or (lambda text: text)

    async def synthesize(self, text: str) -> np.ndarray:
        voice, speed = self._get_voice_speed()
        text = self._pronounce(text)
        if voice:
            return _pcm(await tts_engine.synthesize(text, voice=voice, speed=speed))
        return _pcm(await tts_engine.synthesize(text, speed=speed))

    async def line(self, text: str) -> np.ndarray:
        """A fixed line the assistant says on its own (a thinking line, a farewell), in the
        session's voice, from the phrase cache. The pronunciation table applies, as it does to
        replies; the line itself is what the gate passed."""
        voice, speed = self._get_voice_speed()
        return _pcm(await tts_engine.load_line(self._pronounce(text), voice=voice, speed=speed))

    async def fixed(self, phrase: str) -> np.ndarray:
        # Pre-rendered with say; when the cache cannot render it, synthesise it live instead.
        samples = await tts_engine.load_phrase(phrase)
        if samples is None:
            samples = await tts_engine.synthesize(phrase)
        return _pcm(samples)

    async def warm(self) -> None:
        await tts_engine.prerender_phrase(THINKING_PHRASE)
