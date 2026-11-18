"""Talker over Kokoro-ONNX, with the macOS say fallback and the fixed-phrase cache (tts_engine)."""
from __future__ import annotations

import numpy as np

from app.local import tts_engine

THINKING_PHRASE = "hm"     # the reply chain's thinking cue; rendering it warms the phrase cache


def _pcm(samples) -> np.ndarray:
    if samples is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


class KokoroTalker:
    sample_rate = tts_engine.PLAYBACK_RATE_HZ

    async def synthesize(self, text: str) -> np.ndarray:
        return _pcm(await tts_engine.synthesize(text))

    async def fixed(self, phrase: str) -> np.ndarray:
        # Pre-rendered with say; when the cache cannot render it, synthesise it live instead.
        samples = await tts_engine.load_phrase(phrase)
        if samples is None:
            samples = await tts_engine.synthesize(phrase)
        return _pcm(samples)

    async def warm(self) -> None:
        await tts_engine.prerender_phrase(THINKING_PHRASE)
