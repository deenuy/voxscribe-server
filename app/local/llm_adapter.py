"""LLM adapter: streaming Ollama client for spoken replies.

Streams token-by-token from Ollama's /api/generate endpoint.
Default model: llama3.2 (3B class, per VOICE-AUTOPILOT.md model split).
Configurable via app.state.settings.voice_reply_model.

Ollama must be running locally on port 11434. The wake loop ensures
that only short replies (2-3 sentences max) are requested by the system
prompt; latency budget is 800 ms to first audio on idle machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

log = logging.getLogger("app.local.llm_adapter")

_OLLAMA_URL   = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "gemma4:12b"
_REQUEST_TIMEOUT = 30.0

_SYSTEM_PROMPT = """You are a helpful voice assistant. Reply in 1-2 short sentences only.
Be direct and conversational. No markdown, no lists, no headers.
Keep replies under 30 words."""


async def stream_reply(
    user_text: str,
    *,
    model: str = _DEFAULT_MODEL,
    system_prompt: str = _SYSTEM_PROMPT,
) -> AsyncIterator[str]:
    """Stream reply tokens from Ollama.

    Yields text chunks as they arrive. Each chunk may be a partial word.
    The caller is responsible for splitting into sentences for TTS.

    Raises RuntimeError if Ollama is not available.
    """
    payload = {
        "model": model,
        "prompt": user_text,
        "system": system_prompt,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            async with client.stream("POST", _OLLAMA_URL, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise RuntimeError(
                        f"Ollama returned {resp.status_code}: {text[:200]}"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("response", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Ollama not running at {_OLLAMA_URL}: {exc}"
        ) from exc


async def collect_reply(
    user_text: str,
    *,
    model: str = _DEFAULT_MODEL,
    timeout_s: float = 10.0,
) -> str:
    """Collect the full reply as a single string. Used for gate check."""
    parts: list[str] = []
    try:
        async with asyncio.timeout(timeout_s):
            async for token in stream_reply(user_text, model=model):
                parts.append(token)
    except TimeoutError:
        log.warning("llm_adapter: collect_reply timed out after %.1f s", timeout_s)
    return "".join(parts).strip()
