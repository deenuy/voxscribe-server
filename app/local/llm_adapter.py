"""LLM adapter: streaming Ollama client for spoken replies.

Streams token-by-token from Ollama's /api/chat endpoint (multi-turn).
Default model: llama3.2 (3B class, per VOICE-AUTOPILOT.md model split).
Configurable via app.state.settings.voice_reply_model.

Ollama must be running locally on port 11434. The wake loop ensures
that only short replies (2-3 sentences max) are requested by the system
prompt; latency budget is 800 ms to first audio on idle machine.

Multi-turn: pass `history` as a list of {"role": "user"|"assistant", "content": ...}
dicts (last N turns) to maintain conversation context across wake-loop turns.
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

_CHAT_URL     = "http://localhost:11434/api/chat"
_OLLAMA_URL   = "http://localhost:11434/api/generate"   # kept for collect_reply
_DEFAULT_REPLY_MODEL = "llama3.2"   # 3B class; fast spoken replies
_DEFAULT_MODEL = _DEFAULT_REPLY_MODEL  # backward-compat alias
_REQUEST_TIMEOUT = 30.0   # llama3.2 first-token <3s warm; 30s covers cold load
_KEEP_ALIVE = "10m"       # keep model in VRAM between turns (no reload penalty)
_HISTORY_TURNS = 5        # keep last N user+assistant pairs in context

def _build_system_prompt() -> str:
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    date_str = now.strftime("%A, %B %-d, %Y")  # e.g. "Saturday, October 18, 2026"
    time_str = now.strftime("%-I:%M %p UTC")   # e.g. "12:30 AM UTC"
    return (
        f"You are a helpful voice assistant. Today is {date_str}, {time_str}. "
        "Reply in 1-2 short sentences only. "
        "Be direct and conversational. No markdown, no lists, no headers. "
        "Keep replies under 30 words."
    )


async def stream_reply(
    user_text: str,
    *,
    model: str = _DEFAULT_REPLY_MODEL,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """Stream reply tokens from Ollama using the chat endpoint.

    Yields text chunks as they arrive. Each chunk may be a partial word.
    The caller is responsible for splitting into sentences for TTS.

    Pass `history` as [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    to maintain multi-turn conversation context. Bounded to _HISTORY_TURNS pairs.

    Raises RuntimeError if Ollama is not available.
    """
    effective_prompt = system_prompt or _build_system_prompt()
    messages: list[dict] = [{"role": "system", "content": effective_prompt}]
    if history:
        # Keep last _HISTORY_TURNS * 2 messages (user + assistant per turn)
        messages.extend(history[-(2 * _HISTORY_TURNS):])
    # Repeat the context header just before the user turn so small models see it
    # at maximum recency (they weight early and late tokens most strongly).
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": _KEEP_ALIVE,
    }

    log.info("llm_adapter: POST %s model=%s history_turns=%d prompt_len=%d",
             _CHAT_URL, model, len(history) // 2 if history else 0, len(user_text))
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            async with client.stream("POST", _CHAT_URL, json=payload) as resp:
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
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Ollama not running at {_CHAT_URL}: {exc}"
        ) from exc


async def collect_reply(
    user_text: str,
    *,
    model: str = _DEFAULT_MODEL,
    timeout_s: float = 10.0,
    history: list[dict] | None = None,
) -> str:
    """Collect the full reply as a single string. Used for gate check."""
    parts: list[str] = []
    try:
        async with asyncio.timeout(timeout_s):
            async for token in stream_reply(user_text, model=model, history=history):
                parts.append(token)
    except TimeoutError:
        log.warning("llm_adapter: collect_reply timed out after %.1f s", timeout_s)
    return "".join(parts).strip()
