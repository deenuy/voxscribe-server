"""Session processing handler and the progress hub.

The client records audio locally, then uploads it. The router enqueues a Job;
a queue worker calls process_session. Raises on failure so the queue can retry.
Session context is set by the queue worker before calling here, so every log
line carries the session_id, tenant_id, and device_id automatically.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from . import asr, extraction
from .jobs import Job
from .logging_config import get_logger
from .store import store
from .vocab import now_iso

log = get_logger(__name__)
DATA_DIR = os.environ.get("DATA_DIR", "./data")

try:
    from voxscribe import DictationMode, PostProcessor  # type: ignore
    from voxscribe.providers import OllamaProvider  # type: ignore
    _HAS_VOXSCRIBE = True
except Exception:
    _HAS_VOXSCRIBE = False


class ProgressHub:
    """Per-session fan-out of progress events with replay for late subscribers."""

    def __init__(self) -> None:
        self._history: dict[str, list[dict]] = {}
        self._subs: dict[str, set[asyncio.Queue]] = {}

    def emit(self, session_id: str, event: dict[str, Any]) -> None:
        if event.get("step") == "received":
            self._history[session_id] = []
        self._history.setdefault(session_id, []).append(event)
        for q in self._subs.get(session_id, set()):
            q.put_nowait(event)

    async def subscribe(self, session_id: str):
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(session_id, set()).add(q)
        try:
            for past in list(self._history.get(session_id, [])):
                yield past
                if past.get("done"):
                    return
            while True:
                event = await q.get()
                yield event
                if event.get("done"):
                    return
        finally:
            self._subs.get(session_id, set()).discard(q)


hub = ProgressHub()


def _cleanup_transcript(segments: list[dict]) -> list[dict]:
    if not _HAS_VOXSCRIBE or os.environ.get("CLEANUP", "off") != "on":
        return segments
    pp = PostProcessor(OllamaProvider())
    out = []
    for seg in segments:
        try:
            cleaned = pp.process(seg["text"], DictationMode.PM).text
        except Exception:
            cleaned = seg["text"]
        out.append({**seg, "text": cleaned})
    return out


async def process_session(job: Job) -> None:
    """Processing pipeline for one session. Raises on failure; the queue owns
    retry and dead-letter logic. Session context is set by the caller."""
    sid = job.session_id

    def step(label: str, pct: int, done: bool = False) -> None:
        hub.emit(sid, {"step": label, "pct": pct, "done": done})
        log.info("session.step", extra={"step": label, "pct": pct, "done": done})

    log.info("session.pipeline_start", extra={"idea_id": job.idea_id, "source": job.source_type})
    step("received", 10)
    store.set_session_status(sid, "processing")
    await asyncio.sleep(0)

    if job.audio_path:
        step("transcribing", 35)
        log.debug("session.asr_start", extra={"audio_path": job.audio_path, "participants": job.participants})
        segments = await asyncio.to_thread(asr.transcribe, job.audio_path, job.participants)
        log.debug("session.asr_done", extra={"segment_count": len(segments)})
    else:
        log.debug("session.asr_skipped", extra={"reason": "no audio path"})
        segments = []

    step("diarizing", 55)
    segments = _cleanup_transcript(segments)
    step("cleaning transcript", 70)

    step("extracting idea", 85)
    log.debug("session.extraction_start", extra={"extractor": os.environ.get("EXTRACTOR", "simulated")})
    idea = await asyncio.to_thread(
        extraction.extract_idea,
        idea_id=job.idea_id,
        tenant_id=job.tenant_id,
        workspace_id=job.workspace_id,
        project_id=job.project_id,
        session_id=sid,
        session_title=job.session_title,
        source_type=job.source_type,
        source_ref=job.source_ref,
        participants=job.participants,
        context=job.context,
        transcript=segments,
    )

    store.put_idea(idea)
    sess = store.get_session(job.tenant_id, job.workspace_id, sid) or {}
    sess.update({"status": "complete", "ideaId": job.idea_id, "completedAt": now_iso()})
    store.put_session(sess)
    step("saved", 100, done=True)
    log.info(
        "session.complete",
        extra={"idea_id": job.idea_id, "score": idea.get("statusScore"), "title": idea.get("title")},
    )
