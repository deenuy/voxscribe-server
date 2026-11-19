"""Summary generation service for meetings, and the tasks drafted after it.

Every state change of the summary and of the tasks is reported to the pipeline (pipeline.py), which records it and
publishes `meeting.pipeline` to the meeting's tenant. Tasks come after the summary and cannot undo it: once the
summary is written it stays ready whatever happens to the tasks.
"""

import asyncio
import logging
import os
from datetime import datetime

from voxscribe.knowledge.llm_stream import ProviderUnavailable
from voxscribe.meeting.summary import PROMPT_VERSION, generate_meeting_title, summarize_meeting

from . import meeting_bus, pipeline
from .store_meetings import MeetingStore
from .task_service import draft_and_store

log = logging.getLogger(__name__)

# Recordings shorter than this many words skip the LLM entirely.
MIN_WORDS_FOR_SUMMARY = int(os.getenv("VOXSCRIBE_SUMMARY_MIN_WORDS", "20"))
# Wall-clock cap on the LLM call (per attempt). Prevents a hung model from blocking finalization.
SUMMARY_TIMEOUT_S = int(os.getenv("VOXSCRIBE_SUMMARY_TIMEOUT_S", "120"))
# Set to "1" in test environments that start without an LLM to exercise the unavailable UI path.
SUMMARY_DISABLED = os.getenv("VOXSCRIBE_DISABLE_SUMMARY") == "1"

# The summary written when no speech was captured.
NO_SPEECH_OVERVIEW = "No speech was captured in this recording."


def _word_count(segments: list[dict], notes: list[dict]) -> int:
    text = " ".join(
        (s.get("text") or "") for s in segments
    ) + " " + " ".join(
        (n.get("text") or "") for n in notes
    )
    return len(text.split())


class SummaryService:
    """Service for generating meeting summaries."""

    def __init__(self, store: MeetingStore, llm):
        self.store = store
        self.llm = llm

    @property
    def _pool(self):
        return getattr(self.store, "pool", None)

    async def _status(self, meeting_id: str, status: str, reason: str | None = None) -> None:
        """Set summaries.status and report the summary step it means."""
        await self.store.set_summary_status(meeting_id, status)
        await pipeline.report(self._pool, meeting_id, "summary", *pipeline.summary_state(status, reason))

    async def _tasks_skipped(self, meeting_id: str, reason: str) -> None:
        await pipeline.report(self._pool, meeting_id, "tasks", pipeline.SKIPPED, reason)

    async def generate_summary(self, meeting_id: str) -> None:
        """Generate a summary for a meeting, then draft its tasks.

        Status path:
          pending (set by caller before scheduling) ->
          running (set here before the LLM call) ->
          ready | skipped | unavailable | failed

        Never raises; whatever goes wrong the meeting still transitions to done.
        """
        segments_models = await self._summary(meeting_id)
        if segments_models is not None:
            await self._draft_tasks(meeting_id, segments_models)

    async def _summary(self, meeting_id: str):
        """The summary step. Returns the transcript when the summary is ready and tasks should follow, else None."""
        try:
            if SUMMARY_DISABLED or not self.llm:
                log.info("Summary disabled or LLM unavailable for meeting %s", meeting_id)
                await self._status(meeting_id, "unavailable", pipeline.MODEL_UNAVAILABLE)
                await self._tasks_skipped(meeting_id, pipeline.MODEL_UNAVAILABLE)
                return None

            segments_models = await self.store.get_transcript(meeting_id)
            notes_models = await self.store.get_notes(meeting_id)

            segments = [
                {
                    "id": s.id,
                    "text": s.text,
                    "startMs": s.startMs,
                    "endMs": s.endMs,
                    "source": s.source,
                }
                for s in segments_models
            ]
            notes = [
                {"blockId": n.blockId, "text": n.text, "offsetMs": n.offsetMs}
                for n in notes_models
            ]

            no_speech = not any((s["text"] or "").strip() for s in segments) and not any(
                (n["text"] or "").strip() for n in notes
            )
            if no_speech:
                await self.store.set_summary(
                    meeting_id=meeting_id,
                    overview=NO_SPEECH_OVERVIEW,
                    details=[],
                    generated_at=datetime.utcnow().isoformat() + "Z",
                )
                await pipeline.report(self._pool, meeting_id, "summary", pipeline.SKIPPED, pipeline.NO_SPEECH)
                await self._tasks_skipped(meeting_id, pipeline.NO_SPEECH)
                await meeting_bus.publish(meeting_id, "summary.ready", {})
                log.info("No speech in meeting %s; summary skipped", meeting_id)
                return None

            if _word_count(segments, notes) < MIN_WORDS_FOR_SUMMARY:
                await self._status(meeting_id, "skipped", pipeline.TOO_SHORT)
                await self._tasks_skipped(meeting_id, pipeline.TOO_SHORT)
                log.info(
                    "Meeting %s has fewer than %d words; summary skipped",
                    meeting_id, MIN_WORDS_FOR_SUMMARY,
                )
                return None

            enhanced_resp = await self.store.get_enhanced(meeting_id)
            enhanced = [
                {"blockId": b.blockId, "text": b.text}
                for b in enhanced_resp["blocks"]
            ]

            await self._status(meeting_id, "running")

            # Call core's summarize function off the event loop with a hard timeout.
            # A model that returns nothing usable (still loading) gets one retry.
            summary = None
            failure = pipeline.MODEL_ERROR
            for attempt in (1, 2):
                try:
                    summary = await asyncio.wait_for(
                        asyncio.to_thread(
                            summarize_meeting,
                            self.llm,
                            segments,
                            notes,
                            enhanced if enhanced else None,
                            False,  # allow_extractive
                        ),
                        timeout=SUMMARY_TIMEOUT_S,
                    )
                    break
                except asyncio.TimeoutError:
                    log.error(
                        "Summary for meeting %s timed out after %ds (attempt %d)",
                        meeting_id, SUMMARY_TIMEOUT_S, attempt,
                    )
                    failure = pipeline.MODEL_TIMEOUT
                    if attempt == 2:
                        break
                except ProviderUnavailable:
                    log.error("Summary for meeting %s: the model could not be reached", meeting_id)
                    failure = pipeline.MODEL_UNAVAILABLE
                    break
                except ValueError:
                    failure = pipeline.UNUSABLE_OUTPUT
                    if attempt == 2:
                        break
                    log.warning("Summary for meeting %s was not usable, retrying", meeting_id)

            if summary is None:
                await self._status(meeting_id, "failed", failure)
                await self._tasks_skipped(meeting_id, pipeline.SUMMARY_FAILED)
                return None

            details = [
                {"title": d.title, "body": d.body}
                for d in summary.details
            ]

            # L4: flag suspiciously long summaries before storing
            word_count = len(summary.overview.split()) + sum(len(d["body"].split()) for d in details)
            if word_count > 400:
                log.warning("Summary for %s is %d words (>400); flagging", meeting_id, word_count)

            now = datetime.utcnow().isoformat() + "Z"
            await self.store.set_summary(
                meeting_id=meeting_id,
                overview=summary.overview,
                details=details,
                generated_at=now,
                prompt_version=PROMPT_VERSION,
            )
        except Exception as e:
            log.error("Summary generation error for meeting %s: %s", meeting_id, type(e).__name__, exc_info=True)
            try:
                await self._status(meeting_id, "failed", pipeline.MODEL_ERROR)
                await self._tasks_skipped(meeting_id, pipeline.SUMMARY_FAILED)
            except Exception:
                pass
            return None

        # The summary is written: from here on nothing changes its state.
        await pipeline.report(self._pool, meeting_id, "summary", pipeline.READY)
        try:
            # AI title: best-effort; never blocks or raises
            ai_title = await asyncio.wait_for(
                asyncio.to_thread(generate_meeting_title, self.llm, segments, notes),
                timeout=30.0,
            )
            if ai_title:
                await self.store.update_meeting(meeting_id, title=ai_title)
                log.info("AI title set for meeting %s (%d chars)", meeting_id, len(ai_title))
        except Exception as title_err:
            log.warning("AI title generation failed for meeting %s: %s", meeting_id, type(title_err).__name__)
        try:
            await meeting_bus.publish(meeting_id, "summary.ready", {})
        except Exception:
            log.warning("Could not publish summary.ready for meeting %s", meeting_id, exc_info=True)
        log.info("Summary generated for meeting %s", meeting_id)
        return segments_models

    async def _draft_tasks(self, meeting_id: str, segments_models) -> None:
        """SMART task auto-generation: best-effort, never raises, and never touches the summary."""
        try:
            spoken = [s for s in segments_models if s.text and s.text.strip()]
            if not spoken:
                await self._tasks_skipped(meeting_id, pipeline.NO_SPEECH)
                return
            lines = [s.text.strip() for s in spoken]
            labelled = [
                f"{s.speaker}: {s.text.strip()}" if getattr(s, "speaker", None) else s.text.strip()
                for s in spoken
            ]
            held = datetime.utcnow().strftime("%Y-%m-%d")
            created, skipped = await draft_and_store(
                self._pool, self.llm, meeting_id, lines=lines, labelled=labelled, project=None, fake=False,
                held_on=held,
            )
            log.info("Auto-generated %d task(s) for meeting %s (%d skipped)", len(created), meeting_id, skipped)
        except Exception as task_err:  # the reason is already on the pipeline
            log.warning("Auto task generation failed for meeting %s: %s", meeting_id, type(task_err).__name__)
