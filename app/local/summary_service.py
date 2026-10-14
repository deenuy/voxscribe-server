"""Summary generation service for meetings."""

import asyncio
import logging
import os
from datetime import datetime

from voxscribe.meeting.summary import PROMPT_VERSION, generate_meeting_title, summarize_meeting

from . import meeting_bus
from .store_meetings import MeetingStore

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

    async def generate_summary(self, meeting_id: str) -> None:
        """Generate a summary for a meeting.

        Status path:
          pending (set by caller before scheduling) ->
          running (set here before the LLM call) ->
          ready | skipped | unavailable | failed

        Never raises; whatever goes wrong the meeting still transitions to done.
        """
        try:
            if SUMMARY_DISABLED or not self.llm:
                log.info("Summary disabled or LLM unavailable for meeting %s", meeting_id)
                await self.store.set_summary_status(meeting_id, "unavailable")
                return

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
                await meeting_bus.publish(meeting_id, "summary.ready", {})
                log.info("No speech in meeting %s; summary skipped", meeting_id)
                return

            if _word_count(segments, notes) < MIN_WORDS_FOR_SUMMARY:
                await self.store.set_summary_status(meeting_id, "skipped")
                log.info(
                    "Meeting %s has fewer than %d words; summary skipped",
                    meeting_id, MIN_WORDS_FOR_SUMMARY,
                )
                return

            enhanced_resp = await self.store.get_enhanced(meeting_id)
            enhanced = [
                {"blockId": b.blockId, "text": b.text}
                for b in enhanced_resp["blocks"]
            ]

            await self.store.set_summary_status(meeting_id, "running")

            # Call core's summarize function off the event loop with a hard timeout.
            # A model that returns nothing usable (still loading) gets one retry.
            summary = None
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
                    if attempt == 2:
                        await self.store.set_summary_status(meeting_id, "failed")
                        return
                except ValueError as e:
                    if attempt == 2:
                        raise
                    log.warning(
                        "Summary for meeting %s was not usable, retrying: %s",
                        meeting_id, e,
                    )

            if summary is None:
                await self.store.set_summary_status(meeting_id, "failed")
                return

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

            # AI title: best-effort; never blocks or raises
            try:
                ai_title = await asyncio.wait_for(
                    asyncio.to_thread(generate_meeting_title, self.llm, segments, notes),
                    timeout=30.0,
                )
                if ai_title:
                    await self.store.update_meeting(meeting_id, title=ai_title)
                    log.info("AI title set for meeting %s: %r", meeting_id, ai_title)
            except Exception as title_err:
                log.warning("AI title generation failed for meeting %s: %s", meeting_id, title_err)

            await meeting_bus.publish(meeting_id, "summary.ready", {})
            log.info("Summary generated for meeting %s", meeting_id)

        except Exception as e:
            log.error("Summary generation error for meeting %s: %s", meeting_id, e, exc_info=True)
            try:
                await self.store.set_summary_status(meeting_id, "failed")
            except Exception:
                pass
