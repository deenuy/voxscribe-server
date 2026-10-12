"""Summary generation service for meetings."""

import asyncio
import logging
from datetime import datetime

from voxscribe.meeting.summary import summarize_meeting

from . import meeting_bus
from .store_meetings import MeetingStore

log = logging.getLogger(__name__)

# The summary of a recording in which nothing was transcribed. No model is asked to summarise nothing.
NO_SPEECH_OVERVIEW = "No speech was captured in this recording."


class SummaryService:
    """Service for generating meeting summaries."""

    def __init__(self, store: MeetingStore, llm):
        self.store = store
        self.llm = llm

    async def generate_summary(self, meeting_id: str) -> None:
        """Generate a summary for a meeting.

        If the provider is unavailable, leaves the summary in pending state
        and logs the error but does not raise.
        """
        try:
            # Check if LLM provider is available
            if not self.llm:
                log.error(f"LLM provider unavailable for summary of {meeting_id}")
                return

            # Get transcript and notes
            segments_models = await self.store.get_transcript(meeting_id)
            notes_models = await self.store.get_notes(meeting_id)

            # Convert to dicts for core
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

            # Get enhanced notes if available
            enhanced_resp = await self.store.get_enhanced(meeting_id)
            enhanced = [
                {"blockId": b.blockId, "text": b.text}
                for b in enhanced_resp["blocks"]
            ]

            if not any((s["text"] or "").strip() for s in segments) and not any(
                (n["text"] or "").strip() for n in notes
            ):
                await self.store.set_summary(
                    meeting_id=meeting_id,
                    overview=NO_SPEECH_OVERVIEW,
                    details=[],
                    generated_at=datetime.utcnow().isoformat() + "Z",
                )
                await meeting_bus.publish(meeting_id, "summary.ready", {})
                log.info(f"No speech in meeting {meeting_id}; summary not generated")
                return

            # Call core's summarize function (may be slow; run off event loop). A model that answers with
            # nothing usable (an empty reply while it is still loading, for example) gets one more try.
            for attempt in (1, 2):
                try:
                    summary = await asyncio.to_thread(
                        summarize_meeting,
                        self.llm,
                        segments,
                        notes,
                        enhanced if enhanced else None,
                        False,  # allow_extractive
                    )
                    break
                except ValueError as e:
                    if attempt == 2:
                        raise
                    log.warning(f"Summary for meeting {meeting_id} was not usable, trying once more: {e}")

            # Convert details to list of dicts
            details = [
                {"title": d.title, "body": d.body}
                for d in summary.details
            ]

            # Save summary
            now = datetime.utcnow().isoformat() + "Z"
            await self.store.set_summary(
                meeting_id=meeting_id,
                overview=summary.overview,
                details=details,
                generated_at=now,
            )

            # Publish event
            await meeting_bus.publish(meeting_id, "summary.ready", {})
            log.info(f"Summary generated for meeting {meeting_id}")
        except Exception as e:
            log.error(f"Summary generation error for meeting {meeting_id}: {e}", exc_info=True)
            # Leave summary in pending state on error, don't raise
