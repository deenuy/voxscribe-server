"""Enrichment service for meeting notes with idempotency and debouncing."""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime

from voxscribe.meeting.enrich import Enricher, TranscriptSegment

from . import meeting_bus
from .models import Citation, EnhancedBlock
from .store_meetings import MeetingStore

log = logging.getLogger(__name__)


class EnrichmentService:
    """Service for enriching meeting notes with LLM and retrieval.

    Features:
    - Idempotent enrichment per block (hash-based)
    - Debounced note-triggered runs with task cancellation
    - Never overwrites edited blocks
    - Drops invalid citations
    - Records usage without failing requests
    """

    def __init__(self, store: MeetingStore, llm, retriever=None, debounce_delay: float = 1.0):
        self.store = store
        self.llm = llm
        self.retriever = retriever
        self.enricher = Enricher(llm, retriever, context_window_sec=30)
        self.pending_tasks: dict[str, asyncio.Task] = {}
        self.debounce_delay = debounce_delay
        self.block_hashes: dict[str, str] = {}  # Track hashes to detect changes

    def _block_hash(self, text: str, offset_ms: int, style: str, segment_ids: list[str]) -> str:
        """Hash of block content to detect changes (text, window contents, style)."""
        key = f"{text}|{offset_ms}|{style}|{','.join(segment_ids)}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _citation_valid(self, citation_dict: dict) -> bool:
        """Check if a citation is valid."""
        try:
            Citation(**citation_dict)
            return True
        except Exception:
            return False

    async def _save_and_publish(
        self,
        meeting_id: str,
        block_id: str,
        text: str,
        citations: list[Citation],
        status: str,
    ) -> bool:
        """Persist an enhanced block and announce it. Never overwrites an edited block."""
        current = await self.store.get_enhanced(meeting_id)
        for block in current.get("blocks", []):
            if block.blockId == block_id and block.status == "edited":
                return False
        await self.store.set_enhanced_block(
            meeting_id=meeting_id,
            block_id=block_id,
            text=text,
            citations=citations,
            status=status,
        )
        enhanced_block = EnhancedBlock(
            blockId=block_id,
            text=text,
            status=status,
            citations=citations,
            updatedAt=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        await meeting_bus.publish(
            meeting_id, "enhanced.update", enhanced_block.model_dump(exclude_none=True)
        )
        return True

    async def enrich_block(
        self,
        meeting_id: str,
        block_id: str,
        text: str,
        offset_ms: int,
        style: str = "standard",
    ) -> None:
        """Enrich a single block (idempotent per hash).

        Never overwrites edited blocks. Drops invalid citations.
        Records usage but never fails the request on usage errors.
        """
        try:
            # Check if block is edited (never overwrite)
            enhanced = await self.store.get_enhanced(meeting_id)
            for block in enhanced.get("blocks", []):
                if block.blockId == block_id and block.status == "edited":
                    log.info(f"Skipping enrichment of edited block {block_id}")
                    return

            # Get transcript for context (biased to preceding span)
            segments = await self.store.get_transcript(meeting_id)
            context_window_ms = 30000
            nearby_segments = [
                s
                for s in segments
                if s.startMs >= max(0, offset_ms - context_window_ms)
                and s.startMs <= offset_ms + context_window_ms
            ]

            # Idempotency: has this block + window + style already been enriched?
            hash_key = f"{meeting_id}:{block_id}"
            current_hash = self._block_hash(text, offset_ms, style, [s.id for s in nearby_segments])
            if self.block_hashes.get(hash_key) == current_hash:
                log.debug(f"Block {block_id} unchanged, skipping enrichment")
                return

            if not nearby_segments:
                # Nothing to ground on yet: the block waits for the conversation to cover it
                await self._save_and_publish(meeting_id, block_id, "", [], "pending")
                return

            transcript_segments = [
                TranscriptSegment(
                    id=s.id,
                    start_ms=s.startMs,
                    end_ms=s.endMs,
                    text=s.text,
                )
                for s in nearby_segments
            ]

            # Retrieve relevant source chunks from the sources attached to this meeting
            retrieved_chunks = []
            if self.retriever:
                attachments = await self.store.get_attachments(meeting_id)
                if attachments:
                    try:
                        chunks = await asyncio.to_thread(
                            self.retriever.retrieve, text, 3, attachments
                        )
                        for chunk in chunks:
                            retrieved_chunks.append(
                                {
                                    "source_id": chunk.source_id,
                                    "chunk_id": chunk.chunk_id,
                                    "title": chunk.title,
                                    "locator": chunk.locator,
                                    "text": chunk.text,
                                }
                            )
                    except Exception as e:
                        log.warning(f"Retrieval error: {e}")

            # The model call is blocking; keep it off the event loop
            enriched = await asyncio.to_thread(
                self.enricher.enrich,
                block_id=block_id,
                text=text,
                offset_ms=offset_ms,
                transcript_segments=transcript_segments,
                retrieved_chunks=retrieved_chunks or None,
                style=style,
            )

            # Convert and validate citations (drop invalid ones)
            citations = []
            for c in enriched.citations:
                if self._citation_valid(c):
                    citations.append(Citation(**c))
                else:
                    log.warning(f"Dropped invalid citation: {c}")

            saved = await self._save_and_publish(
                meeting_id, block_id, enriched.text, citations, enriched.status
            )

            # Remember the run only when it produced a usable note, so a pending
            # result (model output unusable) is retried when more transcript arrives.
            if saved and enriched.status != "pending":
                self.block_hashes[hash_key] = current_hash

        except Exception as e:
            log.error(f"Enrichment error for block {block_id}: {e}", exc_info=True)
            # Don't raise; let the request succeed

    async def enrich_block_debounced(
        self,
        meeting_id: str,
        block_id: str,
        text: str,
        offset_ms: int,
        style: str = "standard",
    ) -> None:
        """Debounced enrich: cancel previous task for this block if pending."""
        task_key = f"{meeting_id}:{block_id}"

        # Cancel previous pending task
        if task_key in self.pending_tasks:
            task = self.pending_tasks[task_key]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    log.debug(f"Cancelled previous enrichment task for {block_id}")

        # Schedule new task with debounce delay
        async def debounced_enrich():
            await asyncio.sleep(self.debounce_delay)
            await self.enrich_block(meeting_id, block_id, text, offset_ms, style)

        task = asyncio.create_task(debounced_enrich())
        self.pending_tasks[task_key] = task

    async def finalize_meeting(self, meeting_id: str) -> None:
        """Finalize a meeting after stop: bring every note up to date with the full transcript.

        The caller generates the summary and marks the meeting done afterwards.
        """
        try:
            # Mark any notes that don't have enhanced blocks yet
            notes = await self.store.get_notes(meeting_id)
            meeting = await self.store.get_meeting(meeting_id)

            for note in notes:
                enhanced = await self.store.get_enhanced(meeting_id)
                block_ids = {b.blockId for b in enhanced["blocks"]}
                if note.blockId not in block_ids:
                    await self.store.set_enhanced_block(
                        meeting_id=meeting_id,
                        block_id=note.blockId,
                        text="",
                        citations=[],
                        status="pending",
                    )

            # Run enrichment for all non-edited blocks
            for note in notes:
                await self.enrich_block(
                    meeting_id=meeting_id,
                    block_id=note.blockId,
                    text=note.text,
                    offset_ms=note.offsetMs,
                    style=meeting.enhanceStyle,
                )
        except Exception as e:
            log.error(f"Finalization error for meeting {meeting_id}: {e}", exc_info=True)
            raise
