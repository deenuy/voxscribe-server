"""Live WebSocket endpoint for meetings."""

import hmac
import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from voxscribe.meeting.live import LiveTranscriber

from app.local import meeting_bus, tenancy
from app.local.audio_ring import AudioRing
from app.local.identity import IdentityError, tenant_for
from app.local.enrichment import EnrichmentService
from app.local.live_support import build_live_assist, build_retriever
from app.local.models import Block, EnhancedBlock, Segment
from app.local.store_meetings import MeetingStore

router = APIRouter(prefix="/v1/meetings", tags=["live"])
log = logging.getLogger(__name__)


@router.websocket("/{meeting_id}/live")
async def live_meeting(websocket: WebSocket, meeting_id: str, token: str = Query(None), identity: str = Query(None)):
    """WebSocket endpoint for live meeting transcription.

    Clients connect with token auth (and, when tenant isolation is on, the account's access token as `identity`:
    a browser cannot set headers on a WebSocket). The tenant is fixed once at connect and held for the whole
    recording; everything the connection starts runs inside it.
    """
    app = websocket.scope.get("app")
    if app is not None and getattr(app.state, "identity", None) is not None:
        try:
            scope = await tenant_for(app, identity or websocket.headers.get("x-identity-token"))
        except IdentityError:
            await websocket.close(code=1008, reason="identity required")
            return
        with tenancy.tenant_scope(scope):
            await _live_meeting(websocket, meeting_id, token)
        return
    await _live_meeting(websocket, meeting_id, token)


async def _live_meeting(websocket: WebSocket, meeting_id: str, token: str | None):
    # Extract app from scope first (before validation)
    app = websocket.scope.get("app")
    if not app:
        await websocket.close(code=1011, reason="internal error")
        return

    # Validate token
    if not token:
        await websocket.close(code=1008, reason="token required")
        return

    expected_token = getattr(app.state, "auth_token", None)
    if expected_token and not hmac.compare_digest(token.encode(), expected_token.encode()):
        await websocket.close(code=1008, reason="invalid token")
        return

    pool = app.state.store.pool if hasattr(app.state, "store") else None
    if not pool:
        await websocket.close(code=1011, reason="internal error")
        return

    store = MeetingStore(pool)

    try:
        meeting = await store.get_meeting(meeting_id)
    except Exception:
        await websocket.close(code=1008, reason="meeting not found")
        return

    if meeting.status != "recording":
        await websocket.close(code=1008, reason="meeting not recording")
        return

    await websocket.accept()

    # Initialize live transcriber and state
    try:
        providers = getattr(app.state, "providers", {})
        asr_provider = providers.get("asr")
        if not asr_provider:
            await websocket.send_json({"type": "error", "code": "provider_unavailable", "detail": "ASR provider not available"})
            await websocket.close(code=1011, reason="provider unavailable")
            return

        settings = getattr(app.state, "settings", None)
        transcriber = LiveTranscriber(
            asr_provider,
            session_id=meeting_id,
            vad_mode=getattr(settings, "live_vad_mode", "adaptive"),
            max_segment_ms=getattr(settings, "live_max_segment_ms", 15000) or None,
        )
        speaking = False  # last speech state told to the client

        # Ring buffer: decouples WS receive from push_audio (Fix 1 / S5-A).
        # wait_for_space applies TCP backpressure instead of dropping oldest frames.
        ring = AudioRing(capacity_ms=60_000.0, policy="wait_for_space")

        # Track state for live transcription
        auto_stop_enabled = meeting.autoStopMinutes and meeting.autoStopMinutes > 0
        live_enhance_enabled = True  # contract: live enhancement is on until the client turns it off
        enhance_style = meeting.enhanceStyle or "standard"
        live_assist = None
        background: set[asyncio.Task] = set()

        llm = providers.get("llm")
        enrichment = EnrichmentService(store, llm, build_retriever(app)) if llm else None

        def spawn(coro):
            task = asyncio.create_task(coro)
            background.add(task)
            task.add_done_callback(background.discard)

        async def reenrich_pending():
            """Notes that were waiting for the conversation get another try."""
            if not enrichment or not live_enhance_enabled:
                return
            enhanced = await store.get_enhanced(meeting_id)
            waiting = {b.blockId for b in enhanced["blocks"] if b.status == "pending"}
            for note in await store.get_notes(meeting_id):
                if note.blockId in waiting:
                    await enrichment.enrich_block_debounced(
                        meeting_id, note.blockId, note.text, note.offsetMs, enhance_style
                    )

        async def run_assist(core_seg):
            if live_assist is None:
                return
            try:
                suggestion = await asyncio.to_thread(live_assist.process_segment, core_seg, [])
            except Exception as e:
                log.warning(f"Live assist failed: {e}")
                return
            if suggestion is None or not suggestion.citations:
                return  # only grounded answers are shown
            await meeting_bus.publish(
                meeting_id,
                "assist.suggestion",
                {
                    "id": suggestion.id,
                    "offsetMs": suggestion.offset_ms,
                    "question": suggestion.question,
                    "answer": suggestion.answer,
                    "citations": suggestion.citations,
                },
            )

        # Get meeting bus for events
        bus = meeting_bus.get_bus(meeting_id)

        # Event reader task
        async def event_reader():
            """Read events from the meeting bus and send to client."""
            queue = await bus.subscribe()
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    frame = _to_frame(event)
                    if frame is not None:
                        try:
                            await websocket.send_json(frame)
                        except Exception as e:
                            log.debug(f"Failed to send event: {e}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    log.debug(f"Event reader error: {e}")
                    break

        # Start event reader in background
        event_task = asyncio.create_task(event_reader())

        # Consumer: drains the ring and calls push_audio off the event loop
        async def audio_consumer():
            nonlocal speaking
            _auto_stop_silence_start_ms = None

            while True:
                frame = await ring.pop()
                if frame is None:
                    break

                source_tag = frame[0]
                audio_frame = frame[1:]

                try:
                    new_stable = await asyncio.to_thread(transcriber.push_audio, source_tag, audio_frame)
                except Exception as e:
                    log.error(f"Transcription error: {e}")
                    continue

                if transcriber.in_speech != speaking:
                    speaking = transcriber.in_speech
                    try:
                        await websocket.send_json({"type": "vad", "speech": speaking})
                    except Exception:
                        pass

                partials = transcriber.pop_partials()
                for seg in partials:
                    partial_frame = {
                        "id": seg.id,
                        "startMs": seg.start_ms,
                        "endMs": seg.end_ms,
                        "source": seg.source,
                        "speaker": seg.speaker,
                        "text": seg.text,
                        "state": seg.state,
                    }
                    await meeting_bus.publish(meeting_id, "transcript.partial", partial_frame)
                    try:
                        await websocket.send_json({"type": "transcript.partial", "segment": partial_frame})
                    except Exception:
                        pass

                await _publish_stable(websocket, store, meeting_id, new_stable)

                if new_stable:
                    if live_assist is not None:
                        for seg in new_stable:
                            spawn(run_assist(seg))
                    spawn(reenrich_pending())

                if auto_stop_enabled:
                    if new_stable:
                        _auto_stop_silence_start_ms = None
                    elif _auto_stop_silence_start_ms is None:
                        _auto_stop_silence_start_ms = transcriber.elapsed_ms

                    if _auto_stop_silence_start_ms is not None:
                        silence_ms = transcriber.elapsed_ms - _auto_stop_silence_start_ms
                        silence_threshold_ms = meeting.autoStopMinutes * 60 * 1000
                        if silence_ms >= silence_threshold_ms:
                            log.info(f"Auto-stopping meeting {meeting_id} due to silence")
                            _, stopped_now = await store.stop_once(meeting_id)
                            if stopped_now:
                                from app.local.finalize import schedule_finalize
                                schedule_finalize(websocket.app, meeting_id)
                            try:
                                await websocket.send_json({"type": "status", "state": "processing"})
                            except Exception:
                                pass
                            break

        consumer_task = asyncio.create_task(audio_consumer())

        # Send initial status
        await websocket.send_json({"type": "status", "state": "recording"})

        try:
            while True:
                try:
                    data = await websocket.receive()
                except WebSocketDisconnect:
                    log.info(f"Client disconnected from live session {meeting_id}")
                    break
                except RuntimeError as e:
                    if "disconnect" in str(e).lower():
                        log.info(f"Client disconnect detected: {meeting_id}")
                        break
                    log.error(f"WebSocket receive error: {e}")
                    break
                except Exception as e:
                    log.error(f"WebSocket receive error: {e}")
                    break

                # Handle disconnect message
                if data.get("type") == "websocket.disconnect":
                    log.info(f"Client disconnected (disconnect message) from live session {meeting_id}")
                    break

                # Handle binary audio frames: enqueue to ring (non-blocking)
                if "bytes" in data:
                    frame = data["bytes"]
                    if len(frame) < 1:
                        continue

                    evicted_ms = await ring.push_wait(frame)
                    if evicted_ms > 0:
                        warn_frame = {"type": "audio.warning", "droppedMs": round(evicted_ms, 1)}
                        await meeting_bus.publish(meeting_id, "audio.warning", warn_frame)
                        try:
                            await websocket.send_json(warn_frame)
                        except Exception:
                            pass

                # Handle text control messages
                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "code": "invalid_message", "detail": "invalid JSON"})
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "note":
                        # Upsert a note block
                        block_id = msg.get("blockId")
                        text = msg.get("text")
                        offset_ms = msg.get("offsetMs", 0)

                        if not block_id or not text:
                            await websocket.send_json({"type": "error", "code": "invalid_request", "detail": "blockId and text required"})
                            continue

                        try:
                            existing = await store.get_notes(meeting_id)
                            previous = next((b for b in existing if b.blockId == block_id), None)
                            changed = previous is None or previous.text != text
                            block = Block(blockId=block_id, text=text, offsetMs=offset_ms)
                            await store.put_notes(
                                meeting_id, [block] + [b for b in existing if b.blockId != block_id]
                            )
                            await meeting_bus.publish(meeting_id, "note.updated", {"blockId": block_id})

                            if changed and previous is None:
                                # Show the new line as waiting right away
                                await _announce_pending(store, meeting_id, block_id)
                            if changed and enrichment and live_enhance_enabled:
                                await enrichment.enrich_block_debounced(
                                    meeting_id, block_id, text, offset_ms, enhance_style
                                )
                        except Exception as e:
                            log.error(f"Error updating note: {e}")
                            await websocket.send_json({"type": "error", "code": "internal_error", "detail": str(e)})

                    elif msg_type == "assist":
                        if msg.get("enabled", False):
                            if live_assist is None:
                                attached = await store.get_attachments(meeting_id)
                                live_assist = build_live_assist(app, attached or None)
                                if live_assist is None:
                                    await websocket.send_json({
                                        "type": "error",
                                        "code": "provider_unavailable",
                                        "detail": "live assist is not available",
                                    })
                        else:
                            live_assist = None

                    elif msg_type == "enhance":
                        live_enhance_enabled = bool(msg.get("live", False))
                        enhance_style = msg.get("style", enhance_style)
                        await store.update_meeting(meeting_id, enhance_style=enhance_style)
                        if live_enhance_enabled:
                            spawn(reenrich_pending())

        finally:
            # Close ring and drain consumer before flush so no frames are lost.
            ring.close()
            try:
                await asyncio.wait_for(consumer_task, timeout=120.0)
            except asyncio.TimeoutError:
                consumer_task.cancel()
                log.warning(
                    "Audio consumer drain timeout for %s: %.0f ms remaining",
                    meeting_id, ring.fill_ms,
                )
                try:
                    await consumer_task
                except (asyncio.CancelledError, Exception):
                    pass

            log.info(
                "session %s: pushed_ms=%.0f dropped_ms=%.0f overflows=%d peak_fill_ms=%.0f",
                meeting_id,
                ring.stats.pushed_ms,
                ring.stats.dropped_ms,
                ring.stats.overflow_count,
                ring.stats.peak_fill_ms,
            )

            # Flush any words still in the ASR buffer after consumer drained.
            try:
                tail = await asyncio.to_thread(transcriber.flush)
                await _publish_stable(websocket, store, meeting_id, tail, send=False)
            except Exception as e:
                log.warning(f"Could not keep the last words of {meeting_id}: {e}")
            for task in list(background):
                task.cancel()
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        log.error(f"WebSocket error for meeting {meeting_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "code": "internal_error", "detail": str(e)})
        except Exception:
            pass


async def _publish_stable(websocket: WebSocket, store: MeetingStore, meeting_id: str, segments, send: bool = True) -> None:
    """Persist stable segments, send to the originating socket, and fan out to all watchers via the bus."""
    for seg in segments:
        db_segment = Segment(
            id=seg.id,
            startMs=seg.start_ms,
            endMs=seg.end_ms,
            source=seg.source,
            speaker=seg.speaker,
            text=seg.text,
            state=seg.state,
        )
        try:
            await store.upsert_segment(meeting_id, db_segment)
        except Exception as e:
            log.warning(f"Failed to persist segment: {e}")

        frame = {
            "id": db_segment.id,
            "startMs": db_segment.startMs,
            "endMs": db_segment.endMs,
            "source": db_segment.source,
            "speaker": db_segment.speaker,
            "text": db_segment.text,
            "state": db_segment.state,
        }
        # Fan out to all connected clients (browser watchers, second sockets, etc.)
        await meeting_bus.publish(meeting_id, "transcript.stable", frame)

        if not send:
            continue
        await websocket.send_json({"type": "transcript.stable", "segment": frame})


def _to_frame(event: dict) -> dict | None:
    """Turn an internal bus event into the contract's WebSocket frame."""
    kind = event.get("type")
    data = event.get("data") or {}
    if kind in ("transcript.stable", "transcript.partial"):
        # data is the segment dict; wrap it the same way the direct send does
        return {"type": kind, "segment": data}
    if kind == "enhanced.update":
        return {"type": kind, "block": data}
    if kind == "assist.suggestion":
        return {"type": kind, **data}
    if kind == "status":
        return {"type": kind, "state": data.get("state")}
    if kind == "summary.ready":
        return {"type": kind}
    return None


async def _announce_pending(store: MeetingStore, meeting_id: str, block_id: str) -> None:
    """Record and announce a waiting enhanced block for a line that has none yet."""
    from datetime import UTC, datetime

    enhanced = await store.get_enhanced(meeting_id)
    if any(b.blockId == block_id for b in enhanced["blocks"]):
        return
    await store.set_enhanced_block(meeting_id, block_id, "", [], "pending")
    block = EnhancedBlock(
        blockId=block_id,
        text="",
        status="pending",
        citations=[],
        updatedAt=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    await meeting_bus.publish(meeting_id, "enhanced.update", block.model_dump(exclude_none=True))
