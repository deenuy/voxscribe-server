"""Unit test: auto-stop schedules finalization exactly once (PL01, I3).

When the silence threshold is crossed inside audio_consumer the auto-stop path calls
schedule_finalize and sets finalize_scheduled = True (via nonlocal).  The finally block
checks that flag and must NOT schedule a second finalization.  This test fails without
the nonlocal fix because finalize_scheduled stays False in the outer scope and the
finally block fires a second call.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.local.routers.live import _live_meeting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence_frame(ms: float = 20.0) -> bytes:
    """Source-tagged raw PCM16 silence: 1 byte tag + raw samples."""
    n_samples = int(16_000 * ms / 1000.0)
    return b"\x00" + b"\x00\x00" * n_samples


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_stop_schedules_finalize_exactly_once():
    """schedule_finalize is called exactly once when auto-stop fires.

    With autoStopMinutes=0.001 the silence threshold is 60 ms.  The fake transcriber
    advances elapsed_ms by 50 ms per push so the threshold is crossed on the third
    push (elapsed=150, silence_start=50, silence_ms=100 >= 60).
    """
    # --- meeting -----------------------------------------------------------------
    meeting = MagicMock()
    meeting.status = "recording"
    meeting.autoStopMinutes = 0.001   # 60 ms silence threshold
    meeting.enhanceStyle = "standard"

    # --- transcriber: advances elapsed_ms 50 ms per call, never returns stable segs
    elapsed_values = [50, 100, 150, 200]
    push_call = [0]

    fake_transcriber = MagicMock()

    def _push_audio(source_tag, frame):
        idx = min(push_call[0], len(elapsed_values) - 1)
        fake_transcriber.elapsed_ms = elapsed_values[idx]
        push_call[0] += 1
        return []

    fake_transcriber.push_audio.side_effect = _push_audio
    fake_transcriber.elapsed_ms = 0
    fake_transcriber.pop_speech_transitions.return_value = []
    fake_transcriber.pop_partials.return_value = []
    fake_transcriber.speech_probability = None
    fake_transcriber.close.return_value = []

    # --- store -------------------------------------------------------------------
    store_mock = AsyncMock()
    store_mock.get_meeting.return_value = meeting
    store_mock.stop_once.return_value = (None, True)   # stopped_now = True

    # --- app ---------------------------------------------------------------------
    app_mock = MagicMock()
    app_mock.state.auth_disabled = True
    app_mock.state.store.pool = MagicMock()
    app_mock.state.providers = {"asr": MagicMock(), "llm": None}
    app_mock.state.settings = MagicMock()

    # --- websocket ---------------------------------------------------------------
    # Delivers 4 audio frames, then waits until auto-stop fires, then disconnects.
    frames = [_silence_frame() for _ in range(4)]
    frame_idx = [0]
    auto_stop_event = asyncio.Event()

    async def fake_receive():
        await asyncio.sleep(0)          # always yield so consumer can run
        if auto_stop_event.is_set():
            return {"type": "websocket.disconnect"}
        if frame_idx[0] < len(frames):
            data = {"bytes": frames[frame_idx[0]]}
            frame_idx[0] += 1
            return data
        # All frames sent; wait for auto-stop before disconnecting.
        try:
            await asyncio.wait_for(auto_stop_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        return {"type": "websocket.disconnect"}

    ws_mock = MagicMock()
    ws_mock.scope = {"app": app_mock}
    ws_mock.app = app_mock
    ws_mock.accept = AsyncMock()
    ws_mock.send_json = AsyncMock()
    ws_mock.close = AsyncMock()
    ws_mock.receive = fake_receive

    # --- schedule_finalize counter -----------------------------------------------
    finalize_calls: list[str] = []

    def mock_schedule_finalize(app, meeting_id):
        finalize_calls.append(meeting_id)
        auto_stop_event.set()   # unblock fake_receive so the socket can close

    # --- run ---------------------------------------------------------------------
    with (
        patch("app.local.routers.live.MeetingStore", return_value=store_mock),
        patch("app.local.routers.live.build_live_transcriber", return_value=fake_transcriber),
        patch("app.local.routers.live.build_retriever", return_value=None),
        patch("app.local.meeting_bus.get_bus") as mock_get_bus,
        patch("app.local.meeting_bus.publish", new_callable=AsyncMock),
        patch("app.local.finalize.schedule_finalize", side_effect=mock_schedule_finalize),
    ):
        bus_mock = MagicMock()
        bus_mock.subscribe = AsyncMock(return_value=asyncio.Queue())
        mock_get_bus.return_value = bus_mock

        await _live_meeting(ws_mock, "mtg-finalize-test", None, send_levels=False)

    assert finalize_calls == ["mtg-finalize-test"], (
        f"schedule_finalize must be called exactly once; got {len(finalize_calls)} call(s): {finalize_calls}"
    )
