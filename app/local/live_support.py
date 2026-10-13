"""Builders for the pieces used during a live meeting: the transcriber, retrieval and assist."""

from voxscribe.knowledge import AnswerComposer
from voxscribe.meeting.assist import LiveAssist
from voxscribe.meeting.live import LiveTranscriber
from voxscribe.meeting.qdetect import QuestionDetector
from voxscribe.speech import SEGMENTATION, SPEC_VERSION, available_detectors, create_detector

from .pg_sync_index import PostgresKnowledgeIndexSync, PostgresSyncRetriever

# Detector names the live socket reports. The energy gates have no detector of their own.
DETECTOR_MODES = ("silero", "heuristic")
ENERGY_DETECTOR = "energy"

# vad.level messages are opt-in (`levels=1` on the live URL) and never closer than this on the session timeline:
# at most 10 per second of audio.
LEVEL_INTERVAL_MS = 100


def build_live_transcriber(settings, asr_provider, meeting_id: str) -> LiveTranscriber:
    """The transcriber for one recording, in the configured turn detection mode.

    "adaptive" and "fixed" are core's energy gates. "silero" and "heuristic" build a fresh detector for this
    recording (its state is per stream) and run core's SEGMENTATION policy through `vad_mode="detector"`.
    Raises `SpeechDetectorUnavailable` when the configured detector cannot run; startup already checked, so this
    is a real fault, never a reason to fall back quietly.
    """
    mode = getattr(settings, "live_vad_mode", "adaptive")
    max_segment_ms = getattr(settings, "live_max_segment_ms", 15000) or None
    if mode in DETECTOR_MODES:
        return LiveTranscriber(
            asr_provider,
            session_id=meeting_id,
            vad_mode="detector",
            detector=create_detector(mode),
            policy=SEGMENTATION,
            max_segment_ms=max_segment_ms,
        )
    return LiveTranscriber(asr_provider, session_id=meeting_id, vad_mode=mode, max_segment_ms=max_segment_ms)


def speech_activity_capability(settings) -> dict:
    """The `speech_activity` entry of /v1/health: which detectors this machine can run, the mode the live path is
    configured with, and the spec version of docs/SPEECH-ACTIVITY.md the engine implements."""
    detectors = [name for name in available_detectors() if name != "fake"]
    return {
        "detectors": detectors,
        "live_mode": getattr(settings, "live_vad_mode", "adaptive"),
        "spec_version": SPEC_VERSION,
    }


def vad_message(transition) -> dict:
    """The live socket's `vad` message for one `SpeechTransition` (docs/SPEECH-ACTIVITY.md section 5).

    Old clients read only `type` and `speech`. `at_ms` is on the session timeline, `p` is the deciding frame's
    probability (null for the energy gates) and `detector` names the definition of speech in use.
    """
    return {
        "type": "vad",
        "speech": bool(transition.speaking),
        "at_ms": int(transition.at_ms),
        "p": None if transition.probability is None else round(float(transition.probability), 3),
        "detector": transition.detector,
    }


def level_message(at_ms: int, probability: float) -> dict:
    """The opt-in `vad.level` message: the latest detector frame, never persisted."""
    return {"type": "vad.level", "at_ms": int(at_ms), "p": round(float(probability), 3)}


def build_retriever(app) -> PostgresSyncRetriever | None:
    """Core retriever over the Postgres index, or None when embeddings are unavailable."""
    embedder = app.state.providers.get("embedding_provider")
    if embedder is None:
        return None
    return PostgresSyncRetriever(
        PostgresKnowledgeIndexSync(app.state.settings.database_url, getattr(app.state.store.pool, "base_schema", None)), embedder
    )


class _ScopedComposer:
    """Fixes the source scope so LiveAssist can call stream(question) alone."""

    def __init__(self, composer: AnswerComposer, scope_source_ids: list[str] | None):
        self.composer = composer
        self.scope = scope_source_ids

    def stream(self, query: str):
        return self.composer.stream(query, self.scope)


def build_live_assist(app, scope_source_ids: list[str] | None) -> LiveAssist | None:
    """Live assist on the live model, answering only from the given sources (all if None)."""
    retriever = build_retriever(app)
    llm = app.state.providers.get("live_llm")
    if retriever is None or llm is None:
        return None
    composer = AnswerComposer(
        llm,
        retriever,
        use_fake_embedder=app.state.settings.providers == "fake",
    )
    return LiveAssist(QuestionDetector(), _ScopedComposer(composer, scope_source_ids))
