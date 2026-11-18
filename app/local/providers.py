"""Provider selection and initialization."""

import logging
import os
import threading
from typing import Literal

from voxscribe.audio_providers import WhisperASRProvider
from voxscribe.knowledge import (
    FakeEmbeddingProvider,
    FakeStreamingLLM,
    OllamaEmbeddingProvider,
    OllamaStreamingLLM,
)

log = logging.getLogger(__name__)


class OllamaStreamingAdapter:
    """Adapter that wraps OllamaStreamingLLM to add complete() method.

    Core functions (summarize_meeting, Enricher) expect .complete() but
    OllamaStreamingLLM only provides .stream(). This adapter joins the stream.
    """

    def __init__(self, streaming_llm: OllamaStreamingLLM):
        self.streaming_llm = streaming_llm

    @property
    def name(self) -> str:
        return self.streaming_llm.name

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        """Call stream() and join all tokens into a complete response."""
        tokens = []
        try:
            for token in self.streaming_llm.stream(system, prompt, max_tokens):
                tokens.append(token)
        except Exception as e:
            log.exception(f"Streaming LLM error: {e}")
            raise
        return "".join(tokens)

    def stream(self, system: str, prompt: str, max_tokens: int = 2048):
        """Pass through to the wrapped streaming LLM."""
        return self.streaming_llm.stream(system, prompt, max_tokens)


class FakeASRProvider:
    """Fake ASR for testing - uses core's scripted provider.

    Emits realistic cyclic transcript for UI/test validation.
    Tracks call count to return different script entries each time.
    """

    def __init__(self):
        # Build a realistic demo script with a mix of normal and question-like sentences
        self.script = [
            "The market analysis shows strong growth potential.",
            "We should consider the competitive landscape.",
            "What are our key differentiators?",
            "The timeline looks feasible for Q1.",
            "Customer feedback has been positive so far.",
            "How do we scale to handle demand?",
        ]
        self.name = "fake:deterministic"
        self.call_count = 0

    def transcribe(self, samples, sample_rate, session_id, initial_prompt=None):
        """Transcribe audio using scripted fake responses.

        Each call returns the next entry in the script, cycling through.
        This ensures repeated calls on the same audio buffer return different segments.
        """
        # Get the next script entry (cycling through the list)
        script_idx = self.call_count % len(self.script)
        text = self.script[script_idx]
        self.call_count += 1

        # Create a single TranscriptChunk for this call
        from voxscribe.models import TranscriptChunk

        # Use call count in the chunk ID to ensure uniqueness
        chunk_id = f"{session_id}-c{self.call_count}"

        # Estimate duration: 300ms per word
        duration = max(100, len(text.split()) * 300)
        start_ms = (self.call_count - 1) * (duration + 500)  # Add gap between chunks
        end_ms = start_ms + duration

        return [
            TranscriptChunk(
                chunk_id=chunk_id,
                session_id=session_id,
                text=text,
                t_start_ms=start_ms,
                t_end_ms=end_ms,
                confidence=0.95,
                gap_ms_before=0 if self.call_count == 1 else 500,
            )
        ]

    async def health(self) -> bool:
        return True


class LocalASRProvider:
    """Local ASR using faster-whisper.

    Lazy-loads the model and runs transcription off the event loop
    to avoid blocking the async handler.
    """

    def __init__(self, model_name: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        try:
            import faster_whisper  # noqa: F401
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type
            self.name = f"faster-whisper:{model_name}"
            self._provider = None
        except ImportError as e:
            raise RuntimeError("faster-whisper not installed. Install with: uv sync") from e

    def _load_provider(self):
        """Lazy-load WhisperASRProvider (which loads the model)."""
        if self._provider is None:
            self._provider = _NoSpeechWhisperProvider(
                model_size=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info(f"Loaded {self.name} model")
        return self._provider

    # Load thresholds for the beam-size profile (load-aware ASR, completion plan Phase 1).
    # beam_size=1 at load > 3: faster decode under CPU pressure, slightly lower WER.
    # beam_size=5 otherwise: accuracy-tuned default per WhisperASRProvider docs.
    _BEAM_LOAD_HIGH = 3.0
    _BEAM_SIZE_HIGH = 1
    _BEAM_SIZE_NORMAL = 5

    def transcribe(self, samples, sample_rate, session_id, initial_prompt=None):
        """Transcribe using faster-whisper with load-aware beam size.

        This is called from the event loop via asyncio.to_thread.
        """
        import os as _os
        provider = self._load_provider()
        load = _os.getloadavg()[0]
        target_beam = self._BEAM_SIZE_HIGH if load > self._BEAM_LOAD_HIGH else self._BEAM_SIZE_NORMAL
        if provider.beam_size != target_beam:
            log.info("load-aware ASR: beam_size %d→%d (load=%.1f)", provider.beam_size, target_beam, load)
            provider.beam_size = target_beam
        return provider.transcribe(samples, sample_rate, session_id, initial_prompt)

    def measure_no_speech(self, fn):
        """Run fn on this thread and return (its result, Whisper's no_speech_prob for it).

        The probability is the lowest over the segments the decodes inside fn produced (any
        segment Whisper is sure is speech makes the call speech), or None when they produced
        none. Used by the voice host's speech guard on the turn's final decode.
        """
        return self._load_provider().measure_no_speech(fn)

    async def health(self) -> bool:
        return True


class _NoSpeechProbe(threading.local):
    """Per-thread record of no_speech_prob, active only inside measure_no_speech."""

    active = False
    value: float | None = None


class _NoSpeechWhisperProvider(WhisperASRProvider):
    """WhisperASRProvider that can report faster-whisper's no_speech_prob.

    Core maps Whisper segments to TranscriptChunks and drops no_speech_prob; this wraps the
    loaded model so the segments it yields are observed on their way to core's mapping.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._probe = _NoSpeechProbe()

    def _load(self):
        model = super()._load()
        if not isinstance(model, _ProbedModel):
            model = self._model = _ProbedModel(model, self._probe)
        return model

    def measure_no_speech(self, fn):
        probe = self._probe
        probe.active, probe.value = True, None
        try:
            return fn(), probe.value
        finally:
            probe.active, probe.value = False, None


class _ProbedModel:
    """Proxy over a faster-whisper WhisperModel that records no_speech_prob while active."""

    def __init__(self, model, probe: _NoSpeechProbe) -> None:
        self._inner = model
        self._probe = probe

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def transcribe(self, audio, **kw):
        segments, info = self._inner.transcribe(audio, **kw)
        if not self._probe.active:
            return segments, info
        return self._observe(segments), info

    def _observe(self, segments):
        probe = self._probe
        for seg in segments:
            p = getattr(seg, "no_speech_prob", None)
            if p is not None and (probe.value is None or p < probe.value):
                probe.value = float(p)
            yield seg


def select_providers(
    providers_mode: Literal["fake", "local"] = "local",
    settings=None,
) -> dict:
    """Select and initialize providers based on mode."""

    if providers_mode == "fake":
        return {
            "asr": FakeASRProvider(),
            "llm": FakeStreamingLLM(),
            "live_llm": FakeStreamingLLM(),
            # The database column is 768 wide; a fake that was 384 wide made every ingestion in fake mode fail.
            "embedding": FakeEmbeddingProvider(dim=768),
            "llm_provider": FakeStreamingLLM(),
            "embedding_provider": FakeEmbeddingProvider(dim=768),
        }
    else:
        # local mode - use models from settings
        if settings is None:
            from app.local.config import Config
            settings = Config.from_env()

        # Create streaming LLM instances
        llm_streaming = OllamaStreamingLLM(model=settings.llm_model)
        live_llm_streaming = OllamaStreamingLLM(model=settings.live_model)

        return {
            "asr": LocalASRProvider(model_name=settings.asr_model),
            # main model: enrichment, summaries, Ask. live model: live assist and voice.
            # Wrap streaming LLMs with adapter to support .complete() method
            "llm": OllamaStreamingAdapter(llm_streaming),
            "live_llm": OllamaStreamingAdapter(live_llm_streaming),
            "embedding": OllamaEmbeddingProvider(model=settings.embed_model),
            "llm_provider": OllamaStreamingAdapter(llm_streaming),
            "embedding_provider": OllamaEmbeddingProvider(model=settings.embed_model),
        }
