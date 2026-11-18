"""WakeDetector adapter: open-vocabulary keyword spotting with sherpa-onnx.

The keyword is plain text ("hey vox"), tokenised at start-up with the model's own BPE model, so a
new phrase needs no training and no recorded clips. The model is a 3.3 M parameter streaming
Zipformer transducer trained on GigaSpeech (k2-fsa, Apache-2.0); the int8 files are about 5 MB.

The spotter is an event detector, not a frame classifier: it searches over many frames itself
and reports a keyword once, after `num_trailing_blanks` of silence-like output. `score()`
therefore returns 1.0 on the frame where the keyword is decoded and 0.0 otherwise, and the
adapter declares `patience_frames = 1` so the gate does not wait for a second event that never
comes.

The model is fetched once with `uv run tools/fetch_wake_models.py`; nothing is downloaded at run
time.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from voxscribe.wake.gate import ASR_CONFIRM_PHRASE, SAMPLE_RATE

KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{KWS_MODEL_NAME}.tar.bz2"
)
KWS_MODEL_DIR_ENV = "VOXSCRIBE_KWS_MODEL_DIR"
_DEFAULT_MODEL_DIR = Path.home() / ".cache" / "voxscribe" / "kws" / KWS_MODEL_NAME
_ONNX_SUFFIX = "epoch-12-avg-2-chunk-16-left-64.int8.onnx"

# Picked by the T02 grid on synthetic clips (tools/voice_bench/replay.py): boost 2.0 and
# threshold 0.25 lifted held-out voices from 5 to 9 of 20 with no hit on the 206 s negative set.
KWS_BOOST = 2.0
KWS_THRESHOLD = 0.25
KWS_TRAILING_BLANKS = 1
KWS_MAX_ACTIVE_PATHS = 4
KWS_THREADS = 1
DETECTED_SCORE = 1.0
SILENT_SCORE = 0.0


def default_model_dir() -> Path:
    return Path(os.environ.get(KWS_MODEL_DIR_ENV, "").strip() or _DEFAULT_MODEL_DIR)


class KwsModelMissing(RuntimeError):
    pass


class OpenVocabWakeDetector:
    """`voxscribe.session.ports.WakeDetector` over sherpa-onnx keyword spotting."""

    patience_frames = 1

    def __init__(
        self,
        keyword: str = ASR_CONFIRM_PHRASE,
        model_dir: Path | str | None = None,
        *,
        boost: float = KWS_BOOST,
        threshold: float = KWS_THRESHOLD,
    ) -> None:
        import sentencepiece as spm
        import sherpa_onnx

        d = Path(model_dir) if model_dir is not None else default_model_dir()
        files = {k: d / f"{k}-{_ONNX_SUFFIX}" for k in ("encoder", "decoder", "joiner")}
        missing = [p for p in [d / "tokens.txt", d / "bpe.model", *files.values()] if not p.is_file()]
        if missing:
            raise KwsModelMissing(
                f"keyword spotting model not found in {d}; run `uv run tools/fetch_wake_models.py`")

        self.keyword = " ".join(keyword.lower().split())
        self.model_name = f"openvocab:{self.keyword.replace(' ', '_')}"
        sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
        tokens = sp.encode(self.keyword.upper(), out_type=str)
        label = self.keyword.upper().replace(" ", "_")

        # sherpa-onnx reads keywords from a file at construction; it is not needed afterwards.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"{' '.join(tokens)} @{label}\n")
            kw_path = f.name
        try:
            self._kws = sherpa_onnx.KeywordSpotter(
                tokens=str(d / "tokens.txt"),
                encoder=str(files["encoder"]),
                decoder=str(files["decoder"]),
                joiner=str(files["joiner"]),
                keywords_file=kw_path,
                num_threads=KWS_THREADS,
                sample_rate=SAMPLE_RATE,
                max_active_paths=KWS_MAX_ACTIVE_PATHS,
                keywords_score=boost,
                keywords_threshold=threshold,
                num_trailing_blanks=KWS_TRAILING_BLANKS,
            )
        finally:
            os.unlink(kw_path)
        self._stream = self._kws.create_stream()

    def reset(self) -> None:
        self._stream = self._kws.create_stream()

    def score(self, pcm_float32_mono: np.ndarray) -> float:
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm_float32_mono, dtype=np.float32))
        hit = False
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            if self._kws.get_result(self._stream):
                hit = True
                self._kws.reset_stream(self._stream)
        return DETECTED_SCORE if hit else SILENT_SCORE
