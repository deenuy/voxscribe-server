"""ASR plug point.

The Electron client records the audio and uploads it here. This module turns
that audio into a transcript with timestamps. The default backend is a stub so
the image stays light and runs with no model download. Swap a real backend via
the ASR_BACKEND env var.

  ASR_BACKEND=stub            no model, returns a placeholder transcript, or
                              reads a sidecar .txt if the client uploaded one
  ASR_BACKEND=faster_whisper  local CTranslate2 Whisper (add faster-whisper to
                              requirements; pulls a model on first run)
  ASR_BACKEND=openai          OpenAI Whisper API (needs OPENAI_API_KEY)

A transcript is a list of segments: { speaker, start, end, text }. Diarization
is a separate concern; the stub labels a single speaker.
"""

from __future__ import annotations

import os
from typing import Any

ASR_BACKEND = os.environ.get("ASR_BACKEND", "stub")


def transcribe(audio_path: str, participants: int = 1) -> list[dict[str, Any]]:
    if ASR_BACKEND == "faster_whisper":
        return _faster_whisper(audio_path)
    if ASR_BACKEND == "openai":
        return _openai(audio_path)
    return _stub(audio_path, participants)


def _stub(audio_path: str, participants: int) -> list[dict[str, Any]]:
    # If the client uploaded a transcript text file, use it verbatim.
    if audio_path.endswith((".txt", ".md")):
        with open(audio_path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        return [{"speaker": "SPEAKER_0", "start": 0.0, "end": 0.0, "text": text}]
    # Otherwise return a placeholder so the pipeline is exercisable end to end.
    return [
        {
            "speaker": "SPEAKER_0",
            "start": 0.0,
            "end": 0.0,
            "text": "[stub transcript] set ASR_BACKEND=faster_whisper for real ASR.",
        }
    ]


def _faster_whisper(audio_path: str) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(os.environ.get("WHISPER_MODEL", "base"), compute_type="int8")
    segments, _info = model.transcribe(audio_path, word_timestamps=False)
    return [
        {"speaker": "SPEAKER_0", "start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]


def _openai(audio_path: str) -> list[dict[str, Any]]:
    import json
    import urllib.request

    key = os.environ["OPENAI_API_KEY"]
    boundary = "----voxscribe"
    with open(audio_path, "rb") as fh:
        audio = fh.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio\"\r\n".encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [{"speaker": "SPEAKER_0", "start": 0.0, "end": 0.0, "text": payload.get("text", "")}]
