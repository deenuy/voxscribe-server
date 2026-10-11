"""Turning a transcript someone already has into the lines of a scribe.

Understands the formats people actually have: SubRip (.srt), WebVTT (.vtt), and plain text or Markdown, with or without
times ("[00:12] Alice: ...") and speaker names. Anything with no times gets estimated ones, spaced by how long the words
take to say, so the lines keep their order and every line has a distinct moment.

Pure functions, no database: easy to test, and the same for a pasted text, a file chosen in the browser and a captions
download.
"""

import re

MAX_CHARS = 5_000_000  # a very long conversation is well under this
MAX_LINES = 20_000
MAX_LINE_CHARS = 20_000
WORDS_PER_SECOND = 2.5

_TIME = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_CUE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
_CUE_ANYWHERE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}", re.M)
_STAMP = re.compile(rf"^\s*[\[(]?{_TIME}[\])]?\s*[-–:]?\s*(.*)$")
_SPEAKER = re.compile(r"^([^\W\d_][\w.'’-]*(?: [\w.'’-]+){0,3}):\s+(\S.*)$")
_TAGS = re.compile(r"<[^>]+>")
_VOICE = re.compile(r"<v(?:\.[^ >]+)?\s+([^>]+)>")


class TranscriptError(ValueError):
    """The text could not be read as a transcript."""


def _ms(h, m, s, frac) -> int:
    millis = int((frac or "0").ljust(3, "0")[:3])
    return ((int(h or 0) * 60 + int(m)) * 60 + int(s)) * 1000 + millis


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub("", text)).strip()


def _split_long(text: str) -> list[str]:
    """A line longer than a database row allows is cut at sentence ends."""
    if len(text) <= MAX_LINE_CHARS:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > MAX_LINE_CHARS:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return [p[:MAX_LINE_CHARS] for p in parts]


def _speaker_prefix(text: str) -> tuple[str | None, str]:
    m = _SPEAKER.match(text)
    return (m.group(1), m.group(2)) if m else (None, text)


def _estimate(lines: list[dict]) -> list[dict]:
    """Give lines with no time one, in order, spaced by how long the words take to say."""
    clock = 0
    out = []
    for line in lines:
        if line.get("startMs") is None:
            words = max(1, len(line["text"].split()))
            start = clock
            end = start + max(2000, int(words / WORDS_PER_SECOND * 1000))
        else:
            start = max(line["startMs"], clock)  # never earlier than the line before
            end = max(line.get("endMs") or start, start + 1)
        out.append({"startMs": start, "endMs": end, "speaker": line.get("speaker"), "text": line["text"]})
        clock = end
    return out


def parse_cues(text: str) -> list[dict]:
    """SubRip or WebVTT: blocks of an optional id, a `start --> end` line, then the words."""
    lines = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        rows = [r for r in block.split("\n") if r.strip()]
        i = next((n for n, r in enumerate(rows) if _CUE.match(r)), None)
        if i is None:
            continue
        m = _CUE.match(rows[i])
        start = _ms(*m.group(1, 2, 3, 4))
        end = _ms(*m.group(5, 6, 7, 8))
        body = " ".join(rows[i + 1:])
        voice = _VOICE.search(body)
        speaker = voice.group(1).strip() if voice else None
        words = _clean(body)
        if not words:
            continue
        if speaker is None:
            speaker, words = _speaker_prefix(words)
        lines.append({"startMs": start, "endMs": end, "speaker": speaker, "text": words})
    return lines


def parse_plain(text: str) -> list[dict]:
    """Plain text or Markdown: one line per paragraph, or per line when lines carry their own times or speakers."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    rows = [r.strip() for r in raw.split("\n")]
    structured = sum(1 for r in rows if _STAMP.match(r) or _SPEAKER.match(r)) >= max(2, len([r for r in rows if r]) // 3)
    units: list[str] = [r for r in rows if r] if structured else [" ".join(p.split("\n")).strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    lines = []
    for unit in units:
        unit = re.sub(r"^\s*(#{1,6}\s+|[-*+]\s+|>\s+)", "", unit)  # Markdown headings, bullets, quotes
        start = None
        m = _STAMP.match(unit)
        if m:
            start = _ms(*m.group(1, 2, 3, 4))
            unit = m.group(5)
        unit = _clean(unit)
        if not unit:
            continue
        # A name before a colon is only a speaker when the lines are written that way; in ordinary prose "Note: ..."
        # is just a sentence.
        speaker, words = _speaker_prefix(unit) if structured else (None, unit)
        for piece in _split_long(words):
            lines.append({"startMs": start, "speaker": speaker, "text": piece})
            start = None  # only the first piece of a split line keeps the time
    return lines


def looks_like_cues(text: str, filename: str | None) -> bool:
    name = (filename or "").lower()
    return name.endswith((".srt", ".vtt")) or text.lstrip().startswith("WEBVTT") or bool(_CUE_ANYWHERE.search(text[:5000]))


def parse_transcript(text: str, filename: str | None = None) -> list[dict]:
    """The lines of a transcript: [{startMs, endMs, speaker, text}], in order, with a time on every one."""
    if not isinstance(text, str) or not text.strip():
        raise TranscriptError("the transcript is empty")
    if len(text) > MAX_CHARS:
        raise TranscriptError("the transcript is too long to import")
    text = text.lstrip("﻿")
    lines = parse_cues(text) if looks_like_cues(text, filename) else parse_plain(text)
    lines = [piece for line in lines for piece in ({**line, "text": t} for t in _split_long(line["text"]))]
    if not lines:
        raise TranscriptError("no spoken lines were found in this file")
    if len(lines) > MAX_LINES:
        raise TranscriptError(f"the transcript has more than {MAX_LINES} lines")
    return _estimate(lines)
