"""Importing a YouTube video as a scribe, from the video's captions.

The captions are the transcript the video's owner (or YouTube) already made, so nothing is downloaded or transcribed
locally. This needs the network, so it is only allowed when the person has allowed connectors.
"""

import asyncio
import re
from urllib.parse import parse_qs, urlparse

import httpx

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(ValueError):
    """The link is not a YouTube video, or the video cannot be imported."""


def video_id(url: str) -> str:
    """The 11-character id from a watch, share, shorts, embed or live link."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        raise YouTubeError("that is not a valid link") from None
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                candidate = parts[1]
    if not candidate or not _ID.match(candidate):
        raise YouTubeError("that is not a link to a YouTube video")
    return candidate


async def video_title(vid: str) -> str | None:
    """The video's title (for the scribe's name), or None when it cannot be fetched."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            res = await client.get("https://www.youtube.com/oembed", params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"})
            res.raise_for_status()
            title = res.json().get("title")
            return title.strip() if isinstance(title, str) and title.strip() else None
    except Exception:  # noqa: BLE001 - a missing title only costs the default name
        return None


def _fetch_captions(vid: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    listing = api.list(vid)
    try:
        transcript = listing.find_transcript(["en"])
    except Exception:  # noqa: BLE001 - not in English: take whatever the video has
        transcript = next(iter(listing), None)
    if transcript is None:
        raise YouTubeError("this video has no captions to import")
    return [{"text": s.text, "start": float(s.start), "duration": float(s.duration)} for s in transcript.fetch()]


async def fetch_captions(vid: str, fetch=_fetch_captions) -> list[dict]:
    """The video's captions as [{text, start, duration}] (seconds). Raises YouTubeError with something a person can read."""
    try:
        snippets = await asyncio.to_thread(fetch, vid)
    except YouTubeError:
        raise
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "Disabled" in name or "NoTranscript" in name:
            raise YouTubeError("this video has no captions to import") from None
        if "Unavailable" in name or "Private" in name or "NotFound" in name:
            raise YouTubeError("this video is not available") from None
        if "Blocked" in name or "RequestBlocked" in name or "IpBlocked" in name:
            raise YouTubeError("YouTube is not answering requests from this computer right now; try again later") from None
        raise YouTubeError("the captions could not be fetched") from None
    if not snippets:
        raise YouTubeError("this video has no captions to import")
    return snippets


def captions_to_lines(snippets: list[dict]) -> list[dict]:
    """Captions are cut every few seconds, mid-sentence: join them into lines at sentence ends, keeping the first time."""
    lines, text, start, end = [], "", None, None
    for s in snippets:
        piece = re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", s["text"])).strip()  # [Music], [Applause]
        if not piece:
            continue
        if start is None:
            start = int(s["start"] * 1000)
        text = f"{text} {piece}".strip()
        end = int((s["start"] + s["duration"]) * 1000)
        if re.search(r"[.!?]['\")\]]?$", piece) or len(text) > 400:
            lines.append({"startMs": start, "endMs": max(end, start + 1), "speaker": None, "text": text})
            text, start = "", None
    if text:
        lines.append({"startMs": start, "endMs": max(end or start, start + 1), "speaker": None, "text": text})
    return lines
