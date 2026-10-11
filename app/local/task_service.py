"""Drafting tasks from a transcript.

The model is asked for a short list of things someone agreed or was asked to do, never for anything it was not told
(a made-up task is worse than none). Its answer is parsed defensively: a reply that is not usable is an error the
person can retry, not an empty list that looks like "nothing to do"."""

import asyncio
import re
from datetime import date

from voxscribe.meeting.llm_json import complete_json

from .errors import EngineError, ProviderUnavailableError

MAX_TASKS = 20
# Room for a model that reasons before it answers: that reasoning counts against the limit, and a reply cut off
# before the answer is empty. Not retried automatically: a retry would double a wait that is already long.
MAX_REPLY_TOKENS = 4096
MAX_TRANSCRIPT_CHARS = 30000

SYSTEM = (
    "You turn a recorded conversation into a to-do list. List only things that someone agreed to do, was asked to do, "
    "or clearly needs to be done next. Do not invent tasks and do not list general discussion. Each task has a short "
    "title that starts with a verb (under 120 characters) and optionally one sentence of detail. Give an owner only when "
    "the conversation names who is responsible, and a dueDate (YYYY-MM-DD) only when a date or day is stated, using the "
    "date of the conversation to resolve words like Friday or next week; otherwise leave both empty. If there are no "
    'tasks, return an empty list. Output ONLY valid JSON: {"tasks": [{"title": string, "detail": string, "owner": string, "dueDate": string}]}.'
)

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _real_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


_CUES = re.compile(r"\b(need to|needs to|should|must|have to|let's|will|going to|remember to|don't forget|action item|todo|follow up)\b", re.I)


def build_prompt(lines: list[str], project: dict | None, held_on: str | None = None) -> str:
    transcript = "\n".join(lines)[:MAX_TRANSCRIPT_CHARS]
    head = ""
    if project:
        head = f"This conversation belongs to the project \"{project['name']}\""
        head += f": {project['description']}.\n\n" if project.get("description") else ".\n\n"
    when = f"The conversation took place on {held_on}.\n\n" if held_on else ""
    return f"{head}{when}Transcript:\n{transcript}"


def clean_drafts(raw) -> list[dict]:
    """Keep well-formed drafts only: a string title (trimmed, at most 300), an optional detail, no repeats."""
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("title"), str):
            continue
        title = re.sub(r"\s+", " ", item["title"]).strip().rstrip(".")
        key = title.lower()
        if not title or key in seen:
            continue
        detail = item.get("detail")
        detail = re.sub(r"\s+", " ", detail).strip()[:2000] if isinstance(detail, str) and detail.strip() else None
        owner = item.get("owner")
        owner = re.sub(r"\s+", " ", owner).strip()[:120] if isinstance(owner, str) and owner.strip() else None
        due = item.get("dueDate")
        due = due.strip() if isinstance(due, str) and _DATE.fullmatch(due.strip()) and _real_date(due.strip()) else None
        seen.add(key)
        out.append({"title": title[:300], "detail": detail, "owner": owner, "dueDate": due})
        if len(out) >= MAX_TASKS:
            break
    return out


def extractive_drafts(lines: list[str]) -> list[dict]:
    """The built-in test engine has no model: pick sentences that sound like commitments. Deterministic."""
    drafts = []
    for line in lines:
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if _CUES.search(sentence) and 8 <= len(sentence) <= 200:
                drafts.append({"title": sentence.strip().rstrip(".!?"), "detail": None, "owner": None, "dueDate": None})
    return clean_drafts(drafts)


async def draft_tasks(llm, *, lines: list[str], project: dict | None, fake: bool, held_on: str | None = None, labelled: list[str] | None = None) -> list[dict]:
    if fake:
        return extractive_drafts(lines)
    if llm is None:
        raise ProviderUnavailableError("the on-device AI is not available")
    try:
        parsed, raw = await asyncio.to_thread(complete_json, llm, SYSTEM, build_prompt(labelled or lines, project, held_on), MAX_REPLY_TOKENS, 0)
    except Exception:
        raise ProviderUnavailableError("the on-device AI could not be reached") from None
    if parsed is None or not isinstance(parsed.get("tasks"), list):
        raise EngineError("ai_unreadable", "the AI did not return a usable task list; try again", 502)
    return clean_drafts(parsed["tasks"])
