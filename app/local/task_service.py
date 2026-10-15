"""Drafting tasks from a transcript.

The model is asked for a short list of things someone agreed or was asked to do, never for anything it was not told
(a made-up task is worse than none). Its answer is parsed defensively: a reply that is not usable is an error the
person can retry, not an empty list that looks like "nothing to do"."""

import asyncio
import re
from datetime import date

from voxscribe.meeting.llm_json import complete_json

from .errors import EngineError, ProviderUnavailableError

MAX_TASKS = 5  # SMART ceiling: max 5 tasks regardless of transcript length (see _task_ceiling)
# Room for a model that reasons before it answers: that reasoning counts against the limit, and a reply cut off
# before the answer is empty. Not retried automatically: a retry would double a wait that is already long.
MAX_REPLY_TOKENS = 4096
MAX_TRANSCRIPT_CHARS = 30000

# Word count → max task ceiling (never a target; return fewer if fewer real actions exist)
def _task_ceiling(word_count: int) -> int:
    if word_count < 500:
        return 1
    if word_count < 1500:
        return 2
    if word_count < 3000:
        return 3
    if word_count < 6000:
        return 4
    return 5

SYSTEM = """\
You are a task extraction assistant. Read the transcript and produce a short, high-quality to-do list.

Rules:
- Only use commitments, decisions, or clear next steps that appear in the transcript. Do not invent work.
- Every task must be SMART: Specific (verb-first title), Measurable (a done condition), Achievable, Relevant, Time-bound.
- Give an owner only when the transcript names who is responsible. Otherwise omit it.
- Give a dueDate (YYYY-MM-DD) only when a deadline is stated or clearly implied. Use the conversation date to resolve \
relative words like "Friday" or "next week". Otherwise omit it.
- The detail field must contain: "Done when: [measurable condition]. Why: [one line tied to a stated goal]."
- Merge near-duplicates. If an item is too vague, skip it rather than guessing.
- If there are no actionable items, return an empty list.
- Output ONLY valid JSON (no markdown): \
{"tasks": [{"title": "verb-first string ≤120 chars", "detail": "Done when: … Why: …", "owner": "name or null", "dueDate": "YYYY-MM-DD or null"}]}\
"""

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _real_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


_CUES = re.compile(r"\b(need to|needs to|should|must|have to|let's|will|going to|remember to|don't forget|action item|todo|follow up)\b", re.I)


def build_prompt(lines: list[str], project: dict | None, held_on: str | None = None) -> tuple[str, int]:
    """Returns (prompt_text, word_count) so callers can compute the ceiling."""
    transcript = "\n".join(lines)[:MAX_TRANSCRIPT_CHARS]
    word_count = len(transcript.split())
    ceiling = _task_ceiling(word_count)
    head = ""
    if project:
        head = f"This conversation belongs to the project \"{project['name']}\""
        head += f": {project['description']}.\n\n" if project.get("description") else ".\n\n"
    when = f"The conversation took place on {held_on}.\n\n" if held_on else ""
    limit_line = f"Return at most {ceiling} task(s) based on transcript length ({word_count} words).\n\n"
    return f"{head}{when}{limit_line}Transcript:\n{transcript}", word_count


def clean_drafts(raw, ceiling: int = MAX_TASKS) -> list[dict]:
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
        owner = (
            re.sub(r"\s+", " ", owner).strip()[:120]
            if isinstance(owner, str) and owner.strip() and owner.strip().lower() not in ("null", "n/a", "none", "")
            else None
        )
        due = item.get("dueDate")
        due = due.strip() if isinstance(due, str) and _DATE.fullmatch(due.strip()) and _real_date(due.strip()) else None
        seen.add(key)
        out.append({"title": title[:300], "detail": detail, "owner": owner, "dueDate": due})
        if len(out) >= ceiling:
            break
    return out


def extractive_drafts(lines: list[str]) -> list[dict]:
    """The built-in test engine has no model: pick sentences that sound like commitments. Deterministic."""
    drafts = []
    for line in lines:
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if _CUES.search(sentence) and 8 <= len(sentence) <= 200:
                drafts.append({"title": sentence.strip().rstrip(".!?"), "detail": None, "owner": None, "dueDate": None})
    word_count = len(" ".join(lines).split())
    return clean_drafts(drafts, ceiling=_task_ceiling(word_count))


async def draft_tasks(llm, *, lines: list[str], project: dict | None, fake: bool, held_on: str | None = None, labelled: list[str] | None = None) -> list[dict]:
    if fake:
        return extractive_drafts(lines)
    if llm is None:
        raise ProviderUnavailableError("the on-device AI is not available")
    prompt_text, word_count = build_prompt(labelled or lines, project, held_on)
    try:
        parsed, raw = await asyncio.to_thread(complete_json, llm, SYSTEM, prompt_text, MAX_REPLY_TOKENS, 0)
    except Exception:
        raise ProviderUnavailableError("the on-device AI could not be reached") from None
    if parsed is None or not isinstance(parsed.get("tasks"), list):
        raise EngineError("ai_unreadable", "the AI did not return a usable task list; try again", 502)
    return clean_drafts(parsed["tasks"], ceiling=_task_ceiling(word_count))
