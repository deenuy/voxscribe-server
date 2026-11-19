"""Drafting tasks from a transcript.

The model is asked for a short list of things someone agreed or was asked to do, never for anything it was not told
(a made-up task is worse than none). Its answer is parsed defensively: a reply that is not usable is an error the
person can retry, not an empty list that looks like "nothing to do".

draft_and_store() is the one way tasks are drafted and kept, after a summary and from the Generate route: it
reports the tasks step to the pipeline (working, then ready or failed with a reason) and bounds the model's time."""

import asyncio
import os
import re
from datetime import date

from voxscribe.meeting.llm_json import complete_json

from . import pipeline
from .errors import EngineError, ProviderUnavailableError
from .store_tasks import TaskStore

MAX_TASKS = 5  # SMART ceiling: max 5 tasks regardless of transcript length (see _task_ceiling)
# Room for a model that reasons before it answers: that reasoning counts against the limit, and a reply cut off
# before the answer is empty. Not retried automatically: a retry would double a wait that is already long.
MAX_REPLY_TOKENS = 4096
MAX_TRANSCRIPT_CHARS = 30000
# Wall-clock cap on drafting. The same model as the summary and a prompt of the same size, so the same cap by default.
TASKS_TIMEOUT_S = int(os.getenv("VOXSCRIBE_TASKS_TIMEOUT_S", os.getenv("VOXSCRIBE_SUMMARY_TIMEOUT_S", "120")))

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


def failure_reason(exc: BaseException) -> str:
    """The pipeline reason for a drafting failure: a code, never the exception's text."""
    if isinstance(exc, asyncio.TimeoutError):
        return pipeline.MODEL_TIMEOUT
    if isinstance(exc, ProviderUnavailableError):
        return pipeline.MODEL_UNAVAILABLE
    if isinstance(exc, EngineError) and exc.code == "ai_unreadable":
        return pipeline.UNUSABLE_OUTPUT
    return pipeline.MODEL_ERROR


async def draft_and_store(pool, llm, meeting_id: str, *, lines: list[str], labelled: list[str] | None,
                          project: dict | None, fake: bool, held_on: str | None,
                          timeout: float | None = None) -> tuple[list, int]:
    """Draft tasks and add the new ones to the scribe. Returns (created, skipped).

    Reports the tasks step: working, then ready, or failed with failure_reason(); the error is raised again for the
    caller. A timeout (TASKS_TIMEOUT_S unless given) is asyncio.TimeoutError, reported as model_timeout."""
    run = pipeline.start_run(meeting_id, "tasks")
    await pipeline.report(pool, meeting_id, "tasks", pipeline.WORKING)
    try:
        drafts = await asyncio.wait_for(
            draft_tasks(llm, lines=lines, labelled=labelled, project=project, fake=fake, held_on=held_on),
            timeout=TASKS_TIMEOUT_S if timeout is None else timeout,
        )
        created, skipped = await TaskStore(pool).add_generated(meeting_id, drafts)
    except asyncio.CancelledError:
        raise                               # an erase or a shutdown: nothing more is written for this scribe
    except Exception as exc:
        await pipeline.report(pool, meeting_id, "tasks", pipeline.FAILED, failure_reason(exc))
        raise
    finally:
        pipeline.end_run(meeting_id, "tasks", run)
    await pipeline.report(pool, meeting_id, "tasks", pipeline.READY)
    return created, skipped
