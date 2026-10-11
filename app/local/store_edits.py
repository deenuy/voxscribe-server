"""One atomic commit of an edit session on a scribe.

A person edits a scribe as a whole (its title and project, transcript lines, the summary, its tasks) and presses Save
once. That must be one thing that either happens or does not: one transaction, so a failure part-way never leaves a
half-saved scribe, and one round trip. Every piece carries the revision the person last saw; if any piece changed
underneath them, nothing is written and the answer names the pieces that conflict. The scribe row is locked first, then
the rows being changed in a fixed order, so two commits cannot deadlock and cannot both win.

Safe to retry: with an Idempotency-Key, a commit that already succeeded answers with what it answered the first time
and changes nothing.
"""

import json
import uuid
from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from .edit_models import EditRequest
from .errors import EngineError, InvalidRequestError, NotFoundError
from .store_meetings import MAX_SEGMENT_CHARS, MAX_SPEAKER_CHARS, MeetingStore
from .store_tasks import _COLUMNS as TASK_COLUMNS
from .store_tasks import _FROM as TASK_FROM
from .store_tasks import _task, title_key
from .timefmt import iso_z


def _conflict(items: list[dict]) -> EngineError:
    kinds = sorted({i["kind"] for i in items})
    return EngineError(
        "revision_conflict",
        f"{', '.join(kinds)} changed while you were editing; nothing was saved. Reload to see the latest.",
        409,
        {"items": items},
    )


def _text(value: str | None) -> str | None:
    return (value or "").strip() or None


async def apply_edits(meetings: MeetingStore, meeting_id: str, req: EditRequest, idempotency_key: str | None = None) -> tuple[dict, bool]:
    """Apply `req` atomically. Returns (response, replayed)."""
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    async with meetings.pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            async with conn.transaction():
                m = await (await conn.execute("SELECT * FROM meetings WHERE id = %s FOR UPDATE", (meeting_id,))).fetchone()
                if not m:
                    raise NotFoundError(f"meeting {meeting_id} not found")

                if idempotency_key:
                    seen = await (await conn.execute(
                        "SELECT response FROM edit_commits WHERE idempotency_key = %s AND meeting_id = %s", (idempotency_key, meeting_id))).fetchone()
                    if seen:
                        return seen["response"], True

                done = m["status"] == "done"
                if req.segments and not done:
                    raise EngineError("transcript_not_editable", "the transcript can be edited once the meeting is stopped and published", 409)
                if req.summary and not done:
                    raise EngineError("summary_not_editable", "the summary can be edited once the meeting is stopped and published", 409)

                conflicts: list[dict] = []

                # ---- check every revision before writing anything
                if req.meeting and req.meeting.baseRevision != m["revision"]:
                    conflicts.append({"kind": "meeting", "id": meeting_id, "currentRevision": m["revision"]})

                segs: dict[str, dict] = {}
                if req.segments:
                    ids = sorted({s.id for s in req.segments})
                    rows = await (await conn.execute(
                        "SELECT * FROM transcript_segments WHERE meeting_id = %s AND id = ANY(%s) ORDER BY id FOR UPDATE", (meeting_id, ids))).fetchall()
                    segs = {r["id"]: r for r in rows}
                    for s in req.segments:
                        row = segs.get(s.id)
                        if row is None:
                            conflicts.append({"kind": "segment", "id": s.id, "reason": "missing"})
                        elif row["revision"] != s.baseRevision:
                            conflicts.append({"kind": "segment", "id": s.id, "currentRevision": row["revision"]})

                summary_row = None
                if req.summary:
                    summary_row = await (await conn.execute("SELECT * FROM summaries WHERE meeting_id = %s FOR UPDATE", (meeting_id,))).fetchone()
                    current = summary_row["revision"] if summary_row else 0
                    if current != req.summary.baseRevision:
                        conflicts.append({"kind": "summary", "id": meeting_id, "currentRevision": current})

                tasks = req.tasks
                task_rows: dict[str, dict] = {}
                if tasks and (tasks.update or tasks.delete):
                    ids = sorted({t.id for t in tasks.update} | {t.id for t in tasks.delete})
                    rows = await (await conn.execute(
                        "SELECT id, revision FROM tasks WHERE meeting_id = %s AND id = ANY(%s) ORDER BY id FOR UPDATE", (meeting_id, ids))).fetchall()
                    task_rows = {r["id"]: r for r in rows}
                    for t in tasks.update:
                        row = task_rows.get(t.id)
                        if row is None:
                            conflicts.append({"kind": "task", "id": t.id, "reason": "missing"})
                        elif row["revision"] != t.baseRevision:
                            conflicts.append({"kind": "task", "id": t.id, "currentRevision": row["revision"]})
                    for t in tasks.delete:
                        row = task_rows.get(t.id)
                        if row is not None and t.baseRevision is not None and row["revision"] != t.baseRevision:
                            conflicts.append({"kind": "task", "id": t.id, "currentRevision": row["revision"]})
                if conflicts:
                    raise _conflict(conflicts)

                # ---- validate content (nothing has been written yet)
                for s in req.segments:
                    if s.text is not None:
                        if not s.text.strip():
                            raise InvalidRequestError(f"segments: line {s.id} cannot be empty")
                        if len(s.text.strip()) > MAX_SEGMENT_CHARS:
                            raise InvalidRequestError(f"segments: line {s.id} is longer than {MAX_SEGMENT_CHARS} characters")
                    if s.text is None and "speaker" not in s.model_fields_set:
                        raise InvalidRequestError(f"segments: line {s.id} has nothing to change")
                    if len((s.speaker or "").strip()) > MAX_SPEAKER_CHARS:
                        raise InvalidRequestError(f"segments: the speaker of line {s.id} is too long")
                if req.meeting and "title" in req.meeting.model_fields_set and not (req.meeting.title or "").strip():
                    raise InvalidRequestError("meeting.title: cannot be blank")
                if req.summary:
                    for d in req.summary.details:
                        if not d.title.strip() or not d.body.strip():
                            raise InvalidRequestError("summary.details: every section needs a title and text")
                    if not req.summary.overview.strip():
                        raise InvalidRequestError("summary.overview: cannot be blank")

                # ---- write
                changed = {"meeting": False, "segments": 0, "summary": False, "tasks": 0}
                if req.meeting:
                    fields = req.meeting.model_fields_set
                    sets, args = [], []
                    if "title" in fields:
                        sets.append("title = %s")
                        args.append(req.meeting.title.strip())
                    if "context" in fields:
                        sets.append("context = %s")
                        args.append(_text(req.meeting.context))
                    if "projectId" in fields:
                        sets.append("project_id = %s")
                        args.append(_text(req.meeting.projectId))
                    if sets:
                        await conn.execute(f"UPDATE meetings SET {', '.join(sets)}, revision = revision + 1, updated_at = %s WHERE id = %s", [*args, now, meeting_id])
                        changed["meeting"] = True

                for s in req.segments:
                    new_text = s.text.strip() if s.text is not None else None
                    new_speaker = _text(s.speaker) if "speaker" in s.model_fields_set else None
                    if await meetings._apply_segment_edit(conn, meeting_id, segs[s.id], new_text, "speaker" in s.model_fields_set, new_speaker, now):
                        changed["segments"] += 1

                if req.summary:
                    details = [{"title": d.title.strip(), "body": d.body.strip()} for d in req.summary.details]
                    await conn.execute(
                        """INSERT INTO summaries (meeting_id, status, overview, details, generated_at, created_at, updated_at, revision, edited, edited_at)
                           VALUES (%s, 'ready', %s, %s, %s, %s, %s, 1, true, %s)
                           ON CONFLICT (meeting_id) DO UPDATE SET status = 'ready', overview = EXCLUDED.overview, details = EXCLUDED.details,
                               revision = summaries.revision + 1, edited = true, edited_at = EXCLUDED.edited_at, updated_at = EXCLUDED.updated_at""",
                        (meeting_id, req.summary.overview.strip(), json.dumps(details), now, now, now, now),
                    )
                    await conn.execute("UPDATE meetings SET summary_stale = false, updated_at = %s WHERE id = %s", (now, meeting_id))
                    changed["summary"] = True

                created: dict[str, str] = {}
                if tasks:
                    for t in tasks.delete:
                        await conn.execute("DELETE FROM tasks WHERE id = %s AND meeting_id = %s", (t.id, meeting_id))
                        changed["tasks"] += 1
                    for t in tasks.update:
                        sets, args = [], []
                        for key, col in (("detail", "detail"), ("owner", "owner"), ("dueDate", "due_date"), ("projectId", "project_id")):
                            if key in t.model_fields_set:
                                value = getattr(t, key)
                                sets.append(f"{col} = %s")
                                args.append(value if key == "dueDate" else _text(value))
                        if "title" in t.model_fields_set and t.title is not None:
                            if not t.title.strip():
                                raise InvalidRequestError(f"tasks: the title of {t.id} cannot be blank")
                            sets += ["title = %s", "title_key = %s"]
                            args += [t.title.strip(), title_key(t.title)]
                        if "status" in t.model_fields_set and t.status is not None:
                            sets += ["status = %s", "completed_at = CASE WHEN %s = 'done' THEN COALESCE(completed_at, CURRENT_TIMESTAMP) ELSE NULL END"]
                            args += [t.status, t.status]
                        if sets:
                            try:
                                async with conn.transaction():  # a savepoint, so the clash can be named
                                    await conn.execute(f"UPDATE tasks SET {', '.join(sets)}, updated_at = CURRENT_TIMESTAMP, revision = revision + 1 WHERE id = %s", [*args, t.id])
                            except psycopg.errors.UniqueViolation:
                                raise _conflict([{"kind": "task", "id": t.id, "reason": "duplicate_title"}]) from None
                            changed["tasks"] += 1
                    for t in tasks.create:
                        task_id = f"task_{uuid.uuid4().hex[:16]}"
                        try:
                            async with conn.transaction():
                                await conn.execute(
                                    "INSERT INTO tasks (id, meeting_id, title, detail, source, title_key, owner, due_date, project_id, status, completed_at) "
                                    "VALUES (%s, %s, %s, %s, 'manual', %s, %s, %s, %s, %s, CASE WHEN %s = 'done' THEN CURRENT_TIMESTAMP END)",
                                    (task_id, meeting_id, t.title.strip(), _text(t.detail), title_key(t.title), _text(t.owner), t.dueDate, _text(t.projectId), t.status, t.status),
                                )
                        except psycopg.errors.UniqueViolation:
                            raise _conflict([{"kind": "task", "tempId": t.tempId, "reason": "duplicate_title"}]) from None
                        created[t.tempId] = task_id
                        changed["tasks"] += 1

                # ---- the fresh state, read inside the same transaction
                fresh = await (await conn.execute("SELECT * FROM meetings WHERE id = %s", (meeting_id,))).fetchone()
                summary = await (await conn.execute("SELECT * FROM summaries WHERE meeting_id = %s", (meeting_id,))).fetchone()
                plain = psycopg.rows.tuple_row
                conn.row_factory = plain
                task_rows_out = await (await conn.execute(f"SELECT {TASK_COLUMNS} {TASK_FROM} WHERE t.meeting_id = %s ORDER BY t.created_at, t.id", (meeting_id,))).fetchall()
                conn.row_factory = dict_row
                response = {
                    "meeting": {
                        "id": meeting_id,
                        "title": fresh["title"],
                        "context": fresh["context"],
                        "projectId": fresh["project_id"],
                        "revision": fresh["revision"],
                        "transcriptRevision": fresh["transcript_revision"],
                        "summaryStale": bool(fresh["summary_stale"]),
                        "wordCount": fresh["word_count"],
                        "updatedAt": iso_z(fresh["updated_at"]),
                    },
                    "summary": None if not summary else {
                        "status": summary["status"], "overview": summary["overview"], "details": summary["details"] or [],
                        "revision": summary["revision"], "edited": bool(summary["edited"]), "generatedAt": iso_z(summary["generated_at"]),
                    },
                    "tasks": [_task(r) for r in task_rows_out],
                    "createdTasks": created,
                    "changed": changed,
                }
                if idempotency_key:
                    await conn.execute("DELETE FROM edit_commits WHERE created_at < CURRENT_TIMESTAMP - interval '1 day'")
                    await conn.execute("INSERT INTO edit_commits (idempotency_key, meeting_id, response) VALUES (%s, %s, %s)", (idempotency_key, meeting_id, json.dumps(response)))
                return response, False
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
