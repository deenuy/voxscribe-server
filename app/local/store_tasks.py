"""PostgreSQL storage for tasks (a to-do list per scribe)."""

import re
import uuid

import psycopg

from .errors import ConflictError, EngineError, InvalidRequestError, NotFoundError
from .timefmt import iso_z

# The project shown is the task's own when it has one, otherwise its scribe's.
_COLUMNS = """t.id, t.meeting_id, COALESCE(t.project_id, m.project_id), t.title, t.detail, t.status, t.source, t.created_at, t.updated_at, t.completed_at, t.owner, t.due_date, t.revision, t.project_id"""
_FROM = "FROM tasks t JOIN meetings m ON m.id = t.meeting_id"


def title_key(title: str) -> str:
    """What makes two titles "the same task": lower case, single spaces."""
    return re.sub(r"\s+", " ", title.strip()).lower()


def _task(row) -> dict:
    return {
        "id": row[0],
        "meetingId": row[1],
        "projectId": row[2],
        "title": row[3],
        "detail": row[4],
        "status": row[5],
        "source": row[6],
        "createdAt": iso_z(row[7]),
        "updatedAt": iso_z(row[8]),
        "completedAt": iso_z(row[9]),
        "owner": row[10],
        "dueDate": row[11].isoformat() if row[11] else None,
        "revision": row[12],
        "ownProjectId": row[13],  # the project set on the task itself (null: it follows its scribe's)
    }


class TaskStore:
    def __init__(self, pool):
        self.pool = pool

    async def list_for_meeting(self, meeting_id: str) -> list[dict]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(f"SELECT {_COLUMNS} {_FROM} WHERE t.meeting_id = %s ORDER BY t.created_at, t.id", (meeting_id,))
            return [_task(r) for r in await cur.fetchall()]

    async def list_all(self, *, status: str | None, project_id: str | None, meeting_id: str | None, limit: int) -> tuple[list[dict], int]:
        where, args = [], []
        for clause, value in (("t.status = %s", status), ("COALESCE(t.project_id, m.project_id) = %s", project_id), ("t.meeting_id = %s", meeting_id)):
            if value is not None:
                where.append(clause)
                args.append(value)
        cond = f"WHERE {' AND '.join(where)}" if where else ""
        async with self.pool.connection() as conn:
            total = (await (await conn.execute(f"SELECT COUNT(*) {_FROM} {cond}", args)).fetchone())[0]
            cur = await conn.execute(f"SELECT {_COLUMNS} {_FROM} {cond} ORDER BY t.created_at DESC, t.id LIMIT %s", [*args, limit])
            return [_task(r) for r in await cur.fetchall()], total

    async def get(self, task_id: str) -> dict:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(f"SELECT {_COLUMNS} {_FROM} WHERE t.id = %s", (task_id,))).fetchone()
        if not row:
            raise NotFoundError(f"task {task_id} not found")
        return _task(row)

    async def create(self, meeting_id: str, *, title: str, detail: str | None = None, owner: str | None = None, due_date=None, project_id: str | None = None, status: str = "todo", source: str = "manual") -> dict:
        """Add a task. The same title in the same scribe is a 409 (a second tap of Add must not add it twice)."""
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO tasks (id, meeting_id, title, detail, source, title_key, owner, due_date, project_id, status, completed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CASE WHEN %s = 'done' THEN CURRENT_TIMESTAMP END)",
                    (task_id, meeting_id, title.strip(), detail, source, title_key(title), owner, due_date, project_id, status, status),
                )
                await conn.commit()
        except psycopg.errors.UniqueViolation:
            raise ConflictError("this scribe already has a task with that title") from None
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
        return await self.get(task_id)

    async def add_generated(self, meeting_id: str, drafts: list[dict]) -> tuple[list[dict], int]:
        """Add AI drafts, skipping any title the scribe already has (done, edited or manual ones are never touched).

        Safe to run twice, or twice at once: the unique index decides, so the second run adds nothing."""
        created_ids, skipped = [], 0
        async with self.pool.connection() as conn:
            for d in drafts:
                task_id = f"task_{uuid.uuid4().hex[:16]}"
                cur = await conn.execute(
                    "INSERT INTO tasks (id, meeting_id, title, detail, source, title_key, owner, due_date) VALUES (%s, %s, %s, %s, 'ai', %s, %s, %s) "
                    "ON CONFLICT (meeting_id, title_key) DO NOTHING RETURNING id",
                    (task_id, meeting_id, d["title"], d.get("detail"), title_key(d["title"]), d.get("owner"), d.get("dueDate")),
                )
                if await cur.fetchone():
                    created_ids.append(task_id)
                else:
                    skipped += 1
            await conn.commit()
        return [await self.get(i) for i in created_ids], skipped

    async def update(self, task_id: str, changes: dict, base_revision: int | None = None) -> dict:
        """Apply the fields in `changes` (keys: title, detail, owner, dueDate, projectId, status). A key that is present
        with None clears an optional field; only what is present changes."""
        sets, args = [], []
        for key, column in (("detail", "detail"), ("owner", "owner"), ("dueDate", "due_date"), ("projectId", "project_id")):
            if key in changes:
                sets.append(f"{column} = %s")
                args.append(changes[key])
        if "title" in changes:
            sets += ["title = %s", "title_key = %s"]
            args += [changes["title"].strip(), title_key(changes["title"])]
        if "status" in changes:
            sets += ["status = %s", "completed_at = CASE WHEN %s = 'done' THEN COALESCE(completed_at, CURRENT_TIMESTAMP) ELSE NULL END"]
            args += [changes["status"], changes["status"]]
        sets += ["updated_at = CURRENT_TIMESTAMP", "revision = revision + 1"]
        where, where_args = "id = %s", [task_id]
        if base_revision is not None:
            where += " AND revision = %s"
            where_args.append(base_revision)
        try:
            async with self.pool.connection() as conn:
                cur = await conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE {where} RETURNING id", [*args, *where_args])
                if not await cur.fetchone():
                    row = await (await conn.execute("SELECT revision FROM tasks WHERE id = %s", (task_id,))).fetchone()
                    if not row:
                        raise NotFoundError(f"task {task_id} not found")
                    raise EngineError("revision_conflict", f"this task changed (now revision {row[0]}); reload and try again", 409,
                                      {"items": [{"kind": "task", "id": task_id, "currentRevision": row[0]}]})
                await conn.commit()
        except psycopg.errors.UniqueViolation:
            raise ConflictError("this scribe already has a task with that title") from None
        except psycopg.errors.ForeignKeyViolation:
            raise InvalidRequestError("projectId: no such project") from None
        return await self.get(task_id)

    async def delete(self, task_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
            await conn.commit()
