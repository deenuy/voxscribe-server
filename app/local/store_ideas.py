"""PostgreSQL storage for ideas."""

import json
from datetime import UTC, datetime

from psycopg_pool import AsyncConnectionPool

from .errors import NotFoundError


def _iso_timestamp() -> str:
    """Return current timestamp in ISO format with Z suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _iso_datetime(dt: "datetime | str | None") -> str | None:
    """Convert a datetime to ISO format with a Z suffix. A value that is already text passes through."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat().replace("+00:00", "Z")


class IdeaStore:
    """PostgreSQL-backed ideas storage."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_idea(
        self,
        idea_id: str,
        title: str,
        summary: str = "",
        type_: str = "new_feature",
        project_id: str = None,
    ) -> dict:
        """Create a manual idea."""
        now = datetime.now(UTC)
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ideas (id, title, summary, type, status, score, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (idea_id, title, summary, type_, "raw", 0, now, now),
            )
            await conn.commit()
        return await self.get_idea(idea_id)

    async def get_idea(self, idea_id: str) -> dict:
        """Get an idea by ID."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM ideas WHERE id = %s", (idea_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise NotFoundError(f"idea {idea_id} not found")
            return _idea_row_to_dict(row, cursor.description)

    async def list_ideas(
        self,
        query: str = None,
        status: str = None,
        type_: str = None,
        project_id: str = None,
        min_score: int = None,
        max_score: int = None,
        sort: str = "updated",
        order: str = "desc",
        limit: int = 25,
        cursor: str = None,
    ) -> tuple[list[dict], str | None]:
        """List ideas with filtering and pagination."""
        async with self.pool.connection() as conn:
            where_clauses = []
            params = []

            if query:
                where_clauses.append(
                    "to_tsvector('english', title || ' ' || COALESCE(summary, '')) @@ websearch_to_tsquery(%s)"
                )
                params.append(query)

            if status:
                where_clauses.append("status = %s")
                params.append(status)

            if type_:
                where_clauses.append("type = %s")
                params.append(type_)

            if project_id:
                where_clauses.append("project_id = %s")
                params.append(project_id)

            if min_score is not None:
                where_clauses.append("score >= %s")
                params.append(min_score)

            if max_score is not None:
                where_clauses.append("score <= %s")
                params.append(max_score)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Sort options
            sort_col = "updated_at" if sort == "updated" else (
                "score" if sort == "score" else "title"
            )
            order_sql = "DESC" if order == "desc" else "ASC"

            # Cursor pagination
            cursor_sql = ""
            if cursor:
                if sort == "score":
                    cursor_sql = f"AND score {'<' if order == 'desc' else '>'} %s"
                    params.append(int(cursor))
                else:
                    cursor_sql = f"AND {sort_col} {'<' if order == 'desc' else '>'} %s"
                    params.append(cursor)

            sql = f"""
            SELECT * FROM ideas
            WHERE {where_sql} {cursor_sql}
            ORDER BY {sort_col} {order_sql}
            LIMIT %s
            """
            params.append(limit + 1)  # Fetch one extra to check if there are more

            cursor_obj = await conn.execute(sql, params)
            rows = await cursor_obj.fetchall()

            next_cursor = None
            if len(rows) > limit:
                rows = rows[:limit]
                last_row = rows[-1]
                if sort == "score":
                    next_cursor = str(last_row[9])  # score column
                else:
                    next_cursor = last_row[13]  # updated_at column

            ideas = [_idea_row_to_dict(row, cursor_obj.description) for row in rows]
            return ideas, next_cursor

    async def update_idea(self, idea_id: str, **kwargs) -> dict:
        """Update an idea."""
        allowed_fields = {
            "title",
            "summary",
            "type",
            "project_id",
            "feature_id",
            "owner",
            "rubric",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return await self.get_idea(idea_id)

        now = datetime.now(UTC)
        updates["edited"] = True
        updates["updated_at"] = now

        # Recompute score and status if rubric was updated
        if "rubric" in updates:
            from voxscribe.ideas.scoring import compute_weighted_score, score_to_status

            rubric = updates["rubric"]
            if isinstance(rubric, str):
                rubric = json.loads(rubric)
            score = compute_weighted_score(rubric)
            updates["score"] = score
            updates["status"] = score_to_status(score)

        set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
        values = list(updates.values()) + [idea_id]

        async with self.pool.connection() as conn:
            await conn.execute(
                f"UPDATE ideas SET {set_clause} WHERE id = %s", values
            )
            await conn.commit()
        return await self.get_idea(idea_id)

    async def delete_idea(self, idea_id: str) -> None:
        """Delete an idea."""
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM ideas WHERE id = %s", (idea_id,))
            await conn.commit()


class ProjectStore:
    """PostgreSQL-backed projects storage."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_project(
        self, project_id: str, name: str, description: str = None, parent_id: str = None
    ) -> dict:
        """Create a project."""
        now = datetime.now(UTC)
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO projects (id, name, description, parent_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (project_id, name, description, parent_id, now, now),
            )
            await conn.commit()
        return await self.get_project(project_id)

    async def get_project(self, project_id: str) -> dict:
        """Get a project by ID."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT p.*, COUNT(i.id) as idea_count
                FROM projects p
                LEFT JOIN ideas i ON p.id = i.project_id
                WHERE p.id = %s
                GROUP BY p.id
                """,
                (project_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise NotFoundError(f"project {project_id} not found")

            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "parentId": row[3],
                "createdAt": _iso_datetime(row[4]),
                "ideaCount": row[6],
            }

    async def list_projects(self, limit: int = 25, cursor: str = None) -> tuple[list[dict], str | None]:
        """List all projects."""
        async with self.pool.connection() as conn:
            sql = """
            SELECT p.*, COUNT(i.id) as idea_count
            FROM projects p
            LEFT JOIN ideas i ON p.id = i.project_id
            WHERE parent_id IS NULL
            GROUP BY p.id
            ORDER BY p.created_at DESC
            LIMIT %s
            """

            cursor_obj = await conn.execute(sql, (limit + 1,))
            rows = await cursor_obj.fetchall()

            next_cursor = None
            if len(rows) > limit:
                rows = rows[:limit]
                next_cursor = rows[-1][0]

            projects = []
            for row in rows:
                projects.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "parentId": row[3],
                    "createdAt": _iso_datetime(row[4]),
                    "ideaCount": row[6],
                })

            return projects, next_cursor

    async def update_project(self, project_id: str, **kwargs) -> dict:
        """Update a project."""
        allowed_fields = {"name", "description"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return await self.get_project(project_id)

        now = datetime.now(UTC)
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
        values = list(updates.values()) + [project_id]

        async with self.pool.connection() as conn:
            await conn.execute(
                f"UPDATE projects SET {set_clause} WHERE id = %s", values
            )
            await conn.commit()
        return await self.get_project(project_id)

    async def delete_project(self, project_id: str) -> None:
        """Delete a project if it has no children."""
        async with self.pool.connection() as conn:
            # Check for children
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM projects WHERE parent_id = %s",
                (project_id,),
            )
            count = (await cursor.fetchone())[0]
            if count > 0:
                from .errors import ConflictError

                raise ConflictError("cannot delete project with children")

            await conn.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            await conn.commit()


class FeatureStore:
    """PostgreSQL-backed features storage."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_feature(self, feature_id: str, project_id: str, name: str) -> dict:
        """Create a feature."""
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO features (id, project_id, name)
                VALUES (%s, %s, %s)
                """,
                (feature_id, project_id, name),
            )
            await conn.commit()
        return await self.get_feature(feature_id)

    async def get_feature(self, feature_id: str) -> dict:
        """Get a feature by ID."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM features WHERE id = %s", (feature_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise NotFoundError(f"feature {feature_id} not found")

            return {
                "id": row[0],
                "projectId": row[1],
                "name": row[2],
            }

    async def list_features(self, project_id: str) -> list[dict]:
        """List features for a project."""
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM features WHERE project_id = %s ORDER BY name",
                (project_id,),
            )
            rows = await cursor.fetchall()

            features = []
            for row in rows:
                features.append({
                    "id": row[0],
                    "projectId": row[1],
                    "name": row[2],
                })

            return features


def _idea_row_to_dict(row, description) -> dict:
    """Convert a database row to an idea dict."""
    # Get column names from description
    cols = [desc[0] for desc in description]
    data = dict(zip(cols, row, strict=True))

    return {
        "id": data["id"],
        "title": data["title"],
        "summary": data["summary"],
        "type": data["type"],
        "status": data["status"],
        "score": data["score"],
        "rubric": data["rubric"] or {},
        "projectId": data["project_id"],
        "featureId": data["feature_id"],
        "owner": data["owner"],
        "originator": data["originator"],
        "contributors": data["contributors"] or [],
        "challengers": data["challengers"] or [],
        "evidence": data["evidence"] or [],
        "challenges": data["challenges"] or [],
        "gaps": data["gaps"] or [],
        "assumptions": data["assumptions"] or [],
        "contradictions": data["contradictions"] or [],
        "edited": data["edited"],
        "createdAt": _iso_datetime(data["created_at"]),
        "updatedAt": _iso_datetime(data["updated_at"]),
    }
