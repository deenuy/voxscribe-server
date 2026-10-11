"""Ideas endpoints."""

import json
import logging
import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.local import events
from app.local.errors import ConflictError, EngineError, InvalidRequestError, NotFoundError
from app.local.job_runner import get_runner
from app.local.models import (
    CreateFeatureRequest,
    CreateIdeaRequest,
    CreateProjectRequest,
    ExtractIdeasRequest,
    HandoffIdeaRequest,
    UpdateIdeaRequest,
    UpdateProjectRequest,
)
from app.local.store_meetings import MeetingStore
from app.local.store_ideas import FeatureStore, IdeaStore, ProjectStore, _iso_datetime

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ideas"])


# ============================================================================
# Vocabulary
# ============================================================================


@router.get("/ideas/vocabulary")
async def get_vocabulary():
    """Get ideas vocabulary: types, statuses, rubric."""
    return {
        "types": [
            {"key": "new_product", "label": "New Product"},
            {"key": "new_feature", "label": "New Feature"},
            {"key": "feature_enhancement", "label": "Feature Enhancement"},
            {"key": "ai_capability", "label": "AI Capability"},
            {"key": "workflow_automation", "label": "Workflow Automation"},
            {"key": "data_product", "label": "Data Product"},
            {"key": "platform_capability", "label": "Platform Capability"},
            {"key": "governance_control", "label": "Governance Control"},
            {"key": "experiment", "label": "Experiment"},
            {"key": "technical_enabler", "label": "Technical Enabler"},
            {"key": "risk_compliance", "label": "Risk Compliance"},
        ],
        "statuses": [
            {"key": "raw", "label": "Raw", "min": 0, "max": 39},
            {"key": "needs_discovery", "label": "Needs Discovery", "min": 40, "max": 59},
            {"key": "promising", "label": "Promising", "min": 60, "max": 74},
            {"key": "prd_candidate", "label": "PRD Candidate", "min": 75, "max": 84},
            {"key": "delivery_ready", "label": "Delivery Ready", "min": 85, "max": 100},
        ],
        "rubric": [
            {"key": "problem", "label": "Problem", "weight": 15},
            {"key": "persona", "label": "Persona", "weight": 10},
            {"key": "value", "label": "Value", "weight": 10},
            {"key": "strategy", "label": "Strategy", "weight": 10},
            {"key": "differentiation", "label": "Differentiation", "weight": 5},
            {"key": "feasibility", "label": "Feasibility", "weight": 10},
            {"key": "data_ai", "label": "Data & AI", "weight": 10},
            {"key": "scope", "label": "Scope", "weight": 10},
            {"key": "risk", "label": "Risk", "weight": 10},
            {"key": "execution", "label": "Execution", "weight": 10},
        ],
    }


# ============================================================================
# Ideas
# ============================================================================


@router.get("/ideas")
async def list_ideas(
    request: Request,
    query: str = None,
    status: str = None,
    type: str = None,
    projectId: str = None,
    minScore: int = None,
    maxScore: int = None,
    sort: str = "updated",
    order: str = "desc",
    limit: int = 25,
    cursor: str = None,
):
    """List ideas with filtering and pagination."""
    limit = max(1, min(int(limit), 200))  # a page is bounded: nobody can ask for the whole table
    store = IdeaStore(request.app.state.store.pool)
    ideas, next_cursor = await store.list_ideas(
        query=query,
        status=status,
        type_=type,
        project_id=projectId,
        min_score=minScore,
        max_score=maxScore,
        sort=sort,
        order=order,
        limit=limit,
        cursor=cursor,
    )

    # Get counts by status
    counts_by_status = {}

    return {
        "items": ideas,
        "nextCursor": next_cursor,
        "total": len(ideas),
        "countsByStatus": counts_by_status,
    }


async def _require_project_or_404(request: Request, project_id: str) -> None:
    async with request.app.state.store.pool.connection() as conn:
        if not await (await conn.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,))).fetchone():
            raise NotFoundError(f"project {project_id} not found")


async def _require_project(request: Request, project_id: str) -> None:
    """A reference to a project must name one that exists (400 otherwise, not a dangling pointer)."""
    async with request.app.state.store.pool.connection() as conn:
        if not await (await conn.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,))).fetchone():
            raise InvalidRequestError("projectId: no such project")


@router.post("/ideas", status_code=201)
async def create_idea(request: Request, body: CreateIdeaRequest):
    """Create a manual idea."""
    store = IdeaStore(request.app.state.store.pool)
    idea_id = f"idea_{str(uuid.uuid4())[:8]}"
    if body.projectId:
        await _require_project(request, body.projectId)

    idea = await store.create_idea(
        idea_id=idea_id,
        title=body.title or "Untitled",
        summary=body.summary or "",
        type_=body.type or "new_feature",
        project_id=body.projectId,
    )

    # Log activity
    activity_id = str(uuid.uuid4())
    async with request.app.state.store.pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO idea_activity (id, idea_id, kind, detail)
            VALUES (%s, %s, %s, %s)
            """,
            (activity_id, idea_id, "created", json.dumps({})),
        )

    events.publish("idea.update", {"idea": idea})
    return idea


@router.get("/ideas/graph")
async def get_ideas_graph(request: Request, projectId: str = None):
    """Get all ideas and features for graph view."""
    idea_store = IdeaStore(request.app.state.store.pool)
    feature_store = FeatureStore(request.app.state.store.pool)

    nodes = []
    edges = []

    # Add project node if specified
    if projectId:
        project_store = ProjectStore(request.app.state.store.pool)
        try:
            project = await project_store.get_project(projectId)
            nodes.append({
                "id": projectId,
                "kind": "project",
                "label": project["name"],
            })
        except NotFoundError:
            pass

    # Get all ideas for the project (or all if no projectId)
    ideas, _ = await idea_store.list_ideas(
        project_id=projectId,
        limit=1000,  # Reasonable limit for graph
    )

    for idea in ideas:
        nodes.append({
            "id": idea["id"],
            "kind": "idea",
            "label": idea["title"],
        })

        if idea.get("featureId"):
            # Add edge to feature
            edges.append({
                "from": idea["id"],
                "to": idea["featureId"],
                "kind": "belongs_to",
            })

    # Get features
    if projectId:
        features = await feature_store.list_features(projectId)
        for feature in features:
            nodes.append({
                "id": feature["id"],
                "kind": "feature",
                "label": feature["name"],
            })

    # Add lineage edges
    async with request.app.state.store.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT from_id, to_id, kind FROM idea_edges
            """
        )
        rows = await cursor.fetchall()
        for from_id, to_id, kind in rows:
            # Only include edges between ideas in the graph
            if any(n["id"] == from_id for n in nodes) and any(n["id"] == to_id for n in nodes):
                edges.append({
                    "from": from_id,
                    "to": to_id,
                    "kind": kind,
                })

    return {
        "nodes": nodes,
        "edges": edges,
    }


@router.get("/ideas/{idea_id}")
async def get_idea(request: Request, idea_id: str):
    """Get an idea by ID."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        idea = await store.get_idea(idea_id)
        return idea
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": f"idea {idea_id} not found"})


@router.patch("/ideas/{idea_id}")
async def update_idea(request: Request, idea_id: str, body: UpdateIdeaRequest):
    """Update an idea."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        # Convert model to dict, excluding None values
        updates = body.model_dump(exclude_none=True)
        idea = await store.update_idea(idea_id, **updates)
        events.publish("idea.update", {"idea": idea})
        return idea
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


@router.delete("/ideas/{idea_id}", status_code=204)
async def delete_idea(request: Request, idea_id: str):
    """Delete an idea."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        await store.delete_idea(idea_id)
        events.publish("idea.delete", {"id": idea_id})
    except NotFoundError:
        pass  # already gone: a repeated DELETE still succeeds
    return Response(status_code=204)


@router.get("/ideas/{idea_id}/lineage")
async def get_idea_lineage(request: Request, idea_id: str):
    """Get idea lineage graph."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        idea = await store.get_idea(idea_id)
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})

    nodes = [
        {"id": idea_id, "kind": "idea", "label": idea["title"]},
    ]
    edges = []

    # Get lineage relationships
    async with request.app.state.store.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT from_id, to_id, kind FROM idea_edges
            WHERE from_id = %s OR to_id = %s
            """,
            (idea_id, idea_id),
        )
        rows = await cursor.fetchall()
        for from_id, to_id, kind in rows:
            other_id = to_id if from_id == idea_id else from_id
            if other_id not in [n["id"] for n in nodes]:
                try:
                    other_idea = await store.get_idea(other_id)
                    nodes.append({
                        "id": other_id,
                        "kind": "idea",
                        "label": other_idea["title"],
                    })
                except NotFoundError:
                    pass

            if from_id == idea_id:
                edges.append({"from": from_id, "to": to_id, "kind": kind})
            else:
                edges.append({"from": to_id, "to": from_id, "kind": kind})

    # Add meeting node if idea originated from one
    if idea.get("originatedFromMeetingId"):
        meeting_id = idea["originatedFromMeetingId"]
        nodes.append({"id": meeting_id, "kind": "meeting", "label": "Transcript"})
        edges.append({
            "from": meeting_id,
            "to": idea_id,
            "kind": "originated_from",
        })

    return {
        "nodes": nodes,
        "edges": edges,
    }


@router.post("/ideas/{idea_id}/rescore", status_code=202)
async def rescore_idea(request: Request, idea_id: str):
    """Rescore an idea (creates a job)."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        await store.get_idea(idea_id)
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})

    runner = get_runner()

    async def do_rescore():
        """Rescore the idea using the main model."""
        # TODO: call the main model to rescore the idea based on evidence
        # For now, just update the idea with a log entry
        async with request.app.state.store.pool.connection() as conn:
            activity_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO idea_activity (id, idea_id, kind, detail)
                VALUES (%s, %s, %s, %s)
                """,
                (activity_id, idea_id, "scored", json.dumps({"method": "rescore"})),
            )
            await conn.commit()

        events.publish("idea.update", {"idea": await store.get_idea(idea_id)})

    job_id = await runner.submit(
        kind="rescore_idea",
        work_fn=do_rescore,
        detail={"ideaId": idea_id},
    )
    return {"jobId": job_id}


# ============================================================================
# Extraction from a meeting
# ============================================================================


@router.post("/meetings/{meeting_id}/extract-ideas", status_code=202)
async def extract_ideas(request: Request, meeting_id: str, body: ExtractIdeasRequest):
    """Extract ideas from a meeting transcript.

    Not wired yet: the model call and the mapping into ideas, evidence and lineage are unfinished. Answering 202
    and queueing a job that can only fail would tell the person it is working, so say so plainly instead. The
    meeting must exist and have a transcript first, so the answer is still the right one for a bad request."""
    store = MeetingStore(request.app.state.store.pool)
    meeting = await store.get_meeting(meeting_id)  # 404 if there is no such meeting
    if not meeting.hasTranscript:
        raise ConflictError("this meeting has no transcript to extract ideas from")
    raise EngineError("not_implemented", "idea extraction is not available in this engine build yet", 501)


# ============================================================================
# PRD and handoff
# ============================================================================


@router.post("/ideas/{idea_id}/prd", status_code=202)
async def generate_prd(request: Request, idea_id: str):
    """Generate a PRD for an idea."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        idea = await store.get_idea(idea_id)
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})

    # Check if idea is prd_candidate or delivery_ready
    if idea["status"] not in ["prd_candidate", "delivery_ready"]:
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "detail": f"Idea status is {idea['status']}, must be prd_candidate or delivery_ready",
                "score": idea["score"],
                "threshold": 75,
            },
        )

    runner = get_runner()

    async def do_prd_generation():
        """Generate PRD for the idea."""
        # TODO: call the main model to generate PRD
        # For now, create a placeholder
        markdown = f"# {idea['title']}\n\n## Problem\n{idea['summary']}\n"

        async with request.app.state.store.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO idea_prds (idea_id, markdown, status, generated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (idea_id) DO UPDATE SET
                    markdown = EXCLUDED.markdown,
                    status = EXCLUDED.status,
                    generated_at = NOW()
                """,
                (idea_id, markdown, "ready"),
            )
            await conn.commit()

        events.publish("idea.update", {"idea": await store.get_idea(idea_id)})

    job_id = await runner.submit(
        kind="generate_prd",
        work_fn=do_prd_generation,
        detail={"ideaId": idea_id},
    )
    return {"jobId": job_id}


@router.get("/ideas/{idea_id}/prd")
async def get_prd(request: Request, idea_id: str):
    """Get PRD status and markdown."""
    async with request.app.state.store.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT status, markdown, generated_at FROM idea_prds WHERE idea_id = %s",
            (idea_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {
                "status": "none",
                "markdown": None,
                "generatedAt": None,
            }
        return {
            "status": row[0],
            "markdown": row[1],
            "generatedAt": _iso_datetime(row[2]),
        }


@router.get("/ideas/{idea_id}/export")
async def export_idea(request: Request, idea_id: str, format: str = "markdown"):
    """Export an idea."""
    store = IdeaStore(request.app.state.store.pool)
    try:
        idea = await store.get_idea(idea_id)
        if format == "json":
            return idea

        # Build markdown export
        lines = [f"# {idea['title']}"]
        if idea.get("summary"):
            lines.append(f"\n{idea['summary']}")

        if idea.get("evidence"):
            lines.append("\n## Evidence")
            for ev in idea["evidence"]:
                quote = ev.get("quote", "")
                speaker = ev.get("speaker")
                if speaker:
                    lines.append(f"\n> {quote}\n> -- {speaker}")
                else:
                    lines.append(f"\n> {quote}")

        if idea.get("challenges"):
            lines.append("\n## Challenges")
            for challenge in idea["challenges"]:
                lines.append(f"\n- {challenge.get('question', challenge)}")

        if idea.get("gaps"):
            lines.append("\n## Gaps")
            for gap in idea["gaps"]:
                lines.append(f"\n- {gap}")

        if idea.get("assumptions"):
            lines.append("\n## Assumptions")
            for assumption in idea["assumptions"]:
                lines.append(f"\n- {assumption}")

        # Get PRD if available
        async with request.app.state.store.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT markdown FROM idea_prds WHERE idea_id = %s AND status = %s",
                (idea_id, "ready"),
            )
            prd = await cursor.fetchone()
            if prd:
                lines.append(f"\n\n## Product Requirements Document\n\n{prd[0]}")

        return {"kind": "file", "content": "\n".join(lines)}
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


@router.post("/ideas/{idea_id}/handoff")
async def handoff_idea(request: Request, idea_id: str, body: HandoffIdeaRequest):
    """Handoff an idea to external tools."""
    target = body.target
    if target in ["jira", "azure_devops"]:
        return JSONResponse(
            status_code=501,
            content={"error": "connector_not_configured", "detail": f"{target} connector not configured"},
        )

    store = IdeaStore(request.app.state.store.pool)
    try:
        idea = await store.get_idea(idea_id)
        content = f"# {idea['title']}\n\n{idea['summary']}"
        return {"kind": "file", "content": content}
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


@router.get("/ideas/{idea_id}/activity")
async def get_activity(request: Request, idea_id: str):
    """Get activity log for an idea."""
    async with request.app.state.store.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, kind, detail, created_at FROM idea_activity
            WHERE idea_id = %s
            ORDER BY created_at DESC
            """,
            (idea_id,),
        )
        rows = await cursor.fetchall()

        items = []
        for row in rows:
            items.append({
                "at": _iso_datetime(row[3]),
                "kind": row[1],
                "detail": row[2] or {},
            })

        return {"items": items}


# ============================================================================
# Projects
# ============================================================================


@router.get("/projects")
async def list_projects(request: Request, limit: int = 25, cursor: str = None):
    """List projects."""
    limit = max(1, min(int(limit), 200))
    store = ProjectStore(request.app.state.store.pool)
    projects, next_cursor = await store.list_projects(limit=limit, cursor=cursor)
    return {
        "items": projects,
        "nextCursor": next_cursor,
    }


@router.post("/projects", status_code=201)
async def create_project(request: Request, body: CreateProjectRequest):
    """Create a project."""
    store = ProjectStore(request.app.state.store.pool)
    project_id = f"proj_{str(uuid.uuid4())[:8]}"

    project = await store.create_project(
        project_id=project_id,
        name=body.name,
        description=body.description,
        parent_id=body.parentId,
    )

    events.publish("project.update", project)
    return project


@router.get("/projects/{project_id}")
async def get_project(request: Request, project_id: str):
    """One project, with its idea count."""
    store = ProjectStore(request.app.state.store.pool)
    try:
        return await store.get_project(project_id)
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


@router.patch("/projects/{project_id}")
async def update_project(request: Request, project_id: str, body: UpdateProjectRequest):
    """Update a project."""
    store = ProjectStore(request.app.state.store.pool)
    try:
        updates = body.model_dump(exclude_none=True)
        project = await store.update_project(project_id, **updates)
        events.publish("project.update", project)
        return project
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(request: Request, project_id: str):
    """Delete a project."""
    store = ProjectStore(request.app.state.store.pool)
    try:
        await store.delete_project(project_id)
        return Response(status_code=204)
    except ConflictError as e:
        return JSONResponse(status_code=409, content={"error": "conflict", "detail": str(e)})
    except NotFoundError:
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": "not found"})


# ============================================================================
# Features
# ============================================================================


@router.get("/projects/{project_id}/features")
async def list_features(request: Request, project_id: str):
    """List features for a project."""
    await _require_project_or_404(request, project_id)
    store = FeatureStore(request.app.state.store.pool)
    features = await store.list_features(project_id)
    return {"items": features}


@router.post("/projects/{project_id}/features", status_code=201)
async def create_feature(request: Request, project_id: str, body: CreateFeatureRequest):
    """Create a feature."""
    await _require_project_or_404(request, project_id)
    store = FeatureStore(request.app.state.store.pool)
    feature_id = f"feat_{str(uuid.uuid4())[:8]}"

    feature = await store.create_feature(
        feature_id=feature_id,
        project_id=project_id,
        name=body.name,
    )

    return feature
