"""Ideas endpoints. Tenant and workspace come from request headers."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from ..ids import next_idea_id
from ..store import store
from ..vocab import now_iso, status_for_score

router = APIRouter(prefix="/v1/ideas", tags=["ideas"])


def _ctx(request: Request) -> tuple[str, str]:
    return request.state.tenant_id, request.state.workspace_id


@router.get("/next-id")
def get_next_id():
    return {"id": next_idea_id()}


@router.get("")
def list_ideas(
    request: Request,
    status: Optional[str] = None,
    type: Optional[str] = None,
    projectId: Optional[str] = None,
    scoreMin: int = 0,
    scoreMax: int = 100,
    ownerId: Optional[str] = None,
    hasLineage: Optional[bool] = None,
    search: Optional[str] = None,
    sortBy: str = "score",
    sortDir: str = "desc",
    page: int = 1,
    pageSize: int = 25,
):
    tenant, workspace = _ctx(request)
    items = store.list_ideas(tenant, workspace)

    if status:
        wanted = set(status.split(","))
        items = [i for i in items if status_for_score(i["statusScore"]) in wanted]
    if type:
        wanted = set(type.split(","))
        items = [i for i in items if i.get("type") in wanted]
    if projectId:
        items = [i for i in items if i.get("projectId") == projectId]
    items = [i for i in items if scoreMin <= i["statusScore"] <= scoreMax]
    if ownerId:
        items = [i for i in items if i.get("originator") == ownerId or ownerId in i.get("contributors", [])]
    if hasLineage is not None:
        def has_lineage(i):
            ln = i.get("lineage", {})
            return bool(ln.get("evolvedFrom") or ln.get("mergedWith") or ln.get("linkedToFeature"))
        items = [i for i in items if has_lineage(i) == hasLineage]
    if search:
        q = search.lower()
        def matches(i):
            blob = " ".join([i.get("title", ""), i.get("description", "")]
                            + [e.get("text", "") for e in i.get("evidence", [])])
            return q in blob.lower()
        items = [i for i in items if matches(i)]

    keymap = {
        "score": lambda i: i["statusScore"],
        "date": lambda i: i.get("createdAt", ""),
        "title": lambda i: i.get("title", "").lower(),
        "id": lambda i: i["id"],
    }
    items.sort(key=keymap.get(sortBy, keymap["score"]), reverse=(sortDir == "desc"))

    total = len(items)
    start = (page - 1) * pageSize
    page_items = items[start:start + pageSize]
    total_pages = (total + pageSize - 1) // pageSize if pageSize else 1
    return {"items": page_items, "total": total, "page": page, "pageSize": pageSize, "totalPages": total_pages}


@router.get("/{idea_id}")
def get_idea(request: Request, idea_id: str):
    tenant, workspace = _ctx(request)
    idea = store.get_idea(tenant, workspace, idea_id)
    if not idea:
        raise HTTPException(404, "idea not found")
    return idea


@router.post("")
async def create_idea(request: Request):
    tenant, workspace = _ctx(request)
    body = await request.json()
    if not body.get("title") or not body.get("type"):
        raise HTTPException(422, "title and type are required")
    idea = {
        "id": next_idea_id(),
        "tenantId": tenant,
        "workspaceId": workspace,
        "projectId": body.get("projectId"),
        "statusScore": body.get("statusScore", 0),
        "rubricScores": body.get("rubricScores", {}),
        "lineage": body.get("lineage", {"evolvedFrom": [], "mergedWith": [],
                                        "originatedFromSession": None, "linkedToFeature": None}),
        "createdAt": now_iso(),
        **body,
    }
    idea["id"] = idea["id"]  # keep server-allocated id
    idea["tenantId"], idea["workspaceId"] = tenant, workspace
    store.put_idea(idea)
    return idea


@router.patch("/{idea_id}")
async def patch_idea(request: Request, idea_id: str):
    tenant, workspace = _ctx(request)
    idea = store.get_idea(tenant, workspace, idea_id)
    if not idea:
        raise HTTPException(404, "idea not found")
    body = await request.json()
    body.pop("id", None)
    body.pop("tenantId", None)
    body.pop("workspaceId", None)
    idea.update(body)
    store.put_idea(idea)
    return idea
