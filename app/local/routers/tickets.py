"""POST /v1/tickets: a one-shot ticket for the live socket or the event stream (see app/local/tickets.py).

Request   {"path": "/v1/events"} or {"path": "/v1/meetings/<id>/live"}, authenticated as any other request
          (engine token and, with tenant isolation on, the identity token, both as headers)
Answer    201 {"ticket", "path", "expiresInSeconds"}; the client opens `<path>?ticket=<ticket>` once
"""

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from app.local import tickets
from app.local.errors import InvalidRequestError

router = APIRouter(prefix="/v1", tags=["tickets"])


class TicketRequest(BaseModel):
    path: str = Field(min_length=1, max_length=200)


@router.post("/tickets", status_code=201)
async def issue_ticket(body: TicketRequest, request: Request, response: Response):
    if not tickets.ticketable(body.path):
        raise InvalidRequestError("path: tickets are issued only for /v1/events and /v1/meetings/{id}/live")
    book = request.app.state.tickets
    value = book.issue(body.path, tenant=getattr(request.state, "tenant", None),
                       identity=getattr(request.state, "identity", None))
    response.headers["Cache-Control"] = "no-store"
    return {"ticket": value, "path": body.path, "expiresInSeconds": book.ttl_s}
