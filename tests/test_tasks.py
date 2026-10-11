"""Tasks: a to-do list per scribe, by hand or drafted from the transcript."""

import asyncio

import pytest

from app.local import task_service
from app.local.errors import EngineError, ProviderUnavailableError
from tests.test_transcript_edit import meeting_with_segments

TRANSCRIPT = ("We should send the pricing page to legal by Friday.", "Alice will follow up with the vendor about the contract.", "The weather was nice.")


async def done_scribe(client, texts=TRANSCRIPT, **create):
    mid, _ = await meeting_with_segments(client, texts=texts, **create)
    return mid


@pytest.mark.asyncio
async def test_add_list_edit_complete_and_delete_by_hand(client):
    mid = await done_scribe(client)
    made = await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "  Call the vendor  ", "detail": "About the renewal"})
    assert made.status_code == 201
    task = made.json()
    assert task["title"] == "Call the vendor" and task["source"] == "manual" and task["status"] == "todo" and task["completedAt"] is None
    assert task["createdAt"].endswith("Z") and task["meetingId"] == mid

    assert (await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"][0]["id"] == task["id"]

    done = (await client.patch(f"/v1/tasks/{task['id']}", json={"status": "done"})).json()
    assert done["status"] == "done" and done["completedAt"].endswith("Z")
    again = (await client.patch(f"/v1/tasks/{task['id']}", json={"status": "done"})).json()
    assert again["completedAt"] == done["completedAt"]  # completing twice keeps the first time
    reopened = (await client.patch(f"/v1/tasks/{task['id']}", json={"status": "todo"})).json()
    assert reopened["completedAt"] is None

    renamed = (await client.patch(f"/v1/tasks/{task['id']}", json={"title": "Call the vendor today", "detail": ""})).json()
    assert renamed["title"] == "Call the vendor today" and renamed["detail"] is None

    assert (await client.delete(f"/v1/tasks/{task['id']}")).status_code == 204
    assert (await client.delete(f"/v1/tasks/{task['id']}")).status_code == 204  # already gone is fine
    assert (await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"] == []


@pytest.mark.asyncio
async def test_bad_requests_are_clean_4xx(client):
    mid = await done_scribe(client)
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "   "})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "x" * 301})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", content=b"nope")).status_code == 400
    assert (await client.post("/v1/meetings/missing/tasks", json={"title": "a"})).status_code == 404
    assert (await client.get("/v1/meetings/missing/tasks")).status_code == 404
    t = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "a"})).json()
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={})).status_code == 400
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"status": "later"})).status_code == 400
    assert (await client.patch("/v1/tasks/missing", json={"status": "done"})).status_code == 404
    assert (await client.get("/v1/tasks?status=weird")).status_code == 400
    assert (await client.get("/v1/tasks?limit=0")).status_code == 400


@pytest.mark.asyncio
async def test_the_same_title_twice_is_a_409_however_it_is_spelled(client):
    mid = await done_scribe(client)
    await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Send the deck"})
    dup = await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "  send   THE deck "})
    assert dup.status_code == 409 and dup.json()["error"] == "conflict"
    other = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Another"})).json()
    clash = await client.patch(f"/v1/tasks/{other['id']}", json={"title": "Send the deck"})
    assert clash.status_code == 409


@pytest.mark.asyncio
async def test_generate_drafts_from_the_transcript_and_is_safe_to_repeat(client):
    mid = await done_scribe(client)
    first = (await client.post(f"/v1/meetings/{mid}/tasks/generate")).json()
    titles = sorted(t["title"] for t in first["created"])
    assert titles == ["Alice will follow up with the vendor about the contract", "We should send the pricing page to legal by Friday"]
    assert all(t["source"] == "ai" for t in first["created"]) and first["skipped"] == 0

    again = (await client.post(f"/v1/meetings/{mid}/tasks/generate")).json()
    assert again["created"] == [] and again["skipped"] == 2  # nothing added twice
    assert len((await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"]) == 2


@pytest.mark.asyncio
async def test_generate_never_touches_tasks_the_person_made_or_finished(client):
    mid = await done_scribe(client)
    mine = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "We should send the pricing page to legal by Friday"})).json()
    await client.patch(f"/v1/tasks/{mine['id']}", json={"status": "done"})
    out = (await client.post(f"/v1/meetings/{mid}/tasks/generate")).json()
    assert [t["title"] for t in out["created"]] == ["Alice will follow up with the vendor about the contract"] and out["skipped"] == 1
    kept = next(t for t in (await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"] if t["id"] == mine["id"])
    assert kept["status"] == "done" and kept["source"] == "manual"


@pytest.mark.asyncio
async def test_two_generates_at_once_add_each_task_once(client):
    mid = await done_scribe(client)
    a, b = await asyncio.gather(client.post(f"/v1/meetings/{mid}/tasks/generate"), client.post(f"/v1/meetings/{mid}/tasks/generate"))
    assert a.status_code == 200 and b.status_code == 200
    assert len((await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"]) == 2


@pytest.mark.asyncio
async def test_generate_needs_a_finished_scribe_with_a_transcript(client):
    empty = (await client.post("/v1/meetings", json={})).json()["id"]
    res = await client.post(f"/v1/meetings/{empty}/tasks/generate")
    assert res.status_code == 409
    assert (await client.post("/v1/meetings/missing/tasks/generate")).status_code == 404


@pytest.mark.asyncio
async def test_a_task_carries_its_scribes_project_and_the_list_filters_by_it(client):
    project = (await client.post("/v1/projects", json={"name": "Launch"})).json()
    in_project = await done_scribe(client, projectId=project["id"])
    plain = await done_scribe(client)
    t1 = (await client.post(f"/v1/meetings/{in_project}/tasks", json={"title": "Plan launch"})).json()
    await client.post(f"/v1/meetings/{plain}/tasks", json={"title": "Unrelated"})
    assert t1["projectId"] == project["id"]
    listed = (await client.get(f"/v1/tasks?projectId={project['id']}")).json()
    assert [t["title"] for t in listed["items"]] == ["Plan launch"] and listed["total"] == 1
    assert (await client.get("/v1/tasks?status=todo")).json()["total"] == 2
    assert (await client.get(f"/v1/tasks?meetingId={plain}")).json()["total"] == 1


@pytest.mark.asyncio
async def test_deleting_a_scribe_deletes_its_tasks(client):
    mid = await done_scribe(client)
    await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "x"})
    await client.delete(f"/v1/meetings/{mid}")
    assert (await client.get(f"/v1/tasks?meetingId={mid}")).json()["total"] == 0


# ---- the drafting rules, with a model that answers badly

class Model:
    def __init__(self, reply):
        self.reply = reply

    def stream(self, system, prompt, max_tokens=2048):
        self.prompt = prompt
        yield self.reply


@pytest.mark.asyncio
async def test_a_model_reply_is_cleaned_and_a_bad_one_is_an_error_not_an_empty_list():
    ok = Model('```json\n{"tasks":[{"title":"Email Sam."},{"title":"email sam"},{"title":""},{"nope":1},{"title":"Book room","detail":"Third floor"}]}\n```')
    drafts = await task_service.draft_tasks(ok, lines=["x"], project={"name": "Launch", "description": "Q4 plan"}, fake=False)
    assert drafts == [{"title": "Email Sam", "detail": None, "owner": None, "dueDate": None}, {"title": "Book room", "detail": "Third floor", "owner": None, "dueDate": None}]
    assert "Launch" in ok.prompt and "Q4 plan" in ok.prompt  # the project is context for the model

    with pytest.raises(EngineError) as bad:
        await task_service.draft_tasks(Model("I could not do that"), lines=["x"], project=None, fake=False)
    assert bad.value.status == 502 and bad.value.code == "ai_unreadable"
    assert await task_service.draft_tasks(Model('{"tasks": []}'), lines=["x"], project=None, fake=False) == []  # a real "nothing to do"

    with pytest.raises(ProviderUnavailableError):
        await task_service.draft_tasks(None, lines=["x"], project=None, fake=False)

    class Down:
        def stream(self, *a, **k):
            raise RuntimeError("connection refused")
    with pytest.raises(ProviderUnavailableError):
        await task_service.draft_tasks(Down(), lines=["x"], project=None, fake=False)


def test_the_list_is_capped_and_titles_are_bounded():
    many = [{"title": f"Task {i}"} for i in range(50)] + [{"title": "y" * 500}]
    out = task_service.clean_drafts(many)
    assert len(out) == task_service.MAX_TASKS and all(len(d["title"]) <= 300 for d in out)


@pytest.mark.asyncio
async def test_owner_and_due_date_from_the_model_are_kept_only_when_they_are_real():
    reply = '{"tasks":[{"title":"Send deck","owner":"Alice","dueDate":"2026-10-16"},{"title":"Call vendor","owner":"  ","dueDate":"Friday"},{"title":"Pay invoice","dueDate":"2026-02-30"}]}'
    model = Model(reply)
    drafts = await task_service.draft_tasks(model, lines=["x"], labelled=["Alice: x"], project=None, held_on="2026-10-11", fake=False)
    assert drafts[0]["owner"] == "Alice" and drafts[0]["dueDate"] == "2026-10-16"
    assert drafts[1]["owner"] is None and drafts[1]["dueDate"] is None  # "Friday" is not a date
    assert drafts[2]["dueDate"] is None  # no such day
    assert "2026-10-11" in model.prompt and "Alice: x" in model.prompt  # the date and who spoke reach the model


@pytest.mark.asyncio
async def test_a_task_has_an_owner_a_due_date_a_middle_status_and_a_project_of_its_own(client):
    plan = (await client.post("/v1/projects", json={"name": "Plan"})).json()
    other = (await client.post("/v1/projects", json={"name": "Other"})).json()
    mid = await done_scribe(client, projectId=plan["id"])
    made = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Draft the plan", "owner": " Dana ", "dueDate": "2026-10-20", "status": "in_progress"})).json()
    assert made["owner"] == "Dana" and made["dueDate"] == "2026-10-20" and made["status"] == "in_progress" and made["completedAt"] is None
    assert made["projectId"] == plan["id"]  # the scribe's project, until the task has its own

    moved = (await client.patch(f"/v1/tasks/{made['id']}", json={"projectId": other["id"]})).json()
    assert moved["projectId"] == other["id"]
    back = (await client.patch(f"/v1/tasks/{made['id']}", json={"projectId": None})).json()
    assert back["projectId"] == plan["id"]  # cleared: falls back to the scribe's

    cleared = (await client.patch(f"/v1/tasks/{made['id']}", json={"owner": "", "dueDate": None})).json()
    assert cleared["owner"] is None and cleared["dueDate"] is None and cleared["title"] == "Draft the plan"  # only what was sent changed

    finished = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Already done", "status": "done"})).json()
    assert finished["completedAt"].endswith("Z")
    assert (await client.get("/v1/tasks?status=in_progress")).json()["total"] == 1
    assert (await client.get(f"/v1/tasks?projectId={plan['id']}")).json()["total"] == 2


@pytest.mark.asyncio
async def test_bad_task_details_are_clean_4xx(client):
    mid = await done_scribe(client)
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "a", "dueDate": "next friday"})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "a", "projectId": "nope"})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "a", "owner": "x" * 121})).status_code == 400
    assert (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "a", "status": "blocked"})).status_code == 400
    t = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "ok"})).json()
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"status": None})).status_code == 400
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"title": None})).status_code == 400
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"projectId": "nope"})).status_code == 400
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"dueDate": "2026-13-45"})).status_code == 400


@pytest.mark.asyncio
async def test_a_scribe_can_be_moved_into_a_project_and_out_again(client):
    project = (await client.post("/v1/projects", json={"name": "Plan"})).json()
    mid = await done_scribe(client)
    t = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Follow up"})).json()
    assert t["projectId"] is None

    moved = (await client.patch(f"/v1/meetings/{mid}", json={"projectId": project["id"]})).json()
    assert moved["projectId"] == project["id"]
    assert (await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"][0]["projectId"] == project["id"]  # its tasks follow

    untouched = (await client.patch(f"/v1/meetings/{mid}", json={"title": "Renamed"})).json()
    assert untouched["projectId"] == project["id"]  # leaving projectId out keeps it

    out = (await client.patch(f"/v1/meetings/{mid}", json={"projectId": None})).json()
    assert "projectId" not in out or out["projectId"] is None
    assert (await client.patch(f"/v1/meetings/{mid}", json={"projectId": "nope"})).status_code == 400
