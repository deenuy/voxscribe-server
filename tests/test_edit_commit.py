"""One atomic commit of an edit session: all or nothing, safe to retry, and honest about what conflicted."""

import asyncio

import pytest

from tests.test_transcript_edit import meeting_with_segments


async def scribe(client, **create):
    mid, ids = await meeting_with_segments(client, texts=("First line here", "Second line here", "Third line here"), **create)
    return mid, ids


async def state(client, mid):
    m = (await client.get(f"/v1/meetings/{mid}")).json()
    t = (await client.get(f"/v1/meetings/{mid}/transcript")).json()
    s = (await client.get(f"/v1/meetings/{mid}/summary")).json()
    tasks = (await client.get(f"/v1/meetings/{mid}/tasks")).json()["items"]
    return m, t["segments"], s, tasks


@pytest.mark.asyncio
async def test_one_commit_changes_title_project_lines_summary_and_tasks_together(client):
    project = (await client.post("/v1/projects", json={"name": "Plan"})).json()
    mid, ids = await scribe(client)
    m, segs, _, _ = await state(client, mid)
    t1 = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Old task"})).json()
    t2 = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Drop me"})).json()

    res = await client.post(f"/v1/meetings/{mid}/edits", json={
        "meeting": {"baseRevision": m["revision"], "title": "Renamed", "projectId": project["id"], "context": "Why we met"},
        "segments": [{"id": ids[0], "baseRevision": 0, "text": "First line, corrected", "speaker": "Alice"}],
        "summary": {"baseRevision": 0, "overview": "My own overview", "details": [{"title": "Point", "body": "Text"}]},
        "tasks": {
            "create": [{"tempId": "n1", "title": "New task", "owner": "Dana", "dueDate": "2026-11-01", "status": "in_progress"}],
            "update": [{"id": t1["id"], "baseRevision": t1["revision"], "status": "done", "owner": "Sam"}],
            "delete": [{"id": t2["id"], "baseRevision": t2["revision"]}],
        },
    })
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["meeting"]["title"] == "Renamed" and out["meeting"]["projectId"] == project["id"] and out["meeting"]["revision"] == m["revision"] + 1
    assert out["summary"]["edited"] is True and out["summary"]["revision"] == 1 and out["meeting"]["summaryStale"] is False
    assert out["changed"] == {"meeting": True, "segments": 1, "summary": True, "tasks": 3}
    new_id = out["createdTasks"]["n1"]
    by_id = {t["id"]: t for t in out["tasks"]}
    assert t2["id"] not in by_id and by_id[t1["id"]]["status"] == "done" and by_id[t1["id"]]["owner"] == "Sam"
    assert by_id[new_id]["owner"] == "Dana" and by_id[new_id]["dueDate"] == "2026-11-01" and by_id[new_id]["status"] == "in_progress"

    m2, segs2, s2, tasks2 = await state(client, mid)
    assert m2["title"] == "Renamed" and m2["context"] == "Why we met" and segs2[0]["text"] == "First line, corrected" and segs2[0]["speaker"] == "Alice"
    assert s2["overview"] == "My own overview" and s2["edited"] is True
    assert {t["title"] for t in tasks2} == {"Old task", "New task"}


@pytest.mark.asyncio
async def test_if_any_piece_is_stale_nothing_at_all_is_written_and_the_pieces_are_named(client):
    mid, ids = await scribe(client)
    m, *_ = await state(client, mid)
    await client.patch(f"/v1/meetings/{mid}/transcript/{ids[1]}", json={"text": "Someone else edited this", "baseRevision": 0})  # now revision 1

    res = await client.post(f"/v1/meetings/{mid}/edits", json={
        "meeting": {"baseRevision": m["revision"], "title": "Should not be saved"},
        "segments": [{"id": ids[0], "baseRevision": 0, "text": "Also should not be saved"}, {"id": ids[1], "baseRevision": 0, "text": "stale"}],
    })
    assert res.status_code == 409
    body = res.json()
    assert body["error"] == "revision_conflict" and body["items"] == [{"kind": "segment", "id": ids[1], "currentRevision": 1}]
    m2, segs, *_ = await state(client, mid)
    assert m2["title"] != "Should not be saved" and segs[0]["text"] == "First line here"  # the good pieces were not written either


@pytest.mark.asyncio
async def test_a_bad_value_anywhere_writes_nothing(client):
    mid, ids = await scribe(client)
    m, *_ = await state(client, mid)
    for body in (
        {"meeting": {"baseRevision": m["revision"], "title": "ok"}, "segments": [{"id": ids[0], "baseRevision": 0, "text": "   "}]},
        {"meeting": {"baseRevision": m["revision"], "title": "ok"}, "tasks": {"create": [{"tempId": "x", "title": "t", "dueDate": "nope"}]}},
        {"meeting": {"baseRevision": m["revision"], "title": "ok", "projectId": "no-such-project"}},
        {"meeting": {"baseRevision": m["revision"], "title": "   "}},
        {"summary": {"baseRevision": 0, "overview": "x", "details": [{"title": "", "body": "b"}]}},
    ):
        res = await client.post(f"/v1/meetings/{mid}/edits", json=body)
        assert res.status_code == 400, (body, res.text)
    m2, segs, *_ = await state(client, mid)
    assert m2["title"] == m["title"] and m2["revision"] == m["revision"] and segs[0]["text"] == "First line here"
    assert (await client.post(f"/v1/meetings/{mid}/edits", json={})).status_code == 400
    assert (await client.post("/v1/meetings/missing/edits", json={"meeting": {"baseRevision": 0, "title": "x"}})).status_code == 404


@pytest.mark.asyncio
async def test_a_duplicate_task_title_conflicts_and_rolls_the_whole_commit_back(client):
    mid, _ = await scribe(client)
    m, *_ = await state(client, mid)
    await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "Send the deck"})
    res = await client.post(f"/v1/meetings/{mid}/edits", json={
        "meeting": {"baseRevision": m["revision"], "title": "Rolled back"},
        "tasks": {"create": [{"tempId": "n1", "title": "  send THE deck "}]},
    })
    assert res.status_code == 409 and res.json()["items"] == [{"kind": "task", "tempId": "n1", "reason": "duplicate_title"}]
    assert (await state(client, mid))[0]["title"] == m["title"]


@pytest.mark.asyncio
async def test_a_retried_commit_is_applied_once(client):
    mid, ids = await scribe(client)
    m, *_ = await state(client, mid)
    body = {"meeting": {"baseRevision": m["revision"], "title": "Once"}, "tasks": {"create": [{"tempId": "n1", "title": "Only once"}]}}
    headers = {"Idempotency-Key": "commit-123"}
    first = await client.post(f"/v1/meetings/{mid}/edits", json=body, headers=headers)
    again = await client.post(f"/v1/meetings/{mid}/edits", json=body, headers=headers)  # the answer was lost; the client sends it again
    assert first.status_code == 200 and again.status_code == 200 and again.headers.get("Idempotent-Replay") == "true"
    assert again.json() == first.json()
    assert len((await state(client, mid))[3]) == 1 and (await state(client, mid))[0]["revision"] == m["revision"] + 1
    # Without the key the same stale revision is, rightly, a conflict.
    assert (await client.post(f"/v1/meetings/{mid}/edits", json=body)).status_code == 409


@pytest.mark.asyncio
async def test_two_commits_at_once_only_one_wins(client):
    mid, ids = await scribe(client)
    m, *_ = await state(client, mid)
    mk = lambda title: {"meeting": {"baseRevision": m["revision"], "title": title}, "segments": [{"id": ids[0], "baseRevision": 0, "text": title}]}  # noqa: E731
    a, b = await asyncio.gather(client.post(f"/v1/meetings/{mid}/edits", json=mk("From A")), client.post(f"/v1/meetings/{mid}/edits", json=mk("From B")))
    assert sorted([a.status_code, b.status_code]) == [200, 409]
    m2, segs, *_ = await state(client, mid)
    assert m2["title"] == segs[0]["text"]  # one editor's whole commit, never a mix of the two


@pytest.mark.asyncio
async def test_a_scribe_that_is_still_recording_has_no_editable_transcript_or_summary(client):
    mid = (await client.post("/v1/meetings", json={})).json()["id"]
    await client.post(f"/v1/meetings/{mid}/start", json={"consentAcknowledged": True})
    m = (await client.get(f"/v1/meetings/{mid}")).json()
    res = await client.post(f"/v1/meetings/{mid}/edits", json={"segments": [{"id": "x", "baseRevision": 0, "text": "t"}]})
    assert res.status_code == 409 and res.json()["error"] == "transcript_not_editable"
    res = await client.post(f"/v1/meetings/{mid}/edits", json={"summary": {"baseRevision": 0, "overview": "o"}})
    assert res.status_code == 409 and res.json()["error"] == "summary_not_editable"
    assert (await client.post(f"/v1/meetings/{mid}/edits", json={"meeting": {"baseRevision": m["revision"], "title": "Titled while recording"}})).status_code == 200


@pytest.mark.asyncio
async def test_the_single_edit_endpoints_honour_a_base_revision_when_given(client):
    mid, _ = await scribe(client)
    m, *_ = await state(client, mid)
    ok = await client.patch(f"/v1/meetings/{mid}", json={"title": "First", "baseRevision": m["revision"]})
    assert ok.status_code == 200 and ok.json()["revision"] == m["revision"] + 1
    stale = await client.patch(f"/v1/meetings/{mid}", json={"title": "Second", "baseRevision": m["revision"]})
    assert stale.status_code == 409 and stale.json()["items"][0]["kind"] == "meeting"
    t = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "x"})).json()
    assert (await client.patch(f"/v1/tasks/{t['id']}", json={"status": "done", "baseRevision": t["revision"]})).status_code == 200
    stale = await client.patch(f"/v1/tasks/{t['id']}", json={"status": "todo", "baseRevision": t["revision"]})
    assert stale.status_code == 409 and stale.json()["items"][0]["kind"] == "task"


@pytest.mark.asyncio
async def test_a_summary_a_person_wrote_is_not_replaced_without_saying_so(client):
    mid, _ = await scribe(client)
    await client.post(f"/v1/meetings/{mid}/edits", json={"summary": {"baseRevision": 0, "overview": "Mine"}})
    res = await client.post(f"/v1/meetings/{mid}/summary/regenerate")
    assert res.status_code == 409 and res.json()["error"] == "summary_edited"
    assert (await client.get(f"/v1/meetings/{mid}/summary")).json()["overview"] == "Mine"
    assert (await client.post(f"/v1/meetings/{mid}/summary/regenerate", json={"overwriteEdits": True})).status_code == 202


@pytest.mark.asyncio
async def test_a_task_says_whether_its_project_is_its_own_or_its_scribes(client):
    plan = (await client.post("/v1/projects", json={"name": "Plan"})).json()
    other = (await client.post("/v1/projects", json={"name": "Other"})).json()
    mid, _ = await scribe(client, projectId=plan["id"])
    t = (await client.post(f"/v1/meetings/{mid}/tasks", json={"title": "x"})).json()
    assert t["projectId"] == plan["id"] and t["ownProjectId"] is None
    moved = (await client.patch(f"/v1/tasks/{t['id']}", json={"projectId": other["id"]})).json()
    assert moved["projectId"] == other["id"] and moved["ownProjectId"] == other["id"]
