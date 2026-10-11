"""Transcript editing, publishing, metadata and storage: the rules that keep a long transcript trustworthy."""

import asyncio
import uuid

import pytest

from app.local.models import Segment
from app.local.store_meetings import MeetingStore


def store_of(client) -> MeetingStore:
    return MeetingStore(client._transport.app.state.store.pool)


async def meeting_with_segments(client, texts=("Hello there", "Ship offline mode first"), publish=True, **create):
    mid = (await client.post("/v1/meetings", json=create)).json()["id"]
    store = store_of(client)
    await client.post(f"/v1/meetings/{mid}/start", json={"consentAcknowledged": True})
    ids = []
    for i, text in enumerate(texts):
        seg = Segment(id=str(uuid.uuid4()), startMs=i * 6000, endMs=i * 6000 + 5000, source="mic", speaker="Alice" if i % 2 == 0 else None, text=text, state="stable")
        await store.append_segment(mid, seg)
        ids.append(seg.id)
    await client.post(f"/v1/meetings/{mid}/stop")
    if publish:
        await asyncio.sleep(0)  # let the background finalisation start, then publish deterministically
        await store.mark_done(mid)
    return mid, ids


async def edit(client, mid, sid, rev=0, **body):
    return await client.patch(f"/v1/meetings/{mid}/transcript/{sid}", json={"baseRevision": rev, **body})


@pytest.mark.asyncio
async def test_a_recording_or_processing_transcript_cannot_be_edited(client):
    mid, ids = await meeting_with_segments(client, publish=False)
    # Force the state that stop leaves it in, whatever the background job has managed so far.
    async with client._transport.app.state.store.pool.connection() as conn:
        await conn.execute("UPDATE meetings SET status = 'processing' WHERE id = %s", (mid,))
        await conn.commit()
    res = await edit(client, mid, ids[0], text="Changed")
    assert res.status_code == 409 and res.json()["error"] == "transcript_not_editable"
    body = (await client.get(f"/v1/meetings/{mid}/transcript")).json()
    assert body["editable"] is False and body["segments"][0]["text"] == "Hello there"


@pytest.mark.asyncio
async def test_a_published_transcript_can_be_edited_and_says_so(client):
    mid, ids = await meeting_with_segments(client)
    before = (await client.get(f"/v1/meetings/{mid}/transcript")).json()
    assert before["editable"] is True and before["revision"] == 0

    res = await edit(client, mid, ids[0], text="  Hello, everyone  ", speaker="Alice K")
    assert res.status_code == 200
    out = res.json()
    assert out["segment"]["text"] == "Hello, everyone" and out["segment"]["speaker"] == "Alice K"
    assert out["segment"]["revision"] == 1 and out["segment"]["edited"] is True
    assert out["transcriptRevision"] == 1 and out["summaryStale"] is True

    after = (await client.get(f"/v1/meetings/{mid}/transcript")).json()
    assert after["revision"] == 1
    assert [s["text"] for s in after["segments"]] == ["Hello, everyone", "Ship offline mode first"]
    assert after["segments"][1]["edited"] is False


@pytest.mark.asyncio
async def test_the_recognisers_original_is_kept_in_the_edit_history(client):
    mid, ids = await meeting_with_segments(client)
    await edit(client, mid, ids[0], text="First fix")
    await edit(client, mid, ids[0], rev=1, text="Second fix")
    async with client._transport.app.state.store.pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT revision, before_text, after_text FROM transcript_edits WHERE segment_id = %s ORDER BY revision", (ids[0],)
        )).fetchall()
    assert rows == [(1, "Hello there", "First fix"), (2, "First fix", "Second fix")]


@pytest.mark.asyncio
async def test_a_stale_edit_is_refused_and_writes_nothing(client):
    mid, ids = await meeting_with_segments(client)
    assert (await edit(client, mid, ids[0], text="Mine")).status_code == 200
    stale = await edit(client, mid, ids[0], rev=0, text="Theirs, from an old copy")
    assert stale.status_code == 409 and stale.json()["error"] == "revision_conflict"
    seg = (await client.get(f"/v1/meetings/{mid}/transcript")).json()["segments"][0]
    assert seg["text"] == "Mine" and seg["revision"] == 1


@pytest.mark.asyncio
async def test_two_edits_at_once_one_wins_one_conflicts(client):
    mid, ids = await meeting_with_segments(client)
    a, b = await asyncio.gather(edit(client, mid, ids[0], text="A"), edit(client, mid, ids[0], text="B"))
    assert sorted([a.status_code, b.status_code]) == [200, 409]
    seg = (await client.get(f"/v1/meetings/{mid}/transcript")).json()["segments"][0]
    assert seg["revision"] == 1


@pytest.mark.asyncio
async def test_bad_edits_are_rejected_clearly(client):
    mid, ids = await meeting_with_segments(client)
    assert (await edit(client, mid, ids[0], text="   ")).status_code == 400
    assert (await edit(client, mid, ids[0], text="x" * 20_001)).status_code == 400
    assert (await edit(client, mid, ids[0], speaker="s" * 121)).status_code == 400
    assert (await client.patch(f"/v1/meetings/{mid}/transcript/{ids[0]}", json={"baseRevision": 0})).status_code == 400  # nothing to change
    assert (await edit(client, mid, "no-such-segment", text="x")).status_code == 404
    assert (await edit(client, str(uuid.uuid4()), ids[0], text="x")).status_code == 404
    assert (await client.patch(f"/v1/meetings/{mid}/transcript/{ids[0]}", json={"text": "x"})).status_code == 400  # no baseRevision


@pytest.mark.asyncio
async def test_an_edit_that_changes_nothing_writes_nothing(client):
    mid, ids = await meeting_with_segments(client)
    res = await edit(client, mid, ids[0], text="Hello there")
    assert res.status_code == 200 and res.json()["segment"]["revision"] == 0 and res.json()["transcriptRevision"] == 0


@pytest.mark.asyncio
async def test_clearing_a_speaker_name_is_allowed(client):
    mid, ids = await meeting_with_segments(client)
    out = (await edit(client, mid, ids[0], speaker="")).json()
    assert out["segment"].get("speaker") is None and out["segment"]["text"] == "Hello there"


@pytest.mark.asyncio
async def test_publishing_stores_the_metadata_a_record_needs(client):
    mid, ids = await meeting_with_segments(client, texts=("one two three", "four five"))
    m = (await client.get(f"/v1/meetings/{mid}")).json()
    assert m["status"] == "done" and m["segmentCount"] == 2 and m["wordCount"] == 5
    assert m["publishedAt"] and m["endedAt"] and m["startedAt"] and m["durationMs"] >= 0
    assert m["origin"] == "recorded" and m["transcriptRevision"] == 0 and m["summaryStale"] is False
    res = await edit(client, mid, ids[0], text="one two three four")
    assert res.status_code == 200, res.text
    m = (await client.get(f"/v1/meetings/{mid}")).json()
    assert m["wordCount"] == 6 and m["transcriptRevision"] == 1 and m["summaryStale"] is True


@pytest.mark.asyncio
async def test_publishing_settles_provisional_segments(client):
    mid = (await client.post("/v1/meetings", json={})).json()["id"]
    store = store_of(client)
    await store.append_segment(mid, Segment(id="p1", startMs=0, endMs=900, source="mic", text="half a sen", state="provisional"))
    await store.mark_done(mid)
    assert (await store.get_transcript(mid))[0].state == "stable"


@pytest.mark.asyncio
async def test_a_project_is_stored_with_the_meeting_and_must_exist(client):
    project = (await client.post("/v1/projects", json={"name": "Senior Care"})).json()
    mid = (await client.post("/v1/meetings", json={"projectId": project["id"]})).json()
    assert mid["projectId"] == project["id"]
    bad = await client.post("/v1/meetings", json={"projectId": "nope"})
    assert bad.status_code == 400 and bad.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_a_long_transcript_pages_and_edits_one_row_at_a_time(client):
    texts = [f"segment number {i} with several words in it" for i in range(120)]
    mid, ids = await meeting_with_segments(client, texts=texts)
    first = (await client.get(f"/v1/meetings/{mid}/transcript", params={"limit": 50})).json()
    assert len(first["segments"]) == 50 and first["nextCursor"] == 50
    last = (await client.get(f"/v1/meetings/{mid}/transcript", params={"limit": 50, "cursor": 100})).json()
    assert len(last["segments"]) == 20 and "nextCursor" not in last
    assert (await client.get(f"/v1/meetings/{mid}/transcript")).json()["segments"][119]["text"] == texts[119]  # no limit: everything, compatible with v1
    await edit(client, mid, ids[60], text="the only row that changed")
    async with client._transport.app.state.store.pool.connection() as conn:
        n = (await (await conn.execute("SELECT count(*) FROM transcript_segments WHERE meeting_id = %s AND edited_at IS NOT NULL", (mid,))).fetchone())[0]
    assert n == 1


@pytest.mark.asyncio
async def test_transcript_text_is_searchable_and_edits_are_exported(client):
    mid, ids = await meeting_with_segments(client, texts=("Legal must sign off the pricing", "Offline mode ships first"))
    async with client._transport.app.state.store.pool.connection() as conn:
        hit = (await (await conn.execute(
            "SELECT count(*) FROM transcript_segments WHERE meeting_id = %s AND search @@ plainto_tsquery('simple', 'pricing')", (mid,)
        )).fetchone())[0]
    assert hit == 1
    await edit(client, mid, ids[1], text="Offline mode ships in March")
    md = (await client.get(f"/v1/meetings/{mid}/export", params={"format": "markdown"})).text
    assert "Offline mode ships in March" in md and "ships first" not in md
