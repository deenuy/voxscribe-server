"""Making a scribe from a transcript someone already has, and adding web pages as resources."""

import asyncio

import pytest

from app.local import transcript_import as ti
from app.local import youtube_import as yt

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello everyone, welcome.

2
00:00:04,000 --> 00:00:07,250
<i>Alice:</i> Let's start with pricing.
"""

VTT = """WEBVTT

NOTE a comment

intro
00:01.000 --> 00:03.000
<v Bob>Thanks for joining.</v>

00:00:04.000 --> 00:00:06.000 align:start
Second cue
spanning two rows
"""


# ------------------------------------------------------------------------------------------------ the parsers

def test_srt_has_times_and_text_and_a_named_speaker():
    lines = ti.parse_transcript(SRT, "meeting.srt")
    assert [(l["startMs"], l["endMs"], l["speaker"], l["text"]) for l in lines] == [
        (1000, 3500, None, "Hello everyone, welcome."),
        (4000, 7250, "Alice", "Let's start with pricing."),
    ]


def test_vtt_ignores_the_header_notes_and_ids_reads_voice_tags_and_joins_rows():
    lines = ti.parse_transcript(VTT, "talk.vtt")
    assert [(l["startMs"], l["speaker"], l["text"]) for l in lines] == [(1000, "Bob", "Thanks for joining."), (4000, None, "Second cue spanning two rows")]


def test_a_format_is_recognised_by_its_content_when_the_name_says_nothing():
    assert len(ti.parse_transcript(VTT, "download")) == 2
    assert len(ti.parse_transcript(SRT, None)) == 2


def test_plain_text_is_one_line_per_paragraph_with_estimated_times_in_order():
    lines = ti.parse_transcript("First paragraph here.\nStill the first.\n\nSecond paragraph.\n\n# A heading\n\n- a bullet", "notes.md")
    assert [l["text"] for l in lines] == ["First paragraph here. Still the first.", "Second paragraph.", "A heading", "a bullet"]
    starts = [l["startMs"] for l in lines]
    assert starts == sorted(starts) and len(set(starts)) == 4 and all(l["endMs"] > l["startMs"] for l in lines)
    assert all(b["startMs"] >= a["endMs"] for a, b in zip(lines, lines[1:]))  # no two lines share a moment


def test_lines_with_their_own_times_and_speakers_keep_them():
    text = "[00:12] Alice: We should ship offline first.\n[00:20] Bob: Agreed.\n01:05 Alice: Next topic."
    lines = ti.parse_transcript(text, "call.txt")
    assert [(l["startMs"], l["speaker"], l["text"]) for l in lines] == [(12000, "Alice", "We should ship offline first."), (20000, "Bob", "Agreed."), (65000, "Alice", "Next topic.")]


def test_a_colon_in_ordinary_prose_is_not_a_speaker():
    lines = ti.parse_transcript("Note: this is a sentence that happens to have a colon in the middle of it.\n\nAnother paragraph.", "a.txt")
    assert all(l["speaker"] is None for l in lines) and lines[0]["text"].startswith("Note:")


def test_times_that_go_backwards_are_never_earlier_than_the_line_before():
    lines = ti.parse_transcript("[00:30] A: one\n[00:10] B: two", "x.txt")
    assert lines[1]["startMs"] >= lines[0]["endMs"]


def test_bad_input_is_a_clear_error():
    for bad in ("", "   \n  ", None):
        with pytest.raises(ti.TranscriptError):
            ti.parse_transcript(bad)
    with pytest.raises(ti.TranscriptError):
        ti.parse_transcript("WEBVTT\n\n", "empty.vtt")
    with pytest.raises(ti.TranscriptError):
        ti.parse_transcript("x" * (ti.MAX_CHARS + 1))


def test_a_very_long_line_is_cut_at_sentence_ends_to_fit():
    lines = ti.parse_transcript("A sentence. " * 3000, "long.txt")
    assert len(lines) > 1 and all(len(l["text"]) <= ti.MAX_LINE_CHARS for l in lines)


# ------------------------------------------------------------------------------------------------ YouTube

def test_every_kind_of_youtube_link_gives_the_video_id():
    for url in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=5", "https://youtube.com/shorts/dQw4w9WgXcQ?si=abc", "https://m.youtube.com/embed/dQw4w9WgXcQ", "https://www.youtube.com/live/dQw4w9WgXcQ"):
        assert yt.video_id(url) == "dQw4w9WgXcQ", url
    for bad in ("", "not a url", "https://example.com/watch?v=dQw4w9WgXcQ", "https://youtube.com/watch?v=short", "https://youtube.com/channel/abc", "javascript:alert(1)"):
        with pytest.raises(yt.YouTubeError):
            yt.video_id(bad)


def test_captions_become_sentences_with_the_first_time_and_no_sound_effects():
    snippets = [
        {"text": "so the plan is", "start": 1.0, "duration": 2.0}, {"text": "[Music]", "start": 3.0, "duration": 1.0},
        {"text": "offline first.", "start": 3.5, "duration": 2.0}, {"text": "Any questions?", "start": 6.0, "duration": 2.0}, {"text": "no trailing stop", "start": 9.0, "duration": 1.0},
    ]
    lines = yt.captions_to_lines(snippets)
    assert [(l["startMs"], l["text"]) for l in lines] == [(1000, "so the plan is offline first."), (6000, "Any questions?"), (9000, "no trailing stop")]


@pytest.mark.asyncio
async def test_a_caption_failure_is_said_in_plain_words():
    class TranscriptsDisabled(Exception):
        pass

    class VideoUnavailable(Exception):
        pass

    for exc, said in ((TranscriptsDisabled(), "no captions"), (VideoUnavailable(), "not available"), (RuntimeError("boom"), "could not be fetched")):
        def fail(_vid, exc=exc):
            raise exc

        with pytest.raises(yt.YouTubeError, match=said):
            await yt.fetch_captions("dQw4w9WgXcQ", fetch=fail)
    with pytest.raises(yt.YouTubeError, match="no captions"):
        await yt.fetch_captions("dQw4w9WgXcQ", fetch=lambda _v: [])


# ------------------------------------------------------------------------------------------------ the endpoint

async def wait_done(client, meeting_id, *, gone=False):
    for _ in range(100):
        res = await client.get(f"/v1/meetings/{meeting_id}")
        if gone and res.status_code == 404:
            return None
        if not gone and res.status_code == 200 and res.json()["status"] == "done":
            return res.json()
        await asyncio.sleep(0.05)
    raise AssertionError("the import did not finish")


@pytest.mark.asyncio
async def test_importing_a_transcript_file_makes_a_finished_scribe_with_every_line(client):
    res = await client.post("/v1/meetings/import", json={"kind": "file", "text": SRT, "filename": "Pricing_review.srt"})
    assert res.status_code == 202
    mid = res.json()["meetingId"]
    meeting = await wait_done(client, mid)
    assert meeting["title"] == "Pricing review" and meeting["origin"] == "file" and meeting["segmentCount"] == 2 and meeting["wordCount"] > 0
    segs = (await client.get(f"/v1/meetings/{mid}/transcript")).json()
    assert segs["editable"] is True and [s["text"] for s in segs["segments"]] == ["Hello everyone, welcome.", "Let's start with pricing."]
    assert segs["segments"][1]["speaker"] == "Alice" and segs["segments"][1]["startMs"] == 4000


@pytest.mark.asyncio
async def test_pasted_text_into_a_project_with_a_title(client):
    project = (await client.post("/v1/projects", json={"name": "Plan"})).json()
    res = await client.post("/v1/meetings/import", json={"kind": "text", "title": "My notes", "text": "One thought.\n\nAnother thought.", "projectId": project["id"]})
    meeting = await wait_done(client, res.json()["meetingId"])
    assert meeting["title"] == "My notes" and meeting["projectId"] == project["id"] and meeting["origin"] == "text" and meeting["segmentCount"] == 2


@pytest.mark.asyncio
async def test_a_transcript_file_on_this_computer_can_be_imported_by_path(client, tmp_path):
    f = tmp_path / "standup.vtt"
    f.write_text(VTT)
    res = await client.post("/v1/meetings/import", json={"kind": "file", "path": str(f)})
    assert res.status_code == 202
    assert (await wait_done(client, res.json()["meetingId"]))["title"] == "standup"
    assert (await client.post("/v1/meetings/import", json={"kind": "file", "path": str(tmp_path / "missing.txt")})).status_code == 400


@pytest.mark.asyncio
async def test_bad_imports_are_refused_up_front_and_make_nothing(client):
    for body in (
        {"kind": "file"}, {"kind": "text", "text": "   "}, {"kind": "file", "text": "WEBVTT\n\n", "filename": "x.vtt"},
        {"kind": "youtube"}, {"kind": "youtube", "url": "https://example.com/x"}, {"kind": "text", "text": "ok", "title": "x" * 301},
        {"kind": "text", "text": "ok", "projectId": "no-such-project"},
    ):
        assert (await client.post("/v1/meetings/import", json=body)).status_code == 400, body
    assert (await client.get("/v1/meetings")).json()["items"] == []  # not even an empty scribe


@pytest.mark.asyncio
async def test_audio_is_honestly_not_available_instead_of_saving_made_up_text(client):
    res = await client.post("/v1/meetings/import", json={"kind": "audio", "path": "/tmp/x.wav"})
    assert res.status_code == 501 and res.json()["error"] == "not_implemented"


@pytest.mark.asyncio
async def test_youtube_needs_connectors_to_be_allowed(client):
    res = await client.post("/v1/meetings/import", json={"kind": "youtube", "url": "https://youtu.be/dQw4w9WgXcQ"})
    assert res.status_code == 403 and res.json()["error"] == "network_disabled"


@pytest.mark.asyncio
async def test_youtube_import_uses_the_captions_and_the_videos_title(client, monkeypatch):
    await client.put("/v1/settings", json={"privacyMode": "connectors"})

    async def captions(_vid, fetch=None):
        return [{"text": "Welcome to the talk.", "start": 0.0, "duration": 2.0}, {"text": "Today: pricing.", "start": 2.0, "duration": 3.0}]

    async def title(_vid):
        return "A talk about pricing"

    monkeypatch.setattr(yt, "fetch_captions", captions)
    monkeypatch.setattr(yt, "video_title", title)
    res = await client.post("/v1/meetings/import", json={"kind": "youtube", "url": "https://youtu.be/dQw4w9WgXcQ"})
    assert res.status_code == 202
    meeting = await wait_done(client, res.json()["meetingId"])
    assert meeting["title"] == "A talk about pricing" and meeting["origin"] == "youtube" and meeting["segmentCount"] == 2


@pytest.mark.asyncio
async def test_a_youtube_import_that_fails_leaves_nothing_behind_and_says_why(client, monkeypatch):
    await client.put("/v1/settings", json={"privacyMode": "connectors"})

    async def no_captions(_vid, fetch=None):
        raise yt.YouTubeError("this video has no captions to import")

    monkeypatch.setattr(yt, "fetch_captions", no_captions)
    res = await client.post("/v1/meetings/import", json={"kind": "youtube", "url": "https://youtu.be/dQw4w9WgXcQ"})
    mid, job = res.json()["meetingId"], res.json()["jobId"]
    await wait_done(client, mid, gone=True)
    assert (await client.get(f"/v1/meetings/{mid}")).status_code == 404  # no empty "Imported" scribe left in the list


# ------------------------------------------------------------------------------------------------ web pages

@pytest.mark.asyncio
async def test_a_web_page_resource_is_fetched_indexed_and_can_be_attached(client, monkeypatch):
    await client.put("/v1/settings", json={"privacyMode": "connectors"})
    from app.local import url_fetcher

    async def page(_url, **_kw):
        return "The enterprise tier costs forty nine dollars per seat per month. " * 10

    monkeypatch.setattr(url_fetcher, "fetch_url", page)
    made = await client.post("/v1/sources", json={"kind": "url", "uri": "https://example.com/pricing"})
    assert made.status_code == 201
    sid = made.json()["id"]
    for _ in range(100):
        got = (await client.get(f"/v1/sources/{sid}")).json()
        if got["status"] in ("ready", "error"):
            break
        await asyncio.sleep(0.05)
    assert got["status"] == "ready", got.get("error")
    assert (await client.get(f"/v1/sources/{sid}/chunks")).json()["items"], "the page was read and indexed"
    mid = (await client.post("/v1/meetings", json={})).json()["id"]
    assert (await client.post(f"/v1/meetings/{mid}/attachments", json={"sourceId": sid})).status_code in (200, 201, 204)
    assert [a["sourceId"] for a in (await client.get(f"/v1/meetings/{mid}/attachments")).json()["items"]] == [sid]


@pytest.mark.asyncio
async def test_a_web_page_that_cannot_be_read_ends_in_error_not_ready(client, monkeypatch):
    await client.put("/v1/settings", json={"privacyMode": "connectors"})
    from app.local import url_fetcher

    async def blocked(_url, **_kw):
        raise ValueError("Blocked IP address: 127.0.0.1")

    monkeypatch.setattr(url_fetcher, "fetch_url", blocked)
    sid = (await client.post("/v1/sources", json={"kind": "url", "uri": "http://127.0.0.1/admin"})).json()["id"]
    for _ in range(100):
        got = (await client.get(f"/v1/sources/{sid}")).json()
        if got["status"] in ("ready", "error"):
            break
        await asyncio.sleep(0.05)
    assert got["status"] == "error" and "Blocked" in got["error"]
