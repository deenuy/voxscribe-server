"""Every route the engine serves, checked for the same production properties.

This walks the application's real route table (so a new endpoint is covered the day it is added) and sends
each one the same awkward requests: an id that does not exist, a body that is not JSON, a body of the wrong
shape, hostile query values, a method it does not allow. The rules:
  - never a 500 (a client mistake is a 4xx, and nothing leaks);
  - every error has the one error shape {"error": code, "detail": text};
  - a missing resource is 404, a bad request is 400, a wrong method is 405.
"""

import json
import re

import pytest

SKIP = re.compile(r"^/(docs|redoc|openapi\.json)|/live$|^/v1/events$")
FUZZ_IDS = ["no-such-id", "00000000-0000-0000-0000-000000000000", "a" * 300, "x'; DROP TABLE meetings;--", "%00", "../../etc/passwd", "日本語"]


def routes(app):
    """Every (method, path) the app documents, from its OpenAPI description (works however routers are mounted)."""
    out = []
    for path, ops in app.openapi()["paths"].items():
        if SKIP.search(path):
            continue
        for method in ops:
            if method.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                out.append((method.upper(), path))
    return sorted(out)


def fill(path, value):
    return re.sub(r"\{[^}]+\}", lambda _m: value, path)


async def send(client, method, path, **kw):
    return await client.request(method, path, **kw)


def error_shape_ok(res):
    if res.status_code < 400:
        return True
    try:
        body = res.json()
    except Exception:
        return False
    return isinstance(body, dict) and ("error" in body and "detail" in body)


@pytest.mark.asyncio
async def test_no_route_ever_answers_500_or_breaks_the_error_shape(client):
    app = client._transport.app
    problems = []
    table = routes(app)
    assert len(table) > 50, f"the route walk found only {len(table)} routes"  # a vacuous pass must not look green
    sent = 0
    for method, path in table:
        has_id = "{" in path
        attempts = []
        for fid in (FUZZ_IDS if has_id else [None]):
            url = fill(path, fid) if has_id else path
            attempts.append((url, {}))
            if method in ("POST", "PUT", "PATCH"):
                attempts.append((url, {"content": b"this is not json", "headers": {"content-type": "application/json"}}))
                attempts.append((url, {"json": []}))
                attempts.append((url, {"json": {"x": "y" * 5000}}))
                attempts.append((url, {"json": {"title": None, "name": None, "text": None, "kind": 7}}))
            if method == "GET":
                attempts.append((url + "?limit=-5&cursor=zzz&query=%27%22%3C", {}))
                attempts.append((url + "?limit=999999999999999999999", {}))
        for url, kw in attempts:
            sent += 1
            try:
                res = await send(client, method, url, **kw)
            except Exception as exc:  # the app itself raised: worst case
                problems.append(f"{method} {url[:60]} raised {type(exc).__name__}: {str(exc)[:80]}")
                continue
            if res.status_code >= 500:
                problems.append(f"{method} {url[:60]} -> {res.status_code} {res.text[:80]}")
            elif not error_shape_ok(res):
                problems.append(f"{method} {url[:60]} -> {res.status_code} with a non-standard error body: {res.text[:80]}")
    print(f"routes={len(table)} requests={sent}")
    assert not problems, "\n" + "\n".join(sorted(set(problems))[:60])


# ----------------------------------------------------------------------------- CRUD behaviour per resource


async def _status(client, method, url, **kw):
    return (await client.request(method, url, **kw)).status_code


@pytest.mark.asyncio
async def test_bad_input_is_refused_with_a_400_and_never_stored(client):
    c = client
    assert await _status(c, "POST", "/v1/projects", json={"name": "   "}) == 400
    assert await _status(c, "POST", "/v1/projects", json={"name": "n" * 121}) == 400
    pid = (await c.post("/v1/projects", json={"name": "  Roadmap  "})).json()["id"]
    assert (await c.get(f"/v1/projects/{pid}")).json()["name"] == "Roadmap"  # trimmed
    assert await _status(c, "PATCH", f"/v1/projects/{pid}", json={"name": ""}) == 400
    assert await _status(c, "POST", "/v1/ideas", json={"title": "x", "type": "nonsense"}) == 400
    assert await _status(c, "POST", "/v1/ideas", json={"title": "x", "projectId": "no-such"}) == 400
    assert await _status(c, "POST", "/v1/ideas", json={"title": "x", "projectId": pid}) == 201
    mid = (await c.post("/v1/meetings", json={"title": "   "})).json()
    assert mid["title"] == "New meeting"  # a blank title falls back to the default
    assert await _status(c, "POST", "/v1/meetings", json={"title": "t" * 301}) == 400
    assert await _status(c, "PATCH", f"/v1/meetings/{mid['id']}", json={"title": " "}) == 400
    assert await _status(c, "PATCH", f"/v1/meetings/{mid['id']}", json={"autoStopMinutes": -5}) == 400
    assert await _status(c, "PUT", f"/v1/meetings/{mid['id']}/notes", json={"blocks": [{"blockId": "a", "text": "x", "offsetMs": -1}]}) == 400
    assert await _status(c, "PUT", f"/v1/meetings/{mid['id']}/notes", json={"blocks": [{"blockId": "a", "text": "x" * 50001, "offsetMs": 0}]}) == 400
    assert await _status(c, "POST", "/v1/labels", json={"name": 7}) == 400


@pytest.mark.asyncio
async def test_resources_have_the_operations_a_client_needs(client):
    c = client
    pid = (await c.post("/v1/projects", json={"name": "P"})).json()["id"]
    assert await _status(c, "GET", f"/v1/projects/{pid}") == 200 and await _status(c, "GET", "/v1/projects/nope") == 404
    assert await _status(c, "GET", "/v1/projects/nope/features") == 404
    assert await _status(c, "POST", "/v1/projects/nope/features", json={"name": "f"}) == 404

    lid = (await c.post("/v1/labels", json={"name": "L"})).json()["id"]
    other = (await c.post("/v1/labels", json={"name": "L2"})).json()["id"]
    renamed = await c.patch(f"/v1/labels/{lid}", json={"name": "Renamed", "color": "#fff"})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Renamed"
    assert await _status(c, "PATCH", f"/v1/labels/{other}", json={"name": "Renamed"}) == 409  # names stay unique
    assert await _status(c, "PATCH", "/v1/labels/nope", json={"name": "x"}) == 404
    assert await _status(c, "PATCH", f"/v1/labels/{lid}", json={}) == 400

    tid = (await c.post("/v1/chat/threads", json={})).json()["id"]
    assert await _status(c, "DELETE", f"/v1/chat/threads/{tid}") == 204
    assert await _status(c, "GET", f"/v1/chat/threads/{tid}") == 404
    assert await _status(c, "DELETE", f"/v1/chat/threads/{tid}") == 204  # idempotent
    assert await _status(c, "POST", f"/v1/chat/threads/{tid}/messages", json={"text": "hi"}) == 404

    mid = (await c.post("/v1/meetings", json={})).json()["id"]
    assert (await c.get(f"/v1/meetings/{mid}/attachments")).json() == {"items": []}
    assert await _status(c, "GET", "/v1/meetings/nope/attachments") == 404
    assert await _status(c, "POST", f"/v1/meetings/{mid}/attachments", json={"sourceId": "no-such-source"}) == 404
    assert await _status(c, "POST", f"/v1/meetings/{mid}/attachments", json=["x"]) == 400


@pytest.mark.asyncio
async def test_empty_chat_message_and_bad_imports_are_400_not_500(client):
    c = client
    tid = (await c.post("/v1/chat/threads", json={})).json()["id"]
    assert await _status(c, "POST", f"/v1/chat/threads/{tid}/messages", json={"text": "   "}) == 400
    assert await _status(c, "POST", "/v1/meetings/import", json={"kind": "file"}) == 400  # no path
    assert await _status(c, "POST", "/v1/meetings/import", json={"kind": "youtube"}) == 400  # no url
    assert await _status(c, "POST", "/v1/meetings/import", content=b"nope", headers={"content-type": "application/json"}) == 400


@pytest.mark.asyncio
async def test_deletes_answer_204_with_no_body_and_unknown_addresses_use_the_error_shape(client):
    c = client
    pid = (await c.post("/v1/projects", json={"name": "gone"})).json()["id"]
    iid = (await c.post("/v1/ideas", json={"title": "gone"})).json()["id"]
    for url in (f"/v1/projects/{pid}", f"/v1/ideas/{iid}"):
        r = await c.delete(url)
        assert r.status_code == 204 and r.content == b""  # a 204 carries no body
        assert (await c.delete(url)).status_code in (204, 404)
    r = await c.get("/v1/no/such/address")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = await c.put("/v1/meetings", json={})
    assert r.status_code == 405 and r.json()["error"] == "method_not_allowed"
    assert (await c.get("/v1/ideas/%00")).status_code == 400  # a value the database cannot store is the caller's mistake


@pytest.mark.asyncio
async def test_idea_extraction_is_honest_about_not_being_built(client):
    mid = (await c_create(client))
    assert await _status(client, "POST", "/v1/meetings/nope/extract-ideas", json={}) == 404
    assert await _status(client, "POST", f"/v1/meetings/{mid}/extract-ideas", json={}) == 409  # nothing to extract from yet


async def c_create(client):
    return (await client.post("/v1/meetings", json={})).json()["id"]
