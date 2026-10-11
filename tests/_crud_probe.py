import pytest, json
async def show(client, label, method, url, **kw):
    r = await client.request(method, url, **kw)
    body = r.text[:90].replace("\n"," ")
    print(f"{label:46s} {method:6s} {url[:40]:40s} -> {r.status_code} {body}")
    return r

@pytest.mark.asyncio
async def test_probe(client):
    c = client
    print()
    # --- meetings
    m = (await show(c,"create meeting (blank title)","POST","/v1/meetings",json={"title":"   "})).json()
    await show(c,"create meeting (title 5000 chars)","POST","/v1/meetings",json={"title":"x"*5000})
    mid = (await c.post("/v1/meetings",json={"title":"A"})).json()["id"]
    await show(c,"patch meeting unknown field","PATCH",f"/v1/meetings/{mid}",json={"bogus":1})
    await show(c,"patch meeting empty body","PATCH",f"/v1/meetings/{mid}",json={})
    await show(c,"patch meeting title null","PATCH",f"/v1/meetings/{mid}",json={"title":None})
    await show(c,"patch meeting blank title","PATCH",f"/v1/meetings/{mid}",json={"title":""})
    await show(c,"patch meeting enhanceStyle bad","PATCH",f"/v1/meetings/{mid}",json={"enhanceStyle":"loud"})
    await show(c,"patch meeting autoStop -5","PATCH",f"/v1/meetings/{mid}",json={"autoStopMinutes":-5})
    await show(c,"delete meeting","DELETE",f"/v1/meetings/{mid}")
    await show(c,"delete meeting again (idempotent?)","DELETE",f"/v1/meetings/{mid}")
    await show(c,"get deleted meeting","GET",f"/v1/meetings/{mid}")
    await show(c,"patch deleted meeting","PATCH",f"/v1/meetings/{mid}",json={"title":"z"})
    mid2 = (await c.post("/v1/meetings",json={})).json()["id"]
    await show(c,"list attachments","GET",f"/v1/meetings/{mid2}/attachments")
    await show(c,"put notes duplicate ids","PUT",f"/v1/meetings/{mid2}/notes",json={"blocks":[{"blockId":"d","text":"a","offsetMs":0},{"blockId":"d","text":"b","offsetMs":1}]})
    await show(c,"put notes 1MB text","PUT",f"/v1/meetings/{mid2}/notes",json={"blocks":[{"blockId":"big","text":"x"*1000000,"offsetMs":0}]})
    await show(c,"put notes negative offset","PUT",f"/v1/meetings/{mid2}/notes",json={"blocks":[{"blockId":"neg","text":"a","offsetMs":-1}]})
    await show(c,"start without consent","POST",f"/v1/meetings/{mid2}/start",json={})
    # --- projects
    p = (await show(c,"create project blank name","POST","/v1/projects",json={"name":"  "})).json()
    pr = (await show(c,"create project","POST","/v1/projects",json={"name":"P1"})).json()
    await show(c,"create project duplicate name","POST","/v1/projects",json={"name":"P1"})
    await show(c,"get project by id","GET",f"/v1/projects/{pr.get('id')}")
    await show(c,"patch project blank name","PATCH",f"/v1/projects/{pr.get('id')}",json={"name":""})
    await show(c,"patch unknown project","PATCH","/v1/projects/nope",json={"name":"x"})
    await show(c,"delete project","DELETE",f"/v1/projects/{pr.get('id')}")
    await show(c,"delete project again","DELETE",f"/v1/projects/{pr.get('id')}")
    await show(c,"features of deleted project","GET",f"/v1/projects/{pr.get('id')}/features")
    # --- labels
    l = (await show(c,"create label","POST","/v1/labels",json={"name":"L1"})).json()
    await show(c,"create label duplicate","POST","/v1/labels",json={"name":"L1"})
    await show(c,"create label blank","POST","/v1/labels",json={"name":""})
    await show(c,"patch label (rename)","PATCH",f"/v1/labels/{l.get('id')}",json={"name":"L2"})
    await show(c,"delete label","DELETE",f"/v1/labels/{l.get('id')}")
    await show(c,"delete label again","DELETE",f"/v1/labels/{l.get('id')}")
    # --- ideas
    await show(c,"create idea blank title","POST","/v1/ideas",json={"title":""})
    i = (await show(c,"create idea","POST","/v1/ideas",json={"title":"Idea"})).json()
    await show(c,"create idea bad type","POST","/v1/ideas",json={"title":"x","type":"nonsense"})
    await show(c,"create idea unknown project","POST","/v1/ideas",json={"title":"x","projectId":"nope"})
    await show(c,"patch idea status bad","PATCH",f"/v1/ideas/{i.get('id')}",json={"status":"zzz"})
    await show(c,"delete idea","DELETE",f"/v1/ideas/{i.get('id')}")
    await show(c,"delete idea again","DELETE",f"/v1/ideas/{i.get('id')}")
    # --- chat
    t = (await show(c,"create thread","POST","/v1/chat/threads",json={})).json()
    await show(c,"delete thread","DELETE",f"/v1/chat/threads/{t.get('id')}")
    await show(c,"send empty message","POST",f"/v1/chat/threads/{t.get('id')}/messages",json={"text":""})
    await show(c,"send message unknown thread","POST","/v1/chat/threads/nope/messages",json={"text":"hi"})
    # --- settings
    await show(c,"get settings","GET","/v1/settings")
    await show(c,"put settings partial","PUT","/v1/settings",json={"privacyMode":"offline"})
    await show(c,"put settings bad enum","PUT","/v1/settings",json={"privacyMode":"banana"})
    await show(c,"get settings after partial","GET","/v1/settings")
    # --- data wipe
    await show(c,"DELETE /data with no confirmation","DELETE","/v1/data")
