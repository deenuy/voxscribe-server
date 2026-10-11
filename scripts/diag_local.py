import asyncio, json, os, signal, subprocess, sys, time, uuid
import httpx, psycopg, websockets
sys.path.insert(0, os.getcwd())
from app.local.migrations import run_migrations
BASE = "postgresql://localhost:5432/ideascribe_dev"; TOKEN = "diag"; MODE = sys.argv[1] if len(sys.argv) > 1 else "local"
subprocess.run(["say", "-v", "Samantha", "-o", "/tmp/_clip.aiff", "The market analysis shows strong growth potential. What is the pricing tier for enterprise customers?"], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/_clip.aiff", "-ar", "16000", "-ac", "1", "-f", "s16le", "/tmp/_clip.raw"], check=True)
raw = open("/tmp/_clip.raw", "rb").read()
async def main():
    schema = f"e2e_{uuid.uuid4().hex[:8]}"
    async with await psycopg.AsyncConnection.connect(BASE) as c:
        await c.execute(f"CREATE SCHEMA {schema}"); await c.commit()
    await run_migrations(BASE, schema=schema)
    env = dict(os.environ, VOXSCRIBE_TOKEN=TOKEN, VOXSCRIBE_LLM_MODEL="gemma4:e4b", VOXSCRIBE_LIVE_MODEL="gemma4:e4b")
    eng = subprocess.Popen(["uv", "run", "voxscribe-engine", "--port", "0", "--providers", MODE, "--database-url", f"{BASE}?options=-c%20search_path%3D{schema},public"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    t0 = time.time()
    try:
        port = json.loads(eng.stdout.readline())["port"]; api = f"http://127.0.0.1:{port}/v1"
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60) as http:
            h = (await http.get(f"{api}/health")).json(); print("health", h["status"], {k: (v["ok"], v["name"]) for k, v in h["providers"].items()})
            src_path = f"/tmp/_diag_pricing_{schema}.txt"
            open(src_path, "w").write("Enterprise pricing policy. The enterprise pricing tier costs 49 dollars per seat per month and includes single sign on and priority support. The starter tier is free for up to five seats.")
            sr = await http.post(f"{api}/sources", json={"kind": "file", "uri": src_path}); print("source create", sr.status_code)
            sid = sr.json()["id"]
            for _ in range(60):
                await asyncio.sleep(1); sst = (await http.get(f"{api}/sources/{sid}")).json()["status"]
                if sst in ("ready", "error"): break
            print("source status", sst, (await http.get(f"{api}/sources/{sid}")).json().get("error"))
            m = (await http.post(f"{api}/meetings", json={"title": "diag"})).json(); mid = m["id"]
            print("attach", (await http.post(f"{api}/meetings/{mid}/attachments", json={"sourceId": sid})).status_code)
            await http.post(f"{api}/meetings/{mid}/start", json={"consentAcknowledged": True})
            events = []
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/meetings/{mid}/live?token={TOKEN}") as ws:
                async def rd():
                    try:
                        async for msg in ws:
                            if isinstance(msg, str): e = json.loads(msg); events.append((round(time.time() - t0, 1), e))
                    except Exception: pass
                task = asyncio.create_task(rd())
                await ws.send(json.dumps({"type": "enhance", "live": True, "style": "standard"}))
                await ws.send(json.dumps({"type": "assist", "enabled": True}))
                step = 3200
                for i in range(0, len(raw), step):
                    await ws.send(b"\x00" + raw[i:i + step]); await asyncio.sleep(0.05)
                for _ in range(15): await ws.send(b"\x00" + b"\x00" * step); await asyncio.sleep(0.05)
                await ws.send(json.dumps({"type": "note", "blockId": "b1", "text": "- market looks strong", "offsetMs": 3000}))
                t_note = time.time() - t0
                await asyncio.sleep(40)
                task.cancel()
            kinds = {}
            for t, e in events: kinds[e["type"]] = kinds.get(e["type"], 0) + 1
            print("event kinds:", kinds, "| note sent at", round(t_note, 1), "s")
            for t, e in events:
                if e["type"] in ("transcript.stable", "enhanced.update", "assist.suggestion", "error", "status"): print("  ", t, json.dumps(e)[:230])
            en = (await http.get(f"{api}/meetings/{mid}/enhanced")).json(); print("enhanced via GET:", [(b["blockId"], b["status"], b["text"][:80]) for b in en["blocks"]])
            await http.post(f"{api}/meetings/{mid}/stop")
            for _ in range(40):
                await asyncio.sleep(2); st = (await http.get(f"{api}/meetings/{mid}")).json()["status"]
                if st == "done": break
            fails = []
            kinds_seen = {e["type"] for _, e in events}
            if "transcript.stable" not in kinds_seen: fails.append("no stable transcript")
            en_ok = [b for b in en["blocks"] if b["status"] == "stable" and b["text"] and b["citations"]]
            if not en_ok: fails.append("no enhanced block with text and citations")
            asg = [e for _, e in events if e["type"] == "assist.suggestion"]
            if not asg: fails.append("no assist suggestion")
            elif not all(a.get("citations") for a in asg): fails.append("assist without citations")
            print("assist suggestions:", [(a["question"][:50], a["answer"][:90], [c.get("title") for c in a["citations"]]) for a in asg])
            print("status after stop:", st, "after", round(time.time() - t0, 1), "s"); sm = (await http.get(f"{api}/meetings/{mid}/summary")).json(); print("summary:", sm["status"], (sm.get("overview") or "")[:100])
            if sm["status"] != "ready" or not sm.get("overview"): fails.append("summary not ready")
            print("RESULT:", "PASS" if not fails else "FAIL " + "; ".join(fails))
            os.remove(src_path)
    finally:
        try: os.killpg(eng.pid, signal.SIGTERM)
        except Exception: pass
        try: err = eng.stderr.read()
        except Exception: err = ""
        async with await psycopg.AsyncConnection.connect(BASE) as c:
            await c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE"); await c.commit()
        if os.environ.get("DIAG_ALL"): print("\n".join(l for l in err.splitlines() if "assist" in l.lower() or "question" in l.lower()))
        lines = [l for l in err.splitlines() if l.strip() and "RuntimeWarning" not in l and "warnings.warn" not in l and not l.startswith("INFO")]
        print("--- engine log (non-INFO) ---"); print("\n".join(lines[-25:]))
asyncio.run(main())
