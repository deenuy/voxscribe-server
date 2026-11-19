"""The steps after Stop are published to the meeting's tenant and can be read back (PL36 B).

The real engine with identity on, on a throwaway schema. The model is a scripted fake: it answers the summary, title
and task prompts with fixed JSON, and can be made slow or held until a test lets it go."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from app.local import events, finalize, pipeline, task_service, tenancy
from app.local.models import Segment
from app.local.store_meetings import MeetingStore
from tests import test_tenancy

keys = test_tenancy.keys                     # the engine with identity on, and its signing key (fixtures)
tenant_engine = test_tenancy.tenant_engine

WAIT_S = 20
WORDS = ("We agreed that Dana will send the budget to the board by Friday and that the launch moves to "
         "March so the team has time to finish testing the new release properly before anyone outside sees it.")


class ScriptedModel:
    """Summary, title and task prompts answered with fixed JSON. `task_delay_s` blocks before the task answer;
    `hold` (a threading.Event) holds the summary answer until it is set."""

    name = "fake:scripted"

    def __init__(self, *, task_delay_s: float = 0.0, hold: threading.Event | None = None):
        self.task_delay_s = task_delay_s
        self.hold = hold
        self.summary_started = threading.Event()

    def stream(self, system, prompt, max_tokens=2048):
        if "task extraction" in system:
            if self.task_delay_s:
                time.sleep(self.task_delay_s)
            yield json.dumps({"tasks": [{"title": "Send the budget to the board", "detail": "Done when: sent.",
                                         "owner": "Dana", "dueDate": None}]})
        elif "titler" in system:
            yield json.dumps({"title": "Budget and launch"})
        else:
            self.summary_started.set()
            if self.hold is not None:
                self.hold.wait(WAIT_S)
            yield json.dumps({"overview": "The budget goes to the board and the launch moves.", "details": []})

    def complete(self, system, prompt, max_tokens=2048):
        return "".join(self.stream(system, prompt, max_tokens))


async def recorded_scribe(pool, text: str = WORDS) -> str:
    store = MeetingStore(pool)
    mid = str(uuid.uuid4())
    await store.create_meeting(meeting_id=mid, title="Planning")
    await store.start_meeting(mid)
    if text:
        await store.append_segment(mid, Segment(id=f"seg-{uuid.uuid4().hex[:8]}", startMs=0, endMs=9000,
                                                source="mic", text=text, state="stable"))
    return mid


def pipeline_events(sub) -> list[tuple[str, str, str | None]]:
    out = []
    while not sub.empty():
        e = sub.get_nowait()
        if e.type == "meeting.pipeline":
            out.append((e.data["step"], e.data["state"], e.data["reason"]))
    return out


async def steps(client, headers, mid) -> dict:
    res = await client.get(f"/v1/meetings/{mid}/pipeline", headers=headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["meetingId"] == mid and [s["step"] for s in body["steps"]] == ["transcript", "summary", "tasks"]
    return {s["step"]: (s["state"], s["reason"]) for s in body["steps"]}


async def until(client, headers, mid, check, what: str) -> dict:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        now = await steps(client, headers, mid)
        if check(now):
            return now
        await asyncio.sleep(0.05)
    raise AssertionError(f"{what}: still {now}")


async def finish(tasks) -> None:
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), WAIT_S)


async def test_every_step_after_stop_reaches_the_tenant_and_nobody_else(tenant_engine):
    client, app, as_user = tenant_engine
    app.state.providers["llm"] = ScriptedModel()
    alice, bob = as_user("alice"), as_user("bob")
    pool = app.state.store.pool
    a, b = tenancy.tenant_key("alice"), tenancy.tenant_key("bob")

    with tenancy.tenant_scope(b):
        bob_stream = events.subscribe()
        bob_sub = await bob_stream.__aenter__()
    with tenancy.tenant_scope(a):
        async with events.subscribe() as sub:
            mid = await recorded_scribe(pool)
            assert await steps(client, alice, mid) == {"transcript": ("working", "recording"),
                                                       "summary": ("working", "queued"),
                                                       "tasks": ("working", "queued")}
            assert (await client.post(f"/v1/meetings/{mid}/stop", headers=alice)).status_code == 200
            # The live socket schedules this once the last words are flushed; here, in the same tenant's scope.
            await finish([finalize.schedule_finalize(app, mid)])
            seen = pipeline_events(sub)
    await bob_stream.__aexit__(None, None, None)

    assert seen == [
        ("transcript", "working", None),
        ("transcript", "ready", None),
        ("summary", "working", "queued"),
        ("tasks", "working", "queued"),
        ("summary", "working", None),
        ("summary", "ready", None),
        ("tasks", "working", None),
        ("tasks", "ready", None),
    ]
    assert await steps(client, alice, mid) == {"transcript": ("ready", None), "summary": ("ready", None),
                                               "tasks": ("ready", None)}
    assert pipeline_events(bob_sub) == []
    assert (await client.get(f"/v1/meetings/{mid}/pipeline", headers=bob)).status_code == 404


async def test_a_task_timeout_has_a_reason_leaves_the_summary_ready_and_can_be_retried(tenant_engine, monkeypatch):
    client, app, as_user = tenant_engine
    app.state.providers["llm"] = ScriptedModel(task_delay_s=1.0)
    monkeypatch.setattr(task_service, "TASKS_TIMEOUT_S", 0.2)
    alice = as_user("alice")
    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        async with events.subscribe() as sub:
            mid = await recorded_scribe(app.state.store.pool)
            await client.post(f"/v1/meetings/{mid}/stop", headers=alice)
            await finish([finalize.schedule_finalize(app, mid)])
            assert ("tasks", "failed", "model_timeout") in pipeline_events(sub)

            now = await steps(client, alice, mid)
            assert now == {"transcript": ("ready", None), "summary": ("ready", None),
                           "tasks": ("failed", "model_timeout")}
            assert (await client.get(f"/v1/meetings/{mid}/summary", headers=alice)).json()["status"] == "ready"

            # Retry through the existing route, same shape as before.
            res = await client.post(f"/v1/meetings/{mid}/tasks/generate", headers=alice)
            assert res.status_code == 200 and set(res.json()) == {"created", "skipped"}
            assert pipeline_events(sub) == [("tasks", "working", None), ("tasks", "ready", None)]
            assert (await steps(client, alice, mid))["tasks"] == ("ready", None)


async def test_regenerate_while_a_summary_is_written_attaches_to_that_run(tenant_engine):
    client, app, as_user = tenant_engine
    hold = threading.Event()
    model = ScriptedModel(hold=hold)
    app.state.providers["llm"] = model
    alice = as_user("alice")
    try:
        with tenancy.tenant_scope(tenancy.tenant_key("alice")):
            async with events.subscribe() as sub:
                mid = await recorded_scribe(app.state.store.pool)
                await client.post(f"/v1/meetings/{mid}/stop", headers=alice)
                run = finalize.schedule_finalize(app, mid)
                job = pipeline.running(mid).job_id
                await asyncio.to_thread(model.summary_started.wait, WAIT_S)
                pipeline_events(sub)

                for _ in range(2):          # every request while it writes attaches; none starts another run
                    res = await client.post(f"/v1/meetings/{mid}/summary/regenerate", headers=alice)
                    assert res.status_code == 202 and res.json() == {"jobId": job}
                assert pipeline_events(sub) == [("summary", "working", None)] * 2
                assert (await steps(client, alice, mid))["summary"] == ("working", None)

                hold.set()
                await finish([run])
                assert (await steps(client, alice, mid))["summary"] == ("ready", None)
                pipeline_events(sub)

                # With nothing running, a new run: queued, then written, then tasks again.
                res = await client.post(f"/v1/meetings/{mid}/summary/regenerate", headers=alice)
                assert res.status_code == 202 and res.json()["jobId"] != job
                await finish([pipeline.running(mid).task])
                assert pipeline_events(sub) == [("summary", "working", "queued"), ("summary", "working", None),
                                                ("summary", "ready", None), ("tasks", "working", None),
                                                ("tasks", "ready", None)]

                # A summary edited by hand is still refused without overwriteEdits, and shows why.
                async with app.state.store.pool.connection() as conn:
                    await conn.execute("UPDATE summaries SET edited = true WHERE meeting_id = %s", (mid,))
                res = await client.post(f"/v1/meetings/{mid}/summary/regenerate", headers=alice)
                assert res.status_code == 409 and res.json()["error"] == "summary_edited"
                assert pipeline.running(mid) is None
                assert (await steps(client, alice, mid))["summary"] == ("ready", "edited_by_hand")
    finally:
        hold.set()


async def test_nothing_said_skips_every_step_with_no_speech(tenant_engine):
    client, app, as_user = tenant_engine
    app.state.providers["llm"] = ScriptedModel()
    alice = as_user("alice")
    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        mid = await recorded_scribe(app.state.store.pool, text="")
        await client.post(f"/v1/meetings/{mid}/stop", headers=alice)
        await finish([finalize.schedule_finalize(app, mid)])
    assert await steps(client, alice, mid) == {"transcript": ("skipped", "no_speech"),
                                               "summary": ("skipped", "no_speech"),
                                               "tasks": ("skipped", "no_speech")}


async def test_a_run_a_restart_cut_off_reads_as_interrupted_and_an_import_as_not_generated(tenant_engine):
    client, app, as_user = tenant_engine
    alice = as_user("alice")
    pool = app.state.store.pool
    with tenancy.tenant_scope(tenancy.tenant_key("alice")):
        store = MeetingStore(pool)
        imported = str(uuid.uuid4())
        await store.create_meeting(meeting_id=imported, title="Imported", origin="text")
        await store.import_lines(imported, [{"startMs": 0, "endMs": 1000, "text": WORDS}])
        cut = str(uuid.uuid4())
        await store.create_meeting(meeting_id=cut, title="Cut off")
        await store.import_lines(cut, [{"startMs": 0, "endMs": 1000, "text": WORDS}])
        await store.set_summary_status(cut, "running")          # and no run in this process
        await pipeline.report(pool, cut, "tasks", pipeline.WORKING)
    assert await steps(client, alice, imported) == {"transcript": ("ready", None),
                                                    "summary": ("skipped", "not_generated"),
                                                    "tasks": ("skipped", "not_generated")}
    assert await steps(client, alice, cut) == {"transcript": ("ready", None), "summary": ("failed", "interrupted"),
                                               "tasks": ("failed", "interrupted")}


def test_a_task_failure_reason_is_a_code_never_the_errors_text():
    from app.local.errors import EngineError, ProviderUnavailableError

    assert task_service.failure_reason(asyncio.TimeoutError()) == "model_timeout"
    assert task_service.failure_reason(ProviderUnavailableError("x")) == "model_unavailable"
    assert task_service.failure_reason(EngineError("ai_unreadable", "secret words", 502)) == "unusable_output"
    assert task_service.failure_reason(RuntimeError("the transcript said this")) == "model_error"
    assert all(r == r.lower() and " " not in r for r in pipeline.REASONS)
