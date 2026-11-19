"""T14 delegate lane: the task registry, the talker and thinker split, spoken cancel and change.

A1  during a delegated task an unrelated short question is answered within the L1 bar
A2  cancel and change leave the lane in the right state, and a cancelled result is never spoken
A3  a stale result is offered, not read out
A5  every tool row carries the task id of the delegate task that made the call

No microphone, reply model, search engine or database: the reply model and the search tool are
scripted, the pool records what is written, and time is a fake clock the fakes advance. The
stored privacy mode is "connectors" and the profile personal, so the network decision (PL41,
tests/voice/unit/test_network_decision.py) lets the search run.
"""
from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.local import llm_adapter, reply_chain
from app.local.tools import web_search
from app.local.voice import delegate_lane as dl
from app.local.voice.adapters import delegate_ollama
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.delegate_lane import (
    STALE_RESULT_S,
    DelegateLane,
    IntentKind,
    TaskStatus,
)
from app.local.voice.host import StreamClock, VoiceConfig, VoiceHost

from .fakes import FakeTalker, fake_ports

L1_FIRST_AUDIO_BAR_MS = 800      # bar L1: turn end to first audio (ROOT-CAUSE, FITGAP G5)
MODEL_FIRST_TOKEN_MS = 200       # modelled cost of the reply model's first sentence
SYNTH_MS = 500                   # modelled cost of synthesising one sentence (FITGAP G5)
SEARCH_MS = 10_000               # modelled cost of the thinker's search
TURN_TIMEOUT_S = 5               # a turn that waits on the thinker fails here instead of hanging
MS_PER_S = 1000

SEARCH_REQUEST = "search the web for the best standing desks"
SEARCH_TOPIC = "the best standing desks"
SEARCH_RESULT = "The Uplift V2 is the top pick. It costs about six hundred dollars."
TALKER_ANSWER = "Four."

TOOL_ENTRY_POINTS = ("web_search", "search_answer", "tool_runtime", "runtime.call",
                     "app.local.tools")

UNRELATED_QUESTIONS = (
    "What time is it?", "What day is it?", "What's the date?", "What is 2+2?", "How are you?",
    "Tell me a joke.", "What is the capital of France?", "Spell banana.",
    "How many days are in a week?", "Who wrote Hamlet?",
)


class FakeClock:
    def __init__(self) -> None:
        self.ms = 0

    def seconds(self) -> float:
        return self.ms / MS_PER_S

    def advance_s(self, seconds: float) -> None:
        self.ms += int(seconds * MS_PER_S)


class _Rows:
    def __init__(self, rows) -> None:
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class RecordingPool:
    """Stands in for the connection pool; keeps every statement and its parameters, and answers
    the stored privacy mode."""

    def __init__(self, privacy: str = "connectors") -> None:
        self.statements: list[tuple[str, tuple]] = []
        self.privacy = privacy

    @asynccontextmanager
    async def connection(self):
        yield self

    async def commit(self) -> None:
        pass

    async def execute(self, sql: str, params=()):
        self.statements.append((sql, tuple(params)))
        if "FROM settings" in sql:
            return _Rows([(self.privacy,)])
        return _Rows([])

    def rows(self, table: str) -> list[tuple]:
        return [p for sql, p in self.statements if sql.startswith(f"INSERT INTO {table}")]

    def named_rows(self, table: str) -> list[dict]:
        """Each insert into `table` as {column: value}, from the statement's column list."""
        out = []
        for sql, params in self.statements:
            if sql.startswith(f"INSERT INTO {table}"):
                cols = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
                out.append(dict(zip(cols, params, strict=True)))
        return out


class TimedTalker(FakeTalker):
    """Synthesis costs SYNTH_MS on the fake clock; a cached fixed phrase costs nothing."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self._clock = clock

    async def synthesize(self, text: str):
        self._clock.ms += SYNTH_MS
        return await super().synthesize(text)


class Thinker:
    """A scripted thinker: holds each task until released, then returns the result."""

    def __init__(self, clock: FakeClock | None = None, result: str = SEARCH_RESULT) -> None:
        self.clock, self.result = clock, result
        self.requests: list[str] = []
        self.cancelled: list[str] = []
        self.release = asyncio.Event()

    async def __call__(self, task) -> str:
        self.requests.append(task.request)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.append(task.request)
            raise
        if self.clock is not None:
            self.clock.ms += SEARCH_MS
        return self.result


def _settings():
    return SimpleNamespace(vox_timezone="UTC", vox_location="", vox_user_name="",
                           voice_mode="default", voice_reply_model="fake-talker")


def _app(pool: RecordingPool | None = None):
    return SimpleNamespace(state=SimpleNamespace(
        settings=_settings(), store=SimpleNamespace(pool=pool or RecordingPool())))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def talker_calls(monkeypatch, clock) -> list[str]:
    """The reply model, scripted. Each call is recorded and costs MODEL_FIRST_TOKEN_MS."""
    calls: list[str] = []

    async def stream_reply(user_text, *, model=None, system_prompt=None, history=None):
        calls.append(user_text)
        clock.ms += MODEL_FIRST_TOKEN_MS
        yield TALKER_ANSWER

    monkeypatch.setattr(llm_adapter, "stream_reply", stream_reply)
    return calls


@pytest.fixture
def thinker(clock) -> Thinker:
    return Thinker(clock)


@pytest.fixture
def delegate(clock, thinker, talker_calls) -> OllamaDelegate:
    return OllamaDelegate(_app(), lane=DelegateLane(runner=thinker, clock=clock.seconds))


async def _say(delegate: OllamaDelegate, text: str, turn_id: str = "t1") -> list[str]:
    stream = await delegate.reply(text, "s1", turn_id=turn_id)
    return [s async for s in stream.sentences]


async def _running(lane: DelegateLane, task) -> None:
    for _ in range(10):
        if task.status is TaskStatus.RUNNING:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"thinker did not start: {task.status}")


async def _finish(delegate: OllamaDelegate, thinker: Thinker, task) -> None:
    thinker.release.set()
    await delegate.lane.wait(task.task_id)
    assert task.status is TaskStatus.DONE


# ── the registry ─────────────────────────────────────────────────────────────


def test_submit_and_complete(clock):
    lane = DelegateLane(clock=clock.seconds)
    task = lane.submit(SEARCH_REQUEST)
    assert task.status is TaskStatus.QUEUED and task.result is None
    assert lane.get_pending() == [] and lane.active() == [task]

    assert lane.start(task.task_id)
    assert task.status is TaskStatus.RUNNING
    assert lane.complete(task.task_id, SEARCH_RESULT)
    assert task.status is TaskStatus.DONE and task.result == SEARCH_RESULT
    assert lane.get_pending() == [task] and lane.active() == []

    clock.advance_s(1)
    lane.mark_delivered(task.task_id)
    assert task.status is TaskStatus.DELIVERED and task.delivered_at == clock.seconds()
    assert lane.get_pending() == [] and lane.latest_open() is None


async def test_thinker_runs_in_the_background_and_reports_to_the_lane(thinker):
    lane = DelegateLane(runner=thinker)
    task = lane.submit(SEARCH_REQUEST)
    assert task.status is TaskStatus.QUEUED          # submit returns before the thinker runs
    await _running(lane, task)
    thinker.release.set()
    await lane.wait(task.task_id)
    assert task.status is TaskStatus.DONE and task.result == SEARCH_RESULT


async def test_cancel_running(thinker):
    lane = DelegateLane(runner=thinker)
    task = lane.submit(SEARCH_REQUEST)
    await _running(lane, task)

    assert lane.cancel(task.task_id) is True
    assert task.status is TaskStatus.CANCELLED
    await lane.wait(task.task_id)
    assert thinker.cancelled == [SEARCH_REQUEST]      # the thinker itself was stopped
    assert lane.complete(task.task_id, SEARCH_RESULT) is False
    assert task.status is TaskStatus.CANCELLED and task.result is None
    assert lane.get_pending() == [] and lane.cancel(task.task_id) is False


async def test_thinker_past_its_timeout_is_done_with_the_failure_line():
    async def never(_task) -> str:
        await asyncio.Event().wait()

    lane = DelegateLane(runner=never, timeout_s=0.01)
    task = lane.submit(SEARCH_REQUEST)
    for _ in range(100):
        if task.status is TaskStatus.DONE:
            break
        await asyncio.sleep(0.01)
    assert task.status is TaskStatus.DONE
    assert (task.result, task.error) == (dl.THINKER_FAILED_PHRASE, "timeout")


def test_finished_tasks_are_forgotten_past_the_cap_and_open_ones_never():
    lane = DelegateLane()
    waiting = lane.submit(SEARCH_REQUEST)
    lane.complete(waiting.task_id, SEARCH_RESULT)
    for _ in range(dl.FINISHED_TASKS_KEPT * 2):
        lane.cancel(lane.submit(SEARCH_REQUEST).task_id)
    latest = lane.submit(SEARCH_REQUEST)
    assert lane.get_pending() == [waiting] and lane.active() == [latest]
    assert len(lane._tasks) == dl.FINISHED_TASKS_KEPT + 2


def test_lane_is_safe_to_use_from_many_threads():
    import threading

    lane = DelegateLane()
    per_thread = 50

    def work() -> None:
        for _ in range(per_thread):
            task = lane.submit(SEARCH_REQUEST)
            lane.start(task.task_id)
            lane.complete(task.task_id, SEARCH_RESULT)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pending = lane.get_pending()
    assert len(pending) == per_thread * len(threads)
    assert len({t.task_id for t in pending}) == len(pending)


# ── routing: talker or thinker ───────────────────────────────────────────────


async def test_short_question_no_delegation(delegate, talker_calls):
    assert await _say(delegate, "What is 2+2?") == [TALKER_ANSWER]
    assert talker_calls == ["What is 2+2?"]
    assert delegate.lane.latest_open() is None


async def test_web_search_delegated(delegate, thinker, talker_calls):
    assert await _say(delegate, "search the web for standing desks") == [dl.DELEGATING_PHRASE]
    task = delegate.lane.latest_open()
    assert task is not None and task.request == "search the web for standing desks"
    assert dl.request_topic(task.request) == "standing desks"
    await _running(delegate.lane, task)
    assert thinker.requests == [task.request]
    assert talker_calls == []                         # the reply model was not asked


@pytest.mark.parametrize("text", [
    "Look up the population of Canada", "What's the latest on the election?",
    "Any news about the launch?", "Find me a good ramen place",
    "What's the weather in Toronto?",                 # a pre-router search route, no signal word
])
async def test_each_signal_goes_to_the_thinker(delegate, talker_calls, text):
    assert await _say(delegate, text) == [dl.DELEGATING_PHRASE]
    assert delegate.lane.latest_open().request == text and talker_calls == []


async def test_clock_answer_wins_over_a_signal_word(delegate, talker_calls):
    sentences = await _say(delegate, "Look up what time is it")
    assert len(sentences) == 1 and sentences[0].startswith("It's ")
    assert delegate.lane.latest_open() is None and talker_calls == []


async def test_talker_marker_submits_a_task_and_is_never_spoken(delegate, monkeypatch):
    async def stream_reply(user_text, **_kw):
        for token in ("[DELEGATE: ", "who won the cup? ", "this year] ", "Let me guess. "):
            yield token

    monkeypatch.setattr(llm_adapter, "stream_reply", stream_reply)
    assert await _say(delegate, "Who took the cup home?") == [dl.DELEGATING_PHRASE]
    assert delegate.lane.latest_open().request == "who won the cup? this year"


def test_the_talker_prompt_names_the_marker():
    assert dl.DELEGATE_MARKER_OPEN in reply_chain.talker_system_prompt(_settings())
    assert dl.find_delegate_marker("[delegate: desks ]") == "desks"


# ── spoken cancel and change (A2) ────────────────────────────────────────────


@pytest.mark.parametrize("phrase", [
    "cancel that", "Never mind.", "Forget it", "stop searching", "Actually, never mind.",
    "Oh, cancel that please", "nevermind", "Okay forget it.", "Please stop searching now",
    "No, never mind",
])
async def test_spoken_cancel_detected(delegate, thinker, talker_calls, phrase):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _running(delegate.lane, task)

    assert await _say(delegate, phrase, "t2") == [dl.CANCELLED_PHRASE]
    assert task.status is TaskStatus.CANCELLED
    assert talker_calls == []                         # routed to cancel, not the reply model

    # Nothing is spoken later: the thinker is stopped and the next turn has nothing to deliver.
    thinker.release.set()
    await delegate.lane.wait(task.task_id)
    assert await _say(delegate, "What is 2+2?", "t3") == [TALKER_ANSWER]
    assert task.status is TaskStatus.CANCELLED and delegate.lane.get_pending() == []


@pytest.mark.parametrize("phrase,revised", [
    ("search for office chairs instead", "search for office chairs"),
    ("No, search for office chairs instead.", "No, search for office chairs"),
    ("Actually, search for office chairs", "search for office chairs"),
    ("actually make it office chairs", "make it office chairs"),
    ("Look up office chairs instead", "Look up office chairs"),
])
async def test_spoken_change_detected(delegate, thinker, talker_calls, phrase, revised):
    await _say(delegate, SEARCH_REQUEST)
    prior = delegate.lane.latest_open()
    await _running(delegate.lane, prior)

    assert await _say(delegate, phrase, "t2") == [dl.CHANGED_PHRASE]
    assert prior.status is TaskStatus.CANCELLED
    new = delegate.lane.latest_open()
    assert new is not prior and new.request == revised
    assert delegate.lane.active() == [new] and talker_calls == []

    # Only the new result is spoken.
    await _finish(delegate, thinker, new)
    spoken = await _say(delegate, "What did you find?", "t3")
    assert spoken[0] == dl.DELIVERY_LEAD.format(topic=dl.request_topic(revised))
    assert spoken[1:] == reply_chain.split_sentences(SEARCH_RESULT)
    assert prior.status is TaskStatus.CANCELLED and new.status is TaskStatus.DELIVERED


async def test_cancel_and_change_words_with_no_task_are_ordinary_turns(delegate, talker_calls):
    assert await _say(delegate, "Never mind.") == [TALKER_ANSWER]
    assert await _say(delegate, "Actually, what is 2+2?") == [TALKER_ANSWER]
    assert len(talker_calls) == 2 and delegate.lane.latest_open() is None


@pytest.mark.parametrize("text,kind", [
    ("cancel that", IntentKind.CANCEL), ("search for X instead", IntentKind.CHANGE),
    ("actually, the red one", IntentKind.CHANGE), ("What did you find?", IntentKind.RESULTS),
    ("What is 2+2?", None), ("I'd rather walk instead", None), ("", None),
])
def test_detect_intent(text, kind):
    intent = dl.detect_intent(text)
    assert (intent.kind if intent else None) is kind


@pytest.mark.parametrize("text,yes", [
    ("Yes please.", True), ("Sure, go ahead", True), ("Run it again", True), ("No thanks.", False),
    ("Okay, what time is it over there now?", False), ("", False),
])
def test_is_affirmative(text, yes):
    assert dl.is_affirmative(text) is yes


# ── delivery, and the stale result rule (A3) ─────────────────────────────────


async def test_result_is_delivered_after_the_next_turns_own_reply(delegate, thinker):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _finish(delegate, thinker, task)

    spoken = await _say(delegate, "What is 2+2?", "t2")
    assert spoken == [TALKER_ANSWER, dl.DELIVERY_LEAD.format(topic=SEARCH_TOPIC),
                      *reply_chain.split_sentences(SEARCH_RESULT)]
    assert task.status is TaskStatus.DELIVERED
    assert await _say(delegate, "What is 2+2?", "t3") == [TALKER_ANSWER]     # spoken once


async def test_asking_for_results_while_the_thinker_runs(delegate, thinker, talker_calls):
    assert await _say(delegate, "What did you find?") == [dl.NOTHING_WAITING_PHRASE]
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _running(delegate.lane, task)
    assert await _say(delegate, "What did you find?", "t2") == [
        dl.STILL_WORKING_PHRASE.format(topic=SEARCH_TOPIC)]
    assert task.status is TaskStatus.RUNNING and talker_calls == []


async def test_stale_result(delegate, thinker, clock):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _finish(delegate, thinker, task)
    clock.advance_s(STALE_RESULT_S + 1)
    assert delegate.lane.is_stale(task)

    spoken = await _say(delegate, "What is 2+2?", "t2")
    offer = dl.STALE_OFFER.format(topic=SEARCH_TOPIC)
    assert spoken == [TALKER_ANSWER, offer]           # offered in one sentence
    assert not any(s in " ".join(spoken) for s in reply_chain.split_sentences(SEARCH_RESULT))
    assert task.status is TaskStatus.DELIVERED

    # Declined: dropped. The reply model answers, nothing is run or read out.
    assert await _say(delegate, "No thanks.", "t3") == [TALKER_ANSWER]
    assert delegate.lane.latest_open() is None


async def test_stale_result_is_run_again_on_a_yes(delegate, thinker, clock):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _finish(delegate, thinker, task)
    clock.advance_s(STALE_RESULT_S + 1)
    await _say(delegate, "What did you find?", "t2")

    assert await _say(delegate, "Yes please.", "t3") == [dl.RERUN_PHRASE]
    again = delegate.lane.latest_open()
    assert again is not task and again.request == SEARCH_REQUEST
    assert again.status in dl.ACTIVE_STATUSES


async def test_result_just_inside_the_stale_limit_is_read_out(delegate, thinker, clock):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _finish(delegate, thinker, task)
    clock.advance_s(STALE_RESULT_S - 1)
    spoken = await _say(delegate, "What did you find?", "t2")
    assert spoken[1:] == reply_chain.split_sentences(SEARCH_RESULT)


async def test_a_reply_the_gate_stops_leaves_the_result_waiting(delegate, thinker):
    await _say(delegate, SEARCH_REQUEST)
    task = delegate.lane.latest_open()
    await _finish(delegate, thinker, task)

    stream = await delegate.reply("What is 2+2?", "s1", turn_id="t2")
    async for _sentence in stream.sentences:
        break                                         # what the reply stage does on a refusal
    assert task.status is TaskStatus.DONE and delegate.lane.get_pending() == [task]


# ── A1: the talker answers while the thinker works ───────────────────────────


async def test_unrelated_question_answered_during_delegation(clock, thinker, talker_calls):
    lane = DelegateLane(runner=thinker, clock=clock.seconds)
    ports = fake_ports(delegate=OllamaDelegate(_app(), lane=lane), talker=TimedTalker(clock),
                       clock=StreamClock(mono=lambda: clock.ms))
    host = VoiceHost(VoiceConfig(), ports=ports)

    async def turn(n: int, text: str) -> tuple[str, int]:
        turn_end = clock.ms
        reply = await asyncio.wait_for(
            host.reply_turn("s1", n, text, turn_end_ms=turn_end), TURN_TIMEOUT_S)
        return reply, ports.store.turns[-1][1].first_audio_ms - turn_end

    reply, _ = await turn(1, SEARCH_REQUEST)
    assert reply == dl.DELEGATING_PHRASE
    task = lane.latest_open()
    await _running(lane, task)

    latencies: list[int] = []
    for n, question in enumerate(UNRELATED_QUESTIONS, start=2):
        reply, first_audio = await turn(n, question)
        assert reply and dl.DELEGATING_PHRASE not in reply, question
        assert task.status is TaskStatus.RUNNING, question      # the search is still going
        latencies.append(first_audio)

    within = [ms for ms in latencies if ms <= L1_FIRST_AUDIO_BAR_MS]
    print(f"A1 unit: {len(within)} of {len(latencies)} within {L1_FIRST_AUDIO_BAR_MS} ms; "
          f"first audio max {max(latencies)} ms, min {min(latencies)} ms (fake clock, "
          f"search held for {SEARCH_MS} ms)")
    assert len(within) == len(UNRELATED_QUESTIONS)
    assert max(latencies) < SEARCH_MS                # no turn paid for the search

    # The search answer arrives when it is ready, on the next turn.
    await _finish(ports.delegate, thinker, task)
    reply, _ = await turn(len(UNRELATED_QUESTIONS) + 2, "What did you find?")
    assert SEARCH_RESULT in reply and task.status is TaskStatus.DELIVERED


# ── A5: the talker has no tools; every tool row carries a task id ────────────


async def test_every_tool_row_carries_the_task_id(monkeypatch, talker_calls):
    pool = RecordingPool()
    queries: list[str] = []

    async def search(query: str) -> str:
        queries.append(query)
        return "[SEARCH RESULTS]\n1. Desk: a desk (http://example.test)\n[END SEARCH RESULTS]"

    monkeypatch.setattr(web_search, "search", search)
    delegate = OllamaDelegate(_app(pool))             # the production lane and thinker

    await _say(delegate, SEARCH_REQUEST, "turn-1")
    task = delegate.lane.latest_open()
    await delegate.lane.wait(task.task_id)
    await delegate.drain()

    assert task.status is TaskStatus.DONE and task.result == TALKER_ANSWER
    assert queries == [SEARCH_TOPIC]
    (tool_row,) = pool.named_rows("tool_calls")
    assert tool_row["task_id"] == task.task_id
    assert (tool_row["turn_id"], tool_row["meeting_id"]) == ("turn-1", "s1")
    assert tool_row["tool"] == "web.search" and tool_row["status"] == "executed"
    assert tool_row["tool_version"] and tool_row["duration_ms"] >= 0
    assert SEARCH_TOPIC not in str(tool_row)           # the query is kept as a digest only

    task_rows = pool.rows("delegate_tasks")
    assert {r[0] for r in task_rows} == {task.task_id}
    assert {r[-1] for r in task_rows} == {task.task_id}          # delegate_tasks.task_id
    assert [r[3] for r in task_rows] == ["queued", "running", "done"]

    await _say(delegate, "What did you find?", "turn-2")
    await delegate.drain()
    assert pool.rows("delegate_tasks")[-1][3] == "delivered"


async def test_search_that_is_cancelled_still_writes_its_tool_row(monkeypatch, talker_calls):
    pool = RecordingPool()
    started = asyncio.Event()

    async def search(_query: str) -> str:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(web_search, "search", search)
    delegate = OllamaDelegate(_app(pool))
    await _say(delegate, SEARCH_REQUEST, "turn-1")
    task = delegate.lane.latest_open()
    await asyncio.wait_for(started.wait(), TURN_TIMEOUT_S)

    assert await _say(delegate, "cancel that", "turn-2") == [dl.CANCELLED_PHRASE]
    await delegate.lane.wait(task.task_id)
    await delegate.drain()
    (tool_row,) = pool.named_rows("tool_calls")
    assert tool_row["task_id"] == task.task_id and tool_row["status"] == "cancelled"
    assert pool.rows("delegate_tasks")[-1][3] == "cancelled"


def test_the_talker_path_cannot_reach_a_tool():
    """Inspection: the reply chain reaches tools in one function, and only the thinker calls it.
    The delegate reaches tools only through the tool runtime (its deterministic reminder routes
    and the registry check before the lane), never the search module itself."""
    chain = inspect.getsource(reply_chain)
    assert chain.count("app.local.tools") == 1
    assert "app.local.tools" in inspect.getsource(reply_chain.search_answer)
    adapter = inspect.getsource(delegate_ollama)
    assert "web_search" not in adapter and "tools.web_search" not in adapter
    assert adapter.count("search_answer") == 1
    assert "search_answer" in inspect.getsource(OllamaDelegate._think)
    for talker_fn in (reply_chain.talker_sentences, reply_chain._sentence_stream,
                      reply_chain._sentence_stream_from_tokens, llm_adapter.stream_reply):
        source = inspect.getsource(talker_fn)
        assert not any(name in source for name in TOOL_ENTRY_POINTS), talker_fn.__name__
