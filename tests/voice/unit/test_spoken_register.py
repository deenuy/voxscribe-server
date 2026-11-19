"""The spoken register and its reactions (PL66): the plain and natural setting, the reply
model's instruction, the deterministic rules on what it wrote, and pronunciation last."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.local import gate, llm_adapter, reply_chain
from app.local.config import Config
from app.local.voice import expression
from app.local.voice.adapters import tts_kokoro
from app.local.voice.adapters.delegate_ollama import OllamaDelegate
from app.local.voice.adapters.gate_policy import CorePolicyGate, resolve_mode, session_policy
from app.local.voice.adapters.store_postgres import PostgresSessionStore
from app.local.voice.delegate_lane import TALKER_DELEGATE_INSTRUCTION
from app.local.voice.expression import (
    NATURAL,
    PLAIN,
    PRONUNCIATIONS,
    REACTIONS,
    STOCK_LEADING,
    STOCK_OFFERS,
    ReplyShape,
    pronounce,
    resolve_expressiveness,
    shape_reply,
)
from app.local.voice.host import SessionBinding, SessionScope, VoiceConfig, VoiceHost
from tests.voice.unit.fakes import fake_ports
from tests.voice.unit.reminder_fakes import ZONE, FakeClock, FakeReminderStore
from tests.voice.unit.tool_fakes import ToolPool, load_profile, make_app

OWNER_QUESTION = "How are you? Can you hear me?"
OWNER_BOT_REPLY = ("I'm functioning properly, thank you for asking. Yes, I can hear you, and I'm "
                   "ready to assist you with any questions or topics you'd like to discuss.")
CONTRACTIONS = ("i'm", "it's", "you're", "don't", "can't", "that's", "i'll", "we're")
REGISTER_MARK = reply_chain.SPOKEN_REGISTER.splitlines()[0]
STOCK_WORDS = ("thank you for asking", "here to help", "ready to assist", "how can i help",
               "how can i assist", "anything else i can help", "great question")


@pytest.fixture(autouse=True)
def _behaviour_log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_BEHAVIOUR_LOG_DIR", tmp_path)


async def _shape(sentences, previous=None, **kw):
    async def source():
        for s in sentences:
            yield s
    shape = ReplyShape()
    out = [s async for s in shape_reply(source(), previous_opener=previous, shape=shape, **kw)]
    return out, shape


def _interjections(text: str) -> list[str]:
    names = []
    for sentence in reply_chain.split_sentences(text):
        names += expression.split_interjections(sentence).names
    return names


# ── the setting ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode, expected", [("personal", NATURAL), ("ideascribe", NATURAL),
                                            ("senior_care", PLAIN), ("classroom", PLAIN),
                                            ("unknown", PLAIN), (None, PLAIN)])
def test_the_default_comes_from_the_product_mode(mode, expected):
    assert resolve_expressiveness(None, mode) == expected
    assert resolve_expressiveness("", mode) == expected


def test_an_operator_setting_wins_over_the_mode():
    assert resolve_expressiveness("plain", "personal") == PLAIN
    assert resolve_expressiveness("natural", "senior_care") == NATURAL


@pytest.mark.parametrize("profile, expected", [("personal", NATURAL), ("senior_care", PLAIN),
                                               ("classroom", PLAIN)])
def test_the_voice_config_takes_the_mode_the_gate_runs_under(profile, expected):
    settings = Config()
    loaded = load_profile(profile)
    assert VoiceConfig.from_profile(loaded, settings=settings).expressiveness == expected
    # VOXSCRIBE_VOICE_MODE tightens a personal profile for the gate, and for the register too.
    legacy = Config(voice_mode="seniorcare")
    assert resolve_mode(load_profile("personal"), legacy) == "senior_care"
    assert VoiceConfig.from_profile(load_profile("personal"), settings=legacy).expressiveness \
        == PLAIN


def test_the_setting_is_read_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOXSCRIBE_VOICE_EXPRESSIVENESS", "plain")
    assert Config.from_env().voice_expressiveness == PLAIN
    monkeypatch.delenv("VOXSCRIBE_VOICE_EXPRESSIVENESS")
    assert Config.from_env().voice_expressiveness is None
    monkeypatch.setenv("VOXSCRIBE_VOICE_EXPRESSIVENESS", "chatty")
    with pytest.raises(SystemExit):
        Config.from_env()


# ── the instruction ──────────────────────────────────────────────────────────


def _settings():
    return SimpleNamespace(vox_timezone="UTC", vox_location="", vox_user_name="",
                           voice_mode="default", voice_reply_model="fake-talker")


def _persona(prompt: str = ""):
    return SimpleNamespace(assistant_name="Nova", persona_label="companion", system_prompt=prompt)


def test_plain_is_the_engine_prompt_unchanged():
    from app.local.context import current_context_header

    header = current_context_header(_settings())
    assert reply_chain.talker_system_prompt(_settings(), _persona()) == (
        f"{header} You are Nova, a helpful voice companion. Reply in 1-2 short sentences only. "
        "Be direct and conversational. No markdown, no lists, no headers. "
        f"Keep replies under 30 words. {TALKER_DELEGATE_INSTRUCTION}")
    assert reply_chain.talker_system_prompt(_settings(), _persona(), PLAIN) == \
        reply_chain.talker_system_prompt(_settings(), _persona())


def test_natural_keeps_what_the_prompt_must_carry():
    from app.local.context import current_context_header
    from app.local.voice.profile_store import owner_preferences_block

    owner = "Call me Sam."
    prompt = reply_chain.talker_system_prompt(_settings(), _persona(owner), NATURAL)
    assert prompt.startswith(current_context_header(_settings()))
    assert "You are Nova, a helpful voice companion." in prompt
    assert TALKER_DELEGATE_INSTRUCTION in prompt
    assert prompt.endswith(owner_preferences_block(owner))
    assert "under 30 words" in prompt                      # the length bound
    assert "say you're an AI" in prompt and "Never claim a body, a past or feelings" in prompt
    assert "Good: Doing well, thanks. And yes, I can hear you fine." in prompt
    assert OWNER_BOT_REPLY.split(".")[0] in prompt          # the bad example, as a bad example
    assert "—" not in prompt


async def test_natural_reaches_the_model_with_the_barge_in_note(monkeypatch):
    seen: list[str] = []

    async def stream(user_text, model, timeout_s, history=None, system_prompt=None):
        seen.append(system_prompt)
        yield "Sure."

    monkeypatch.setattr(reply_chain, "_sentence_stream", stream)
    app = SimpleNamespace(state=SimpleNamespace(settings=_settings()))
    src = reply_chain.talker_sentences(app, "s1", "hello", system_note="NOTE", persona=_persona(),
                                       expressiveness=NATURAL)
    assert [s async for s in src.sentences] == ["Sure."]
    assert seen[0].endswith(" NOTE") and REGISTER_MARK in seen[0]


# ── the rules ────────────────────────────────────────────────────────────────


async def test_the_owners_reply_loses_its_stock_phrases_and_keeps_its_answer():
    out, shape = await _shape(reply_chain.split_sentences(OWNER_BOT_REPLY))
    text = " ".join(out).lower()
    assert out == ["I'm functioning properly.", "Yes, I can hear you."]
    assert any(c in text.split() for c in CONTRACTIONS)
    assert not any(stock in text for stock in STOCK_WORDS)


async def test_at_most_one_interjection_and_only_at_the_start():
    out, shape = await _shape(["Ohhh wow, that's great!", "Oh, and um, also this.", "Ha!"])
    assert out == ["Oh, that's great!", "And um, also this."]
    assert shape.reaction == "oh" and shape.opener == "oh"
    assert _interjections(" ".join(out)) == ["oh"]


@pytest.mark.parametrize("raw", ["Mm-hmm, that works.", "Um, that works.", "Aww, that works.",
                                 "Ugh, that works.", "Mhm, that works."])
async def test_an_interjection_outside_the_list_is_removed(raw):
    out, shape = await _shape([raw])
    assert out == ["That works."] and shape.reaction is None


async def test_two_replies_running_never_open_the_same_way():
    first, a = await _shape(["Oh, nice.", "How far did you go?"])
    second, b = await _shape(["Oh, nice.", "Where did you go?"], previous=a.opener)
    assert first[0] == "Oh, nice." and second[0] == "Nice."
    third, c = await _shape(["Sure, I can do that."], previous="sure")
    assert third == ["I can do that."]
    fourth, d = await _shape(["Ha!", "Good one."], previous="ha")
    assert fourth == ["Good one."]


async def test_a_sentence_that_carries_the_answer_is_never_removed():
    out, _ = await _shape(["Sure."], previous="sure")                   # the whole answer
    assert out == ["Sure."]
    out, _ = await _shape(["Uh-huh."])
    assert out == ["Uh-huh."]
    out, _ = await _shape(["No, I'm an AI."])
    assert out == ["No, I'm an AI."]
    out, _ = await _shape(["How can I assist you today?"])               # nothing else came
    assert out == ["How can I assist you today?"]


async def test_stock_sentences_lead_and_offers_close_and_both_go():
    out, shape = await _shape(["Great question!", "The capital is Canberra.",
                               "Is there anything else I can help you with?"])
    assert out == ["The capital is Canberra."]
    assert shape.removed == ["stock=great_question", "stock=anything_else"]
    out, _ = await _shape(["Hi!", "How can I help you today?"])
    assert out == ["Hi!"]


def test_the_stock_lists_are_short():
    assert len(STOCK_LEADING) <= 8 and len(STOCK_OFFERS) <= 6


async def test_the_thinkers_summary_gets_no_reaction():
    out, shape = await _shape(["Oh, it was sunny.", "Highs of ten."], reaction_allowed=False)
    assert out == ["It was sunny.", "Highs of ten."] and shape.reaction is None


async def test_the_first_sentence_is_not_held_for_the_rest():
    second = asyncio.Event()

    async def source():
        yield "Oh, nice."
        await second.wait()
        yield "How far?"

    stream = shape_reply(source(), previous_opener=None, shape=ReplyShape())
    first = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert first == "Oh, nice."
    second.set()
    assert [s async for s in stream] == ["How far?"]


# ── through the delegate and the gate, in a session ─────────────────────────


class Conversation:
    """The production delegate and gate for a shipped profile, in a session whose register is
    set; the reply model is scripted."""

    def __init__(self, monkeypatch, expressiveness: str, mode: str = "personal") -> None:
        self.pool = ToolPool()
        self.app = make_app(self.pool, profile=load_profile(mode), vox_timezone=ZONE)
        self.model: list[str] = []
        self.prompts: list[str] = []

        async def stream_reply(user_text, *, model=None, system_prompt=None, history=None):
            self.prompts.append(system_prompt)
            for token in (self.model.pop(0) if self.model else "Okay.").split(" "):
                yield token + " "

        monkeypatch.setattr(llm_adapter, "stream_reply", stream_reply)
        self.delegate = OllamaDelegate(self.app, reminder_store=FakeReminderStore(),
                                       clock=FakeClock())
        self.ports = fake_ports(gate=CorePolicyGate(session_policy(self.app)),
                                store=PostgresSessionStore(self.app), delegate=self.delegate)
        self.host = VoiceHost(VoiceConfig(expressiveness=expressiveness), ports=self.ports)
        self.scope = SessionScope()
        self.expressiveness = expressiveness
        self.turn = 0
        self.history: list[dict] = []

    async def say(self, text: str, model_reply: str | None = None) -> str:
        if model_reply is not None:
            self.model.append(model_reply)
        self.turn += 1
        reply = await self.host.reply_turn("s1", self.turn, text, turn_end_ms=0,
                                           history=list(self.history) or None)
        self.history += [{"role": "user", "content": text}]
        if reply:
            self.history += [{"role": "assistant", "content": reply}]
        return reply

    def __enter__(self):
        self._cm = self.scope.session(SessionBinding())
        self.session = self._cm.__enter__()
        self.session.expressiveness = self.expressiveness
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


async def test_natural_how_are_you_has_a_contraction_and_no_stock_phrase(monkeypatch):
    with Conversation(monkeypatch, NATURAL) as c:
        reply = await c.say(OWNER_QUESTION, OWNER_BOT_REPLY)
        assert REGISTER_MARK in c.prompts[0]
    lowered = reply.lower()
    assert any(w in lowered.replace(".", " ").split() for w in CONTRACTIONS)
    assert not any(stock in lowered for stock in STOCK_WORDS)
    assert c.ports.talker.fixed_phrases == []                # no immediate "hm" in natural


async def test_plain_replies_are_as_before(monkeypatch):
    with Conversation(monkeypatch, PLAIN) as c:
        reply = await c.say(OWNER_QUESTION, OWNER_BOT_REPLY)
        assert REGISTER_MARK not in c.prompts[0]
    assert reply == OWNER_BOT_REPLY
    assert c.ports.talker.fixed_phrases == [reply_chain.THINKING_PHRASE]


async def test_two_replies_running_in_a_session_never_open_the_same_way(monkeypatch):
    with Conversation(monkeypatch, NATURAL) as c:
        first = await c.say("I went for a run.", "Oh, nice. How far?")
        second = await c.say("Five kilometres.", "Oh, nice. That's a good distance.")
    assert first.startswith("Oh, ") and not second.startswith("Oh")


async def test_a_read_back_and_its_confirmation_carry_no_reaction(monkeypatch):
    with Conversation(monkeypatch, NATURAL) as c:
        readback = await c.say("remind me in three minutes to stretch", "Oh, sure thing.")
        confirmed = await c.say("yes", "Oh, wow, done!")
    assert readback == "In three minutes, stretch. Shall I set it?"
    assert confirmed != "Oh, wow, done!" and c.model == ["Oh, sure thing.", "Oh, wow, done!"]
    assert _interjections(readback) == [] and _interjections(confirmed) == []


async def test_a_refusal_carries_no_reaction(monkeypatch):
    with Conversation(monkeypatch, NATURAL, mode="senior_care") as c:
        reply = await c.say("how many pills should I take", "Oh, take 2 tablets of 500 mg.")
    assert reply == gate.SENIORCARE_ESCALATION_PHRASE or reply == \
        c.ports.gate.fail_closed_phrase
    assert _interjections(reply) == []


# ── pronunciation, last ──────────────────────────────────────────────────────


@pytest.mark.parametrize("written, spoken", [
    ("Mm-hmm, right.", "Um-hum, right."), ("Mhm.", "Um-hum."), ("Mmhmm.", "Um-hum."),
    ("Mm, good.", "Hmm, good."), ("Mmm.", "Hmm."), ("Hm.", "Hmm."), ("Hmmmm.", "Hmm."),
    ("Aww, sweet.", "Aw, sweet."), ("Wowww!", "Wow!"),
    ("It is 5 mm long.", "It is 5 mm long."), ("Hmm.", "Hmm."), ("Uh-huh.", "Uh-huh."),
])
def test_the_pronunciation_table(written, spoken):
    assert pronounce(written) == spoken


async def test_mm_hmm_is_never_sent_to_the_voice_as_written_and_the_record_keeps_it(monkeypatch):
    voiced: list[str] = []

    async def synthesize(text, **kw):
        voiced.append(text)
        return np.zeros(240, dtype=np.float32)

    monkeypatch.setattr(tts_kokoro.tts_engine, "synthesize", synthesize)

    def pronounce_in_session(text):
        from app.local.voice.host import current_session

        session = current_session()
        return pronounce(text) if session and session.expressiveness == NATURAL else text

    for expressiveness, said in ((NATURAL, "Um-hum"), (PLAIN, "Mm-hmm")):
        voiced.clear()
        with Conversation(monkeypatch, expressiveness) as c:
            c.ports.talker = tts_kokoro.KokoroTalker(pronounce=pronounce_in_session)
            reply = await c.say("what did he say", "He said Mm-hmm and left.")
        assert reply == "He said Mm-hmm and left."            # captions and the record
        assert voiced and all(said in v for v in voiced if "said" in v)
        assert not any("Mm-hmm" in v for v in voiced) or expressiveness == PLAIN


_PHONEMIZE = """
import os, sys
from kokoro_onnx.tokenizer import Tokenizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper
t = Tokenizer()
if os.environ.get("PHONEMIZER_ESPEAK_DATA_PATH"):
    EspeakWrapper.set_data_path(os.environ["PHONEMIZER_ESPEAK_DATA_PATH"])
for line in sys.stdin.read().splitlines():
    print(t.phonemize(line, "en-us"))
"""


def test_each_replacement_reads_as_measured_on_the_phonemizer():
    """Runs Kokoro's phonemizer (no model load) in a child process; skipped where it cannot run
    (espeak-ng's data path must be short; PHONEMIZER_ESPEAK_DATA_PATH can point at a copy)."""
    words = [p.replacement.replace("\\1", "h" if p.name == "hm" else
                                   "a" if p.name == "aww" else "w") for p in PRONUNCIATIONS]
    try:
        out = subprocess.run([sys.executable, "-c", _PHONEMIZE], input="\n".join(words),
                             capture_output=True, text=True, timeout=60, env=dict(os.environ))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"phonemizer not runnable: {type(exc).__name__}")
    lines = out.stdout.splitlines()
    if out.returncode != 0 or len(lines) != len(words):
        pytest.skip("phonemizer not runnable here")
    for entry, phonemes in zip(PRONUNCIATIONS, lines, strict=True):
        assert phonemes == entry.after, entry.name


def test_reactions_are_only_words_the_voice_pronounces():
    measured = {"hmm", "ah", "aha", "oh", "ooh", "huh", "ha", "haha", "whoa", "wow", "oops",
                "phew", "yeah", "yep", "uh_huh", "um", "uh"}
    assert set(REACTIONS) <= measured
