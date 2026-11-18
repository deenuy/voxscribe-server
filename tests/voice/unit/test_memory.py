"""Unit tests for MemoryStore and the [Memory] block injection (T19).

No TTS, no LLM, no I/O.  Covers:
  - Each of the 7 extraction patterns
  - forget() marks a fact so it no longer appears in retrieve()
  - update() replaces an existing fact
  - Retrieved memories appear in the [Memory] block in the system prompt
  - Empty store produces no [Memory] block
  - A fact extracted on turn 2 is retrieved correctly on turn 8
  - Host passes the memory block to the delegate as the system prefix
"""
from __future__ import annotations

import pytest

from app.local.voice.memory import (
    MEMORY_BLOCK_HEADER,
    Memory,
    MemoryStore,
    memory_block,
)


# ---------------------------------------------------------------------------
# Individual extraction patterns
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extract_name_my_name_is(self):
        store = MemoryStore()
        mems = store.extract("My name is Alex.", turn_n=1)
        assert any(m.key == "user_name" and m.value == "Alex" for m in mems)

    def test_extract_name_im(self):
        store = MemoryStore()
        mems = store.extract("I'm Sam and I live here.", turn_n=1)
        assert any(m.key == "user_name" and m.value == "Sam" for m in mems)

    def test_extract_location_live_in(self):
        store = MemoryStore()
        mems = store.extract("I live in Toronto.", turn_n=1)
        assert any(m.key == "user_location" and "Toronto" in m.value for m in mems)

    def test_extract_location_im_from(self):
        store = MemoryStore()
        mems = store.extract("I'm from Vancouver.", turn_n=1)
        assert any(m.key == "user_location" and "Vancouver" in m.value for m in mems)

    def test_extract_role_work_as(self):
        store = MemoryStore()
        mems = store.extract("I work as a software engineer.", turn_n=1)
        assert any(m.key == "user_role" and "engineer" in m.value.lower() for m in mems)

    def test_extract_role_im_a(self):
        store = MemoryStore()
        mems = store.extract("I'm a teacher.", turn_n=1)
        assert any(m.key == "user_role" and "teacher" in m.value.lower() for m in mems)

    def test_extract_preference(self):
        store = MemoryStore()
        mems = store.extract("I prefer shorter answers.", turn_n=1)
        assert any(m.key == "user_preference" and "shorter" in m.value.lower() for m in mems)

    def test_extract_remember_that(self):
        store = MemoryStore()
        mems = store.extract("Remember that my password hint is blue.", turn_n=1)
        assert any(m.key == "user_note" and "blue" in m.value.lower() for m in mems)


# ---------------------------------------------------------------------------
# Forget and update
# ---------------------------------------------------------------------------

class TestForgetUpdate:
    def test_forget_removes_from_retrieve(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        assert any(m.key == "user_name" for m in store.retrieve())
        result = store.forget("user_name")
        assert result is True
        assert not any(m.key == "user_name" for m in store.retrieve())

    def test_forget_nonexistent_returns_false(self):
        store = MemoryStore()
        assert store.forget("user_name") is False

    def test_forget_via_spoken_command(self):
        """'forget my name' should mark user_name as forgotten."""
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.extract("forget my name", turn_n=2)
        assert not any(m.key == "user_name" for m in store.retrieve())

    def test_update_replaces_value(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.update("user_name", "Sam")
        mems = store.retrieve()
        name_mems = [m for m in mems if m.key == "user_name"]
        assert len(name_mems) == 1
        assert name_mems[0].value == "Sam"

    def test_update_clears_forgotten_flag(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.forget("user_name")
        store.update("user_name", "Sam")
        mems = store.retrieve()
        assert any(m.key == "user_name" and m.value == "Sam" for m in mems)

    def test_later_extraction_updates_key(self):
        """A second extraction of the same key overwrites the first."""
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.extract("My name is Jordan.", turn_n=3)
        mems = [m for m in store.retrieve() if m.key == "user_name"]
        assert len(mems) == 1
        assert mems[0].value == "Jordan"


# ---------------------------------------------------------------------------
# memory_block formatting
# ---------------------------------------------------------------------------

class TestMemoryBlock:
    def test_retrieve_injected_into_prompt(self):
        """Retrieved memories appear in the [Memory] block."""
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.extract("I live in Toronto.", turn_n=1)
        mems = store.retrieve()
        block = memory_block(mems)
        assert MEMORY_BLOCK_HEADER in block
        assert "user_name: Alex" in block
        assert "user_location" in block

    def test_no_injection_when_empty(self):
        """An empty store produces no [Memory] block."""
        store = MemoryStore()
        block = memory_block(store.retrieve())
        assert block == ""
        assert MEMORY_BLOCK_HEADER not in block

    def test_no_block_after_all_forgotten(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        store.forget("user_name")
        block = memory_block(store.retrieve())
        assert block == ""

    def test_block_header_present(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=1)
        block = memory_block(store.retrieve())
        assert block.startswith(MEMORY_BLOCK_HEADER)


# ---------------------------------------------------------------------------
# Turn survival (M1 scenario at unit level)
# ---------------------------------------------------------------------------

class TestTurnSurvival:
    def test_fact_survives_to_turn_8(self):
        """A fact extracted on turn 2 is returned unchanged on turn 8."""
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=2)
        # Simulate turns 3..8 with unrelated text.
        for t in range(3, 9):
            store.extract("what is the weather today", turn_n=t)
        mems = store.retrieve()
        name_mems = [m for m in mems if m.key == "user_name"]
        assert name_mems, "user_name should still be in memory on turn 8"
        assert name_mems[0].value == "Alex"

    def test_multiple_facts_all_survive(self):
        store = MemoryStore()
        store.extract("My name is Alex.", turn_n=2)
        store.extract("I live in Toronto.", turn_n=3)
        store.extract("I prefer shorter answers.", turn_n=4)
        for t in range(5, 9):
            store.extract("some unrelated text", turn_n=t)
        mems = store.retrieve()
        keys = {m.key for m in mems}
        assert "user_name" in keys
        assert "user_location" in keys
        assert "user_preference" in keys


# ---------------------------------------------------------------------------
# Host integration: delegate receives the memory block as system_prefix
# ---------------------------------------------------------------------------

class TestHostMemoryIntegration:
    @pytest.mark.asyncio
    async def test_memory_injected_into_delegate_on_turn_2(self):
        """A fact from turn 1 is included in the system_prefix on turn 2."""
        from tests.voice.unit.fakes import (
            FakeDelegate,
            FakeRecognizer,
            FakeSpeech,
            fake_ports,
            EndlessQueue,
        )
        from app.local.voice.host import VoiceConfig, VoiceHost

        # Turn 1: user states their name -> extracted into memory
        # Turn 2: unrelated question -> delegate should receive [Memory] block
        fake_delegate = FakeDelegate(replies=[["Hello.", "How can I help?"], ["That is correct."]])
        recognizer = FakeRecognizer(turns=["my name is Alex", "what is two plus two"])
        speech = FakeSpeech()

        ports = fake_ports(recognizer=recognizer, delegate=fake_delegate, speech=speech)
        host = VoiceHost(VoiceConfig(), ports=ports)

        await host.run_session(EndlessQueue(), score=1.0)

        assert len(fake_delegate.calls) == 2
        # Turn 2's prefix should contain the memory block with "Alex"
        prefix_turn2 = fake_delegate.prefixes[1]
        assert prefix_turn2 is not None, "Expected a system_prefix on turn 2"
        assert "user_name" in prefix_turn2
        assert "Alex" in prefix_turn2
        assert MEMORY_BLOCK_HEADER in prefix_turn2

    @pytest.mark.asyncio
    async def test_no_memory_block_when_no_facts(self):
        """No memory block is passed to the delegate when nothing has been extracted."""
        from tests.voice.unit.fakes import (
            FakeDelegate,
            FakeRecognizer,
            FakeSpeech,
            fake_ports,
            EndlessQueue,
        )
        from app.local.voice.host import VoiceConfig, VoiceHost

        fake_delegate = FakeDelegate()
        recognizer = FakeRecognizer(turns=["what is the weather"])
        speech = FakeSpeech()

        ports = fake_ports(recognizer=recognizer, delegate=fake_delegate, speech=speech)
        host = VoiceHost(VoiceConfig(), ports=ports)

        await host.run_session(EndlessQueue(), score=1.0)

        assert len(fake_delegate.calls) == 1
        # No facts were extracted on turn 1 from "what is the weather"
        prefix = fake_delegate.prefixes[0]
        # Either None or absent of MEMORY_BLOCK_HEADER
        assert prefix is None or MEMORY_BLOCK_HEADER not in (prefix or "")
