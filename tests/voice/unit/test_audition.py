"""The audition (PL68): every reaction, thinking line and farewell, rendered, listed by name and
played in turn. A fake synthesiser and a fake speaker; no Kokoro."""
from __future__ import annotations

import wave

import numpy as np
import pytest

from app.local import tts_engine
from app.local.voice import audition as audition_mod
from app.local.voice import expression
from app.local.voice.audition import Entry, audition, entries, main

RATE = 24_000


async def _synth(text: str) -> np.ndarray:
    return np.full(len(text) * 100, 0.1, dtype=np.float32)


def test_every_line_of_every_list_is_auditioned_once_by_name():
    lines = entries()
    expected = [*(("reaction", n) for n in expression.REACTIONS),
                *(("thinking", n) for n in expression.THINKING),
                *(("farewell", n) for n in expression.FAREWELLS),
                *(("plain_farewell", n) for n in expression.PLAIN_FAREWELL)]
    assert [(e.list_name, e.name) for e in lines] == expected
    assert len({e.file_name for e in lines}) == len(lines)


async def test_each_line_is_written_listed_and_played_in_turn(tmp_path):
    played: list[int] = []
    printed: list[str] = []
    done = await audition(entries(), _synth, tmp_path, play=lambda pcm, rate: played.append(
        len(pcm)), pause_s=0, echo=printed.append)
    assert all(path is not None and path.exists() for _e, path in done)
    for entry, path in done:
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == RATE and w.getnchannels() == 1
            assert w.getnframes() == len(expression.pronounce(entry.text)) * 100
        assert any(entry.name in row and path.name in row for row in printed)
    assert played == [len(expression.pronounce(e.text)) * 100 for e in entries()]


async def test_the_audition_says_what_the_voice_is_given(tmp_path):
    said: list[str] = []

    async def synth(text):
        said.append(text)
        return np.ones(10, np.float32)

    await audition([Entry("reaction", "test", "Mm-hmm.")], synth, tmp_path, echo=lambda _: None)
    assert said == ["Um-hum."]


async def test_a_line_the_gate_refuses_is_neither_written_nor_played(tmp_path):
    played: list[int] = []
    done = await audition(entries(), _synth, tmp_path, passes=lambda t: t != "Talk soon.",
                          play=lambda pcm, rate: played.append(1), pause_s=0,
                          echo=lambda _: None)
    refused = [e for e, path in done if path is None]
    assert [e.name for e in refused] == ["talk_soon"]
    assert not (tmp_path / "farewell-talk_soon.wav").exists()
    assert len(played) == len(entries()) - 1


async def test_a_name_struck_from_its_list_is_gone_from_the_audition_and_the_assistant(
        monkeypatch, tmp_path):
    thinking = {k: v for k, v in expression.THINKING.items() if k != "okay_so"}
    monkeypatch.setattr(expression, "THINKING", thinking)
    assert "okay_so" not in [e.name for e in entries()]
    picked = {expression.pick(expression.THINKING, last) for last in [None, *thinking]}
    assert "okay_so" not in picked


@pytest.mark.parametrize("mode", ["personal", "ideascribe", "senior_care", "classroom"])
def test_every_line_passes_each_products_gate(mode):
    passes = audition_mod._gate(mode)
    assert all(passes(e.text) for e in entries())


def test_the_command_runs_end_to_end_with_no_model(tmp_path, monkeypatch, capsys):
    voices: list[tuple[str, float]] = []

    async def synthesize(text, *, voice="", speed=1.0):
        voices.append((voice, speed))
        return np.full(240, 0.1, np.float32)

    played: list[int] = []
    monkeypatch.setattr(tts_engine, "synthesize", synthesize)
    monkeypatch.setattr(audition_mod, "_speaker", lambda: lambda pcm, rate: played.append(rate))
    assert main(["--out", str(tmp_path / "a"), "--play", "--voice", "af_test", "--speed",
                 "1.1", "--pause", "0"]) == 0
    out = capsys.readouterr().out
    assert f"Audition: {len(entries())} lines, voice af_test" in out
    assert set(voices) == {("af_test", 1.1)} and len(played) == len(entries())
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == \
        sorted(e.file_name for e in entries())
    assert "expression.py" in out
