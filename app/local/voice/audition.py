"""Hear every line the assistant may say on its own (PL68).

    python -m app.local.voice.audition --out ./audition [--play] [--voice af_heart] [--speed 1.0]

Synthesises each reaction, thinking line and farewell in app/local/voice/expression.py with the
configured voice (Kokoro, as replies are; the voice is KOKORO_VOICE unless --voice names one),
writes each as a WAV file named after its list and its name, prints the list, and with --play
plays them in turn through the default output with a pause between. Each line is first put to
the gate of the chosen product mode; one the gate refuses is listed and neither written nor
played. The text synthesised is what the voice is given (the pronunciation table applied).

Strike a line by deleting its name from its list in expression.py: it is then gone from the
assistant and from this audition alike. Loads Kokoro, so it is for the owner to run, not for
the engine or the tests.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local.voice import expression

SAMPLE_RATE = 24_000
PAUSE_S = 1.0


@dataclass(frozen=True)
class Entry:
    list_name: str          # reaction | thinking | farewell | plain_farewell
    name: str               # the entry's name in its list
    text: str               # the words, as the list holds them

    @property
    def file_name(self) -> str:
        return f"{self.list_name}-{self.name}.wav"


def entries() -> list[Entry]:
    """Every line, in list order: reactions, thinking lines, farewells, the plain farewell."""
    out: list[Entry] = []
    for list_name, items in (("reaction", expression.REACTIONS),
                             ("thinking", expression.THINKING),
                             ("farewell", expression.FAREWELLS),
                             ("plain_farewell", expression.PLAIN_FAREWELL)):
        out += [Entry(list_name, name, text) for name, text in items.items()]
    return out


Synthesise = Callable[[str], Awaitable[np.ndarray | None]]
Play = Callable[[np.ndarray, int], None]
Check = Callable[[str], bool]


def write_wav(path: Path, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    samples = (np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


async def audition(lines: Iterable[Entry], synthesise: Synthesise, out_dir: Path, *,
                   passes: Check = lambda text: True, play: Play | None = None,
                   pause_s: float = PAUSE_S, rate: int = SAMPLE_RATE,
                   echo: Callable[[str], None] = print) -> list[tuple[Entry, Path | None]]:
    """Render every line to `out_dir`, print one row each, and play them in turn when `play` is
    given. Returns each entry with its file (None: refused by the gate or not rendered)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done: list[tuple[Entry, Path | None]] = []
    for entry in lines:
        if not passes(entry.text):
            echo(f"  {entry.list_name:<15}{entry.name:<15}{entry.text:<18}refused by the gate")
            done.append((entry, None))
            continue
        pcm = await synthesise(expression.pronounce(entry.text))
        if pcm is None or not len(pcm):
            echo(f"  {entry.list_name:<15}{entry.name:<15}{entry.text:<18}not rendered")
            done.append((entry, None))
            continue
        path = out_dir / entry.file_name
        write_wav(path, pcm, rate)
        echo(f"  {entry.list_name:<15}{entry.name:<15}{entry.text:<18}{path.name}"
             f"  {len(pcm) / rate:.1f} s")
        done.append((entry, path))
    if play is not None:
        for entry, path in done:
            if path is None:
                continue
            echo(f"playing {entry.list_name} {entry.name}")
            with wave.open(str(path), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
            play(pcm / 32768.0, rate)
            time.sleep(pause_s)
    return done


def _gate(mode: str) -> Check:
    """The gate of `mode`'s shipped profile, as a yes or no on a fixed line."""
    from pathlib import Path as _Path

    import voxscribe.profile as core_profile
    from voxscribe.profile.loader import load_profile
    from voxscribe.session.ports import GATE_PASS

    from app.local.voice.adapters.gate_policy import CorePolicyGate, load_policy

    profile = load_profile(_Path(core_profile.__file__).parent / "profiles" / f"{mode}.yaml")
    gate = CorePolicyGate(load_policy(profile, None))
    return lambda text: gate.check(text).decision == GATE_PASS


def _speaker() -> Play:
    """The default output: sounddevice when it is installed, else the macOS player."""
    try:
        import sounddevice as sd

        def play(pcm: np.ndarray, rate: int) -> None:
            sd.play(pcm, rate)
            sd.wait()
        return play
    except Exception:  # noqa: BLE001 - no PortAudio: fall back to a player
        import subprocess
        import tempfile

        def play(pcm: np.ndarray, rate: int) -> None:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                write_wav(Path(f.name), pcm, rate)
                subprocess.run(["afplay", f.name], check=False)
        return play


def main(argv: list[str] | None = None) -> int:
    from app.local import tts_engine

    parser = argparse.ArgumentParser(prog="python -m app.local.voice.audition",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="folder for the WAV files")
    parser.add_argument("--play", action="store_true", help="play each line in turn")
    parser.add_argument("--voice", default=tts_engine._KOKORO_VOICE,
                        help="Kokoro voice (default: KOKORO_VOICE, else af_heart)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause", type=float, default=PAUSE_S,
                        help="seconds between lines when playing")
    parser.add_argument("--mode", default="personal",
                        choices=("personal", "ideascribe", "senior_care", "classroom"),
                        help="the product mode whose gate each line is put to")
    args = parser.parse_args(argv)

    async def synthesise(text: str):
        return await tts_engine.synthesize(text, voice=args.voice, speed=args.speed)

    lines = entries()
    print(f"Audition: {len(lines)} lines, voice {args.voice}, speed {args.speed:.2f}, "
          f"gate {args.mode}, into {args.out}")
    asyncio.run(audition(lines, synthesise, args.out, passes=_gate(args.mode),
                         play=_speaker() if args.play else None, pause_s=args.pause,
                         rate=tts_engine.PLAYBACK_RATE_HZ))
    print("To strike a line, delete its name from its list in app/local/voice/expression.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
