#!/usr/bin/env python3
"""Generate synthetic wake-word clips with macOS `say` for training and the T02 bake-off.

Writes 16 kHz mono 16-bit WAV files under --out (default: <workspace>/scratch/wake):

  positive/     "hey vox" in the training voices (Daniel, Moira, Samantha), rates 0.9x to 1.3x.
                Input to tools/train_wake_model.py.
  heldout/      "hey vox" in voices the trained model never saw. Scored, never trained on.
  hey_jarvis/   "hey jarvis" in a spread of voices. Must be rejected by the product.
  negative/     confusable phrases and ordinary sentences in many voices. Training negatives only;
                the bake-off negative set (scratch/voicebench/fixtures) is never used for training.

The default speaking rate of `say` is about 175 words per minute; rates are expressed as a
multiple of that so the spread stays meaningful if a voice changes. The exact 1.0x rate is
skipped for training voices because the hey_vox_*.aiff fixtures were made at the default rate.

Usage:
  uv run tools/wake_synth_clips.py
  uv run tools/wake_synth_clips.py --out /tmp/wake --only positive
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUT = _WORKSPACE / "scratch" / "wake"

_SAY_DEFAULT_WPM = 175
_DATA_FORMAT = "LEI16@16000"   # little-endian int16 at 16 kHz, what the wake path consumes

_WAKE_TEXT = "hey vox"
_JARVIS_TEXT = "hey jarvis"

_TRAIN_VOICES = ("Daniel", "Moira", "Samantha")
_TRAIN_CLIPS = 40
_TRAIN_RATE_MIN = 0.9
_TRAIN_RATE_MAX = 1.3
_DEFAULT_RATE_GUARD = 0.02   # training rates closer than this to 1.0x are moved off it

_HELDOUT_VOICES = ("Karen", "Rishi", "Fred", "Albert", "Kathy", "Ralph",
                   "Eddy (English (US))", "Flo (English (UK))", "Reed (English (UK))",
                   "Shelley (English (US))")
_HELDOUT_RATES = (0.95, 1.15)

_JARVIS_VOICES = _TRAIN_VOICES + _HELDOUT_VOICES[:7]
_JARVIS_CLIPS = 20
_JARVIS_RATES = (0.9, 1.1)

_NEGATIVE_VOICES = _TRAIN_VOICES + _HELDOUT_VOICES
_NEGATIVE_TEXTS = (
    # near misses: share a word or a sound with the wake phrase
    "hey", "vox", "hey you", "hey there", "hey fox", "hey box", "hey socks", "hey max",
    "hey jack", "hey siri", "hey jarvis", "okay google", "hey folks", "hey mom",
    "a box of rocks", "the fox jumped", "they walked home", "hey what's up", "pay the bills",
    "hello", "hi everyone", "vaccine", "faxes", "hey vicky", "say vote",
    # ordinary speech
    "what time is it", "can you set a timer", "I think we should leave soon",
    "the meeting moved to three o'clock", "did you feed the cat", "let me check my calendar",
    "turn the music down please", "we need more milk and bread", "how was your day",
    "that was a great movie", "call me when you get there", "it's raining again today",
)
_NEGATIVE_RATES = (0.9, 1.15)


def _rates(n: int, lo: float, hi: float, skip_default: bool) -> list[float]:
    """n rates spread evenly over [lo, hi]; moves any rate near 1.0x off the default when asked."""
    step = (hi - lo) / max(1, n - 1)
    out = []
    for i in range(n):
        r = round(lo + i * step, 3)
        if skip_default and abs(r - 1.0) < _DEFAULT_RATE_GUARD:
            r = 1.0 - _DEFAULT_RATE_GUARD if r < 1.0 else 1.0 + _DEFAULT_RATE_GUARD
        out.append(r)
    return out


def _say(text: str, voice: str, rate: float, path: Path) -> None:
    wpm = str(int(round(_SAY_DEFAULT_WPM * rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-v", voice, "-r", wpm, "-o", str(path), f"--data-format={_DATA_FORMAT}", text],
        check=True,
    )


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s.split(" (")[0]).strip("_")


def gen_positive(out: Path) -> int:
    rates = _rates(_TRAIN_CLIPS, _TRAIN_RATE_MIN, _TRAIN_RATE_MAX, skip_default=True)
    for i, (rate, voice) in enumerate(zip(rates, itertools.cycle(_TRAIN_VOICES))):
        _say(_WAKE_TEXT, voice, rate, out / "positive" / f"hey_vox_{_slug(voice)}_{i:02d}_r{rate:.2f}.wav")
    return len(rates)


def gen_heldout(out: Path) -> int:
    n = 0
    for voice in _HELDOUT_VOICES:
        for rate in _HELDOUT_RATES:
            _say(_WAKE_TEXT, voice, rate, out / "heldout" / f"hey_vox_{_slug(voice)}_r{rate:.2f}.wav")
            n += 1
    return n


def gen_jarvis(out: Path) -> int:
    pairs = list(itertools.product(_JARVIS_VOICES, _JARVIS_RATES))[:_JARVIS_CLIPS]
    for voice, rate in pairs:
        _say(_JARVIS_TEXT, voice, rate, out / "hey_jarvis" / f"hey_jarvis_{_slug(voice)}_r{rate:.2f}.wav")
    return len(pairs)


def gen_negative(out: Path) -> int:
    n = 0
    for t_i, text in enumerate(_NEGATIVE_TEXTS):
        for v_i, voice in enumerate(_NEGATIVE_VOICES):
            # Every text in every voice would be ~500 clips; rotate voices so each text gets four.
            if (t_i + v_i) % len(_NEGATIVE_VOICES) >= 4:
                continue
            rate = _NEGATIVE_RATES[(t_i + v_i) % len(_NEGATIVE_RATES)]
            _say(text, voice, rate, out / "negative" / f"neg_{t_i:02d}_{_slug(voice)}_r{rate:.2f}.wav")
            n += 1
    return n


_SETS = {"positive": gen_positive, "heldout": gen_heldout, "hey_jarvis": gen_jarvis,
         "negative": gen_negative}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--only", choices=sorted(_SETS), action="append",
                   help="generate only this set (repeatable); default all")
    args = p.parse_args(argv)
    if shutil.which("say") is None:
        print("macOS `say` not found", file=sys.stderr)
        return 1
    for name in args.only or list(_SETS):
        target = args.out / name
        if target.exists():
            shutil.rmtree(target)
        n = _SETS[name](args.out)
        print(f"{name:11s} {n:4d} clips -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
