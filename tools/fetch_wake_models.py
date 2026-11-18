#!/usr/bin/env python3
"""Download the model files the wake gate needs that are not in any installed wheel.

* sherpa-onnx keyword spotting model for the open-vocabulary adapter (k2-fsa release,
  Apache-2.0, about 19 MB unpacked; only the int8 files are loaded), into
  ~/.cache/voxscribe/kws/ or the directory named by VOXSCRIBE_KWS_MODEL_DIR.
* openWakeWord feature models (melspectrogram and speech embedding, Apache-2.0) into the
  openwakeword package, used by the OWW adapter and the trainer, plus the stock hey_jarvis model
  the bake-off scores as a cross-check.

The Silero voice check and the Whisper confirm use files faster-whisper already ships or
caches. Run once per machine:

  uv run tools/fetch_wake_models.py
"""
from __future__ import annotations

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.local.voice.adapters.wake_openvocab import (  # noqa: E402
    KWS_MODEL_NAME,
    KWS_MODEL_URL,
    default_model_dir,
)

_TIMEOUT_S = 120
_OWW_STOCK = ["hey_jarvis_v0.1"]


def fetch_kws() -> None:
    target = default_model_dir()
    if (target / "tokens.txt").is_file():
        print(f"kws model present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{KWS_MODEL_NAME}.tar.bz2"
        print(f"downloading {KWS_MODEL_URL}")
        with urllib.request.urlopen(KWS_MODEL_URL, timeout=_TIMEOUT_S) as r:
            archive.write_bytes(r.read())
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(tmp, filter="data")
        (Path(tmp) / KWS_MODEL_NAME).rename(target)
    print(f"kws model ready: {target}")


def fetch_oww() -> None:
    from openwakeword.utils import download_models

    download_models(model_names=_OWW_STOCK)
    print("openwakeword feature models ready")


def main() -> int:
    fetch_kws()
    fetch_oww()
    return 0


if __name__ == "__main__":
    sys.exit(main())
