#!/usr/bin/env python3
"""Train an openWakeWord classifier head for "hey vox" from synthetic clips.

openWakeWord splits detection into a frozen front end (melspectrogram plus a shared speech
embedding model, both shipped with the package) and a small per-phrase head that reads the last
16 embedding frames (1.28 s) and returns a probability. This script trains only that head, on
CPU, in about a minute, without PyTorch:

  1. Positives: every clip in --clips, augmented (speed, gain, noise at a random SNR, phrase end
     placed 0 to 400 ms before the window end so consecutive frames fire and patience can pass).
  2. Negatives: every clip in --negatives (confusable phrases and ordinary speech), the first
     part of each positive alone ("hey ..." without "vox"), and noise-only windows.
  3. A one-hidden-layer MLP (scikit-learn) on the flattened 16 x 96 embedding window.
  4. Export to ONNX with input [1, 16, 96] and output [1, 1], the contract openwakeword's
     Model() loads; the feature standardisation is folded into the first layer.

The bake-off negative set (scratch/voicebench/fixtures) and the hey_vox_*.aiff fixtures are
never read here, so the bake-off scores are not training scores.

Inputs come from tools/wake_synth_clips.py. Usage:

  uv run tools/wake_synth_clips.py
  uv run --extra wake --extra train tools/train_wake_model.py

Outputs --output/hey_vox.onnx and --output/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("train_wake_model")

_SERVER_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _SERVER_ROOT.parent
_DEFAULT_CLIPS = _WORKSPACE / "scratch" / "wake" / "positive"
_DEFAULT_NEGATIVES = _WORKSPACE / "scratch" / "wake" / "negative"
_DEFAULT_OUTPUT = _SERVER_ROOT / "app" / "local" / "voice" / "models"
_MODEL_STEM = "hey_vox"          # openwakeword keys predictions by file stem

_SR = 16_000
_WINDOW_S = 2.0                  # 2.0 s of audio gives exactly the 16 embedding frames the head reads
_WINDOW_FRAMES = 16
_EMBED_DIM = 96
_FRAME_SAMPLES = 1280            # 80 ms at 16 kHz, the openwakeword frame

_MIN_POSITIVE_CLIPS = 10
_AUG_PER_POSITIVE = 30
_END_OFFSET_MAX_S = 0.40         # phrase end lands this far or less before the window end
_SPEED_RANGE = (0.92, 1.08)
_GAIN_DB_RANGE = (-12.0, 6.0)
_SNR_DB_RANGE = (5.0, 35.0)
_PARTIAL_RANGE = (0.40, 0.60)    # fraction of the phrase kept for "hey ..." hard negatives
_PARTIAL_PER_POSITIVE = 6
_NEG_AUG_PER_CLIP = 3
_NEG_LEAD_MAX_S = 1.0
_NEG_TAIL_MAX_S = 1.0
_NOISE_STREAM_S = 30.0
_NOISE_STREAMS = 6
_TARGET_PEAK = 0.5               # peak normalisation before the gain draw

_HIDDEN_UNITS = 32
_L2 = 1e-4
_MAX_ITER = 300
_VAL_FRACTION = 0.15
_DETECT_THRESHOLD = 0.5          # reported metrics use the gate's stage-1 threshold
_EMBED_BATCH = 64


def _read_wav(path: Path):
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != _SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, _SR)
        data = resample_poly(data, _SR // g, sr // g).astype(np.float32)
    return data


def _trim(x, floor: float = 0.01):
    import numpy as np
    idx = np.flatnonzero(np.abs(x) > floor * max(1e-6, float(np.abs(x).max())))
    return x[idx[0]: idx[-1] + 1] if idx.size else x


def _speed(x, factor: float, np):
    n = max(1, int(round(len(x) / factor)))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _noise(n: int, kind: str, rng, np):
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.arange(spec.shape[0], dtype=np.float32) + 1.0
    spec /= np.sqrt(f) if kind == "pink" else f
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / max(1e-6, float(np.std(out)))


def _mix(sig, rng, np, babble_pool):
    """Add noise at a random SNR; a quarter of the time the noise is quiet background speech."""
    p_sig = float(np.mean(sig ** 2)) + 1e-9
    if babble_pool and rng.random() < 0.25:
        b = babble_pool[int(rng.integers(len(babble_pool)))]
        b = np.resize(b, sig.shape[0]).astype(np.float32)
        noise = b / max(1e-6, float(np.std(b)))
    else:
        noise = _noise(sig.shape[0], ("white", "pink", "brown")[int(rng.integers(3))], rng, np)
    snr = rng.uniform(*_SNR_DB_RANGE)
    noise *= np.sqrt(p_sig / (10 ** (snr / 10)))
    return sig + noise


def _to_int16(x, rng, np):
    x = x / max(1e-6, float(np.abs(x).max())) * _TARGET_PEAK
    x = x * (10 ** (rng.uniform(*_GAIN_DB_RANGE) / 20))
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def _place_at_end(phrase, end_offset_s: float, np):
    """A 2.0 s window whose phrase ends end_offset_s before the window end."""
    n = int(_WINDOW_S * _SR)
    out = np.zeros(n, dtype=np.float32)
    tail = int(end_offset_s * _SR)
    seg = phrase[-(n - tail):] if len(phrase) > n - tail else phrase
    start = n - tail - len(seg)
    out[start: start + len(seg)] = seg
    return out


def _build_positive_windows(clips, rng, np, babble):
    pos, partial = [], []
    for x in clips:
        x = _trim(x)
        for _ in range(_AUG_PER_POSITIVE):
            y = _speed(x, rng.uniform(*_SPEED_RANGE), np)
            w = _place_at_end(y, rng.uniform(0.0, _END_OFFSET_MAX_S), np)
            pos.append(_to_int16(_mix(w, rng, np, babble), rng, np))
        for _ in range(_PARTIAL_PER_POSITIVE):
            cut = x[: int(len(x) * rng.uniform(*_PARTIAL_RANGE))]
            w = _place_at_end(cut, rng.uniform(0.0, _END_OFFSET_MAX_S), np)
            partial.append(_to_int16(_mix(w, rng, np, babble), rng, np))
    return pos, partial


def _build_negative_streams(clips, rng, np, babble):
    streams = []
    for x in clips:
        for _ in range(_NEG_AUG_PER_CLIP):
            y = _speed(_trim(x), rng.uniform(*_SPEED_RANGE), np)
            lead = np.zeros(int(rng.uniform(0.0, _NEG_LEAD_MAX_S) * _SR) + int(_WINDOW_S * _SR),
                            dtype=np.float32)
            tail = np.zeros(int(rng.uniform(0.0, _NEG_TAIL_MAX_S) * _SR), dtype=np.float32)
            s = np.concatenate([lead, y, tail])
            streams.append(_to_int16(_mix(s, rng, np, babble), rng, np))
    for i in range(_NOISE_STREAMS):
        kind = ("white", "pink", "brown")[i % 3]
        s = _noise(int(_NOISE_STREAM_S * _SR), kind, rng, np) * 0.05
        streams.append(_to_int16(s, rng, np))
    return streams


def _embed_windows(feats, windows, np):
    """Last 16 embedding frames of each 2.0 s window -> (N, 16, 96)."""
    if not windows:
        return np.zeros((0, _WINDOW_FRAMES, _EMBED_DIM), dtype=np.float32)
    emb = feats.embed_clips(np.stack(windows), batch_size=_EMBED_BATCH)
    return emb[:, -_WINDOW_FRAMES:, :].astype(np.float32)


def _embed_streams(feats, streams, np):
    """Every 16-frame window of each stream -> (N, 16, 96)."""
    out = []
    for s in streams:
        # embed_clips needs a length that is a whole number of 80 ms frames.
        s = np.pad(s, (0, (-len(s)) % _FRAME_SAMPLES))
        emb = feats.embed_clips(s[None, :], batch_size=1)[0]
        for i in range(0, emb.shape[0] - _WINDOW_FRAMES + 1):
            out.append(emb[i: i + _WINDOW_FRAMES])
    if not out:
        return np.zeros((0, _WINDOW_FRAMES, _EMBED_DIM), dtype=np.float32)
    return np.stack(out).astype(np.float32)


def _export_onnx(mlp, mean, std, path: Path, np) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w1, w2 = mlp.coefs_
    b1, b2 = mlp.intercepts_
    # Fold (x - mean) / std into the first layer: W1' = W1 / std, b1' = b1 - (mean / std) @ W1.
    w1f = (w1 / std[:, None]).astype(np.float32)
    b1f = (b1 - (mean / std) @ w1).astype(np.float32)
    inits = [
        numpy_helper.from_array(w1f, "w1"), numpy_helper.from_array(b1f, "b1"),
        numpy_helper.from_array(w2.astype(np.float32), "w2"),
        numpy_helper.from_array(b2.astype(np.float32), "b2"),
    ]
    nodes = [
        helper.make_node("Flatten", ["x"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w1", "b1"], ["h"]),
        helper.make_node("Relu", ["h"], ["hr"]),
        helper.make_node("Gemm", ["hr", "w2", "b2"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["score"]),
    ]
    graph = helper.make_graph(
        nodes, _MODEL_STEM,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, _WINDOW_FRAMES, _EMBED_DIM])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _metrics(p, y, np) -> dict:
    pred = p >= _DETECT_THRESHOLD
    pos = y == 1
    return {
        "positives": int(pos.sum()), "negatives": int((~pos).sum()),
        "recall": round(float(pred[pos].mean()) if pos.any() else 0.0, 4),
        "false_positive_rate": round(float(pred[~pos].mean()) if (~pos).any() else 0.0, 5),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clips", type=Path, default=_DEFAULT_CLIPS, help="positive wav clips")
    p.add_argument("--negatives", type=Path, default=_DEFAULT_NEGATIVES, help="negative wav clips")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="output directory")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import numpy as np
        from openwakeword.utils import AudioFeatures
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:
        log.error("missing training dependency (%s); run with --extra wake --extra train", exc)
        return 1

    positives = sorted(args.clips.glob("*.wav"))
    negatives = sorted(args.negatives.glob("*.wav"))
    if len(positives) < _MIN_POSITIVE_CLIPS:
        log.error("only %d positive clips in %s; need %d (run tools/wake_synth_clips.py)",
                  len(positives), args.clips, _MIN_POSITIVE_CLIPS)
        return 1
    log.info("positive clips=%d negative clips=%d", len(positives), len(negatives))

    rng = np.random.default_rng(args.seed)
    pos_audio = [_read_wav(f) for f in positives]
    neg_audio = [_read_wav(f) for f in negatives]

    # Split by source clip so no augmentation of a validation clip is trained on.
    def split(items):
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * _VAL_FRACTION))
        return [items[i] for i in idx[n_val:]], [items[i] for i in idx[:n_val]]

    pos_tr, pos_va = split(pos_audio)
    neg_tr, neg_va = split(neg_audio) if neg_audio else ([], [])
    babble = [_trim(x) for x in neg_tr]

    feats = AudioFeatures(inference_framework="onnx")

    def build(pos_clips, neg_clips):
        pw, partial = _build_positive_windows(pos_clips, rng, np, babble)
        streams = _build_negative_streams(neg_clips, rng, np, babble)
        xp = _embed_windows(feats, pw, np)
        xn = np.concatenate([_embed_windows(feats, partial, np), _embed_streams(feats, streams, np)])
        x = np.concatenate([xp, xn]).reshape(-1, _WINDOW_FRAMES * _EMBED_DIM)
        y = np.concatenate([np.ones(len(xp)), np.zeros(len(xn))]).astype(np.int64)
        return x, y

    log.info("augmenting and embedding (training split)")
    x_tr, y_tr = build(pos_tr, neg_tr)
    log.info("augmenting and embedding (validation split)")
    x_va, y_va = build(pos_va, neg_va)
    log.info("train windows pos=%d neg=%d  val windows pos=%d neg=%d",
             int(y_tr.sum()), int((y_tr == 0).sum()), int(y_va.sum()), int((y_va == 0).sum()))

    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0) + 1e-6
    mlp = MLPClassifier(hidden_layer_sizes=(_HIDDEN_UNITS,), alpha=_L2, max_iter=_MAX_ITER,
                        early_stopping=True, random_state=args.seed)
    mlp.fit((x_tr - mean) / std, y_tr)

    train_m = _metrics(mlp.predict_proba((x_tr - mean) / std)[:, 1], y_tr, np)
    val_m = _metrics(mlp.predict_proba((x_va - mean) / std)[:, 1], y_va, np)
    log.info("train %s", train_m)
    log.info("val   %s", val_m)

    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / f"{_MODEL_STEM}.onnx"
    _export_onnx(mlp, mean, std, onnx_path, np)

    # The exported graph must reproduce the sklearn scores, or openwakeword would run a different model.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = x_va[: min(len(x_va), 32)].reshape(-1, _WINDOW_FRAMES, _EMBED_DIM)
    onnx_scores = np.array([sess.run(None, {"x": w[None]})[0][0][0] for w in probe])
    sk_scores = mlp.predict_proba((probe.reshape(len(probe), -1) - mean) / std)[:, 1]
    max_diff = float(np.abs(onnx_scores - sk_scores).max())
    if max_diff > 1e-3:
        log.error("ONNX export does not match the trained model (max diff %.5f)", max_diff)
        return 1

    import openwakeword
    import sklearn
    manifest = {
        "model": _MODEL_STEM,
        "file": onnx_path.name,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "positive_clips": len(positives),
        "negative_clips": len(negatives),
        "positive_source": "macOS say: Daniel, Moira, Samantha, rates 0.9x to 1.3x (tools/wake_synth_clips.py)",
        "front_end": f"openwakeword {getattr(openwakeword, '__version__', '0.6.0')} melspectrogram + embedding (Apache-2.0)",
        "head": f"scikit-learn {sklearn.__version__} MLP, {_HIDDEN_UNITS} hidden units",
        "seed": args.seed,
        "metrics_at_threshold": _DETECT_THRESHOLD,
        "train": train_m,
        "validation": val_m,
        "onnx_vs_sklearn_max_diff": round(max_diff, 6),
        "not_used_for_training": ["ideascribe-app/docs/PRD/fixtures/hey_vox_*.aiff",
                                  "scratch/voicebench/fixtures"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("model %s sha256=%s", onnx_path, manifest["sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
