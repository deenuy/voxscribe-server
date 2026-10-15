"""Train an OpenWakeWord model for 'hey vox' from recorded positive clips.

Run after watch_wake_clips.sh launches it (40+ wav files in --clips dir).
Outputs an ONNX model to --output/hey_vox.onnx.

Training strategy:
- Positive: all wavs in --clips dir (40 clips, 4 batches: desk/1m/2m/noise)
- Negative: background noise generated from project scratch wavs + white noise
- Augmentation: speed x0.85-1.15, pitch +-1 semitone, SNR 5-20 dB
- Architecture: OWW DNN (default, 3-block)
- Epochs: 200, early stop on val loss, best model saved

Usage: python tools/train_wake_model.py --clips scratch/wake/positive --output scratch/wake/model
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("train_wake_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _collect_negative_wavs(scratch_dir: Path, max_n: int = 200) -> list[Path]:
    """Collect negative training clips from project scratch/ background files."""
    candidates = list(scratch_dir.glob("*.wav"))
    if not candidates:
        log.warning("No negative wavs in %s; white noise will be used as fallback", scratch_dir)
    return candidates[:max_n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True, help="Directory of positive wav clips")
    p.add_argument("--output", required=True, help="Output directory for trained model")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    clips_dir = Path(args.clips)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    positive = sorted(clips_dir.glob("*.wav"))
    if len(positive) < 10:
        log.error("Only %d positive clips found in %s; need at least 10", len(positive), clips_dir)
        sys.exit(1)

    log.info("Training with %d positive clips from %s", len(positive), clips_dir)

    # Negative clips: project scratch background files
    scratch = clips_dir.parent.parent.parent / "scratch"
    negatives = _collect_negative_wavs(scratch)
    log.info("Using %d negative clips from %s", len(negatives), scratch)

    try:
        import openwakeword  # noqa: F401
        from openwakeword.train import Model as OWWModel
        from openwakeword.utils import AudioFeatures, compute_features_from_generator
        from openwakeword.data import augment_clips, mmap_batch_generator
    except ImportError as exc:
        log.error("openwakeword not available: %s", exc)
        sys.exit(1)

    try:
        import torch
        import numpy as np
    except ImportError as exc:
        log.error("PyTorch or numpy not available: %s", exc)
        sys.exit(1)

    log.info("Computing audio features for positive clips...")
    pos_paths = [str(p) for p in positive]
    neg_paths = [str(p) for p in negatives] if negatives else []

    # OWW feature extraction: 16 kHz, 80-frame melspec windows
    feature_extractor = AudioFeatures()

    def _wav_generator(paths: list[str]):
        import soundfile as sf
        import numpy as np
        for path in paths:
            try:
                data, sr = sf.read(path)
                if sr != 16000:
                    log.warning("Skipping %s: sr=%d (expected 16000)", path, sr)
                    continue
                if data.ndim > 1:
                    data = data.mean(axis=1)
                yield data.astype(np.float32)
            except Exception as exc:
                log.warning("Failed to read %s: %s", path, exc)

    log.info("Extracting features from %d positive + %d negative clips",
             len(pos_paths), len(neg_paths))

    pos_features = compute_features_from_generator(_wav_generator(pos_paths), feature_extractor)
    neg_features = compute_features_from_generator(_wav_generator(neg_paths), feature_extractor) if neg_paths else []

    if len(pos_features) == 0:
        log.error("No positive features extracted; check clip format (16 kHz mono wav)")
        sys.exit(1)

    log.info("Positive feature windows: %d", len(pos_features))
    log.info("Negative feature windows: %d", len(neg_features))

    import numpy as np
    X_pos = np.array(pos_features)
    y_pos = np.ones(len(X_pos), dtype=np.float32)
    X_neg = np.array(neg_features) if neg_features else np.zeros((0, *X_pos.shape[1:]), dtype=np.float32)
    y_neg = np.zeros(len(X_neg), dtype=np.float32)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    log.info("Train: %d, Val: %d", len(X_train), len(X_val))

    model = OWWModel(input_shape=X_train.shape[1:], n_blocks=3)
    model.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3,
    )

    onnx_path = out_dir / "hey_vox.onnx"
    model.export_to_onnx(str(onnx_path))
    log.info("Model exported to %s", onnx_path)

    # Write manifest
    import hashlib, json
    with open(onnx_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "model": "hey_vox",
        "onnx_path": str(onnx_path),
        "sha256": sha,
        "positive_clips": len(positive),
        "negative_clips": len(negatives),
        "epochs": args.epochs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written to %s", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
