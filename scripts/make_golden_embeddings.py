"""
Generate tests/golden/embedder_golden.npz from the CURRENT runtime.

Run once on a known-good stack; the golden test then checks that any later
runtime (Python, ONNX Runtime, OpenCV, numpy upgrades, or a rewritten
preprocessing path) produces the same embeddings for the same model.

Inputs are synthetic (seeded noise + gradients), never real faces, so the
file contains no biometric data.

    python scripts/make_golden_embeddings.py [--out tests/golden/embedder_golden.npz]
"""
import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_inputs() -> np.ndarray:
    rng = np.random.default_rng(20261117)
    noise = rng.integers(0, 256, size=(4, 112, 112, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:112, 0:112]
    grad1 = np.stack([xx * 2, yy * 2, (xx + yy)], axis=-1).clip(0, 255).astype(np.uint8)
    grad2 = np.stack([255 - xx * 2, (yy * 3) % 256, np.full_like(xx, 128)], axis=-1).astype(np.uint8)
    return np.concatenate([noise, grad1[None], grad2[None]], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden/embedder_golden.npz")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    from app.core.embedder import FaceEmbedder

    model = Path(config.MOBILEFACENET_MODEL)
    inputs = synthetic_inputs()
    embs = FaceEmbedder().embed(list(inputs))
    import onnxruntime
    np.savez_compressed(
        args.out,
        inputs=inputs,
        embeddings=embs.astype(np.float32),
        model_sha256=np.array(sha256(model)),
        generated_with=np.array(f"python {sys.version.split()[0]}, onnxruntime {onnxruntime.__version__}, numpy {np.__version__}"),
    )
    print(f"wrote {args.out}: {embs.shape} embeddings, model sha256 {sha256(model)[:16]}...")


if __name__ == "__main__":
    main()
