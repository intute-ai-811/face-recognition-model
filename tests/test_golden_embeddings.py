"""
Golden-embedding regression test.

tests/golden/embedder_golden.npz was produced by the production stack before
the PR 4 upgrade (Python 3.11, onnxruntime 1.19.2) with scripts/make_golden_embeddings.py.
Any runtime or preprocessing change must reproduce those vectors for the same
model, so enrolled embeddings stay compatible.

Skipped when the model file is not present (e.g. in CI, where models are not
stored in git). Run locally or on the VPS with models/ available.
"""
import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest

from app import config

GOLDEN = Path(__file__).parent / "golden" / "embedder_golden.npz"
MIN_COSINE = 0.9999


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


def test_golden_file_contains_no_real_images(golden):
    # Synthetic only: 4 noise tiles + 2 gradients, 112x112.
    assert golden["inputs"].shape == (6, 112, 112, 3)


def test_embeddings_match_golden(golden):
    model = Path(config.MOBILEFACENET_MODEL)
    if not model.exists():
        pytest.skip(f"model not present at {model}")
    pytest.importorskip("insightface")
    assert _sha256(model) == str(golden["model_sha256"]), (
        "Different model file than the golden was generated with. "
        "A model change needs a new golden and a re-enrollment plan (see docs/MODERNIZATION_PLAN.md)."
    )

    logging.disable(logging.CRITICAL)
    try:
        from app.core.embedder import FaceEmbedder
        embs = FaceEmbedder().embed(list(golden["inputs"]))
    finally:
        logging.disable(logging.NOTSET)

    expected = golden["embeddings"]
    cos = np.sum(embs * expected, axis=1) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(expected, axis=1)
    )
    assert cos.min() >= MIN_COSINE, f"min cosine {cos.min():.6f} < {MIN_COSINE}"
