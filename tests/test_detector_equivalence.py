"""
Equivalence of the in-house SCRFD/ArcFace wrappers with insightface 0.7.3 (PR 6).

insightface is no longer a dependency, so this runs only where it is installed
alongside the model files (e.g. `uv pip install insightface==0.7.3` in a dev
environment). It is the direct proof that removing insightface changed nothing:
identical boxes, scores, landmarks and embeddings on the repository's sample
images and on synthetic inputs.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import config

insightface = pytest.importorskip("insightface")
REPO = Path(__file__).resolve().parents[1]
SAMPLES = [p for p in (REPO / n for n in ("rhythm1.jpeg", "rhythm2.jpeg", "rhythm3.jpeg", "test.jpg")) if p.exists()]


def _need(path):
    if not Path(path).exists():
        pytest.skip(f"model not present at {path}")


@pytest.fixture(scope="module")
def detectors():
    _need(config.SCRFD_MODEL)
    from insightface.model_zoo.scrfd import SCRFD
    from app.core.detector import ScrfdOnnx

    ref = SCRFD(str(config.SCRFD_MODEL))
    ref.prepare(ctx_id=-1, input_size=(640, 640))
    ours = ScrfdOnnx(Path(config.SCRFD_MODEL), input_size=(640, 640))
    for d in (ref, ours):
        d.det_thresh, d.nms_thresh = 0.35, config.DET_NMS_THRESH
    return ref, ours


def _images():
    imgs = [cv2.imread(str(p)) for p in SAMPLES]
    rng = np.random.default_rng(7)
    imgs.append(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))          # no face, landscape
    imgs.append(cv2.resize(imgs[0], (300, 900)) if SAMPLES else rng.integers(0, 256, (900, 300, 3), dtype=np.uint8))
    return imgs


def test_scrfd_matches_insightface(detectors):
    ref, ours = detectors
    for img in _images():
        rb, rk = ref.detect(img)
        ob, ok = ours.detect(img)
        assert rb.shape == ob.shape
        np.testing.assert_allclose(ob, rb, rtol=0, atol=1e-3)
        if rk is not None and len(rk):
            np.testing.assert_allclose(ok, rk, rtol=0, atol=1e-3)


def test_arcface_matches_insightface():
    _need(config.MOBILEFACENET_MODEL)
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from app.core.embedder import ArcFaceOnnx

    ref = ArcFaceONNX(str(config.MOBILEFACENET_MODEL))
    ref.prepare(ctx_id=-1)
    ours = ArcFaceOnnx(Path(config.MOBILEFACENET_MODEL), config.EMBED_INPUT_MEAN, config.EMBED_INPUT_STD)
    assert (ref.input_mean, ref.input_std) == (ours.input_mean, ours.input_std)

    rng = np.random.default_rng(11)
    faces = [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
    for f in faces:
        np.testing.assert_allclose(ours.get_feat(f), ref.get_feat(f), rtol=0, atol=1e-5)
