"""Detector threshold and pipeline edge cases (PR 1). Uses fakes; no model files needed."""
import numpy as np
import pytest

from app import config

pytest.importorskip("insightface")
pytest.importorskip("cv2")

from app.core.detector import FaceDetector  # noqa: E402
from app.core.recognize import Pipeline  # noqa: E402


class FakeSCRFD:
    """Mimics SCRFD.detect: returns every box at or above det_thresh."""

    def __init__(self, scores):
        self.scores = scores
        self.det_thresh = None
        self.thresh_seen = []

    def detect(self, img):
        self.thresh_seen.append(self.det_thresh)
        keep = [s for s in self.scores if s >= self.det_thresh]
        if not keep:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.array([[0, 0, 200, 200, s] for s in keep], np.float32)
        kps = np.zeros((len(keep), 5, 2), np.float32)
        return boxes, kps


def make_detector(scores):
    d = FaceDetector.__new__(FaceDetector)
    d.detector = FakeSCRFD(scores)
    d.native_thresh = min(config.DET_SCORE_THRESH, config.ENROLL_DET_SCORE_THRESH)
    d.detector.det_thresh = d.native_thresh
    return d


def test_default_detect_uses_recognition_threshold():
    d = make_detector([0.9, 0.44, 0.40, 0.30])
    boxes, kps = d.detect(np.zeros((8, 8, 3), np.uint8))
    assert boxes[:, 4].tolist() == pytest.approx([0.9])
    assert kps.shape == (1, 5, 2)


def test_enroll_threshold_is_per_call():
    d = make_detector([0.9, 0.44, 0.40, 0.30])
    boxes, _ = d.detect(np.zeros((8, 8, 3), np.uint8), min_score=config.ENROLL_DET_SCORE_THRESH)
    assert boxes[:, 4].tolist() == pytest.approx([0.9, 0.44, 0.40])


def test_shared_detector_threshold_is_never_mutated():
    d = make_detector([0.9, 0.40])
    img = np.zeros((8, 8, 3), np.uint8)
    d.detect(img, min_score=0.35)
    d.detect(img)
    d.detect(img, min_score=0.35)
    assert set(d.detector.thresh_seen) == {d.native_thresh}


def test_threshold_below_floor_is_rejected():
    d = make_detector([0.9])
    with pytest.raises(ValueError):
        d.detect(np.zeros((8, 8, 3), np.uint8), min_score=d.native_thresh - 0.01)


def test_no_boxes_above_threshold_returns_empty_arrays():
    d = make_detector([0.40])
    boxes, kps = d.detect(np.zeros((8, 8, 3), np.uint8))
    assert boxes.shape == (0, 5) and kps.shape == (0, 5, 2)


class FakeDet:
    def __init__(self):
        self.calls = []

    def detect(self, img, min_score=None):
        self.calls.append(min_score)
        boxes = np.array([[0, 0, 200, 200, 0.9]], np.float32)
        kps = np.array([[[70, 90], [130, 90], [100, 120], [75, 150], [125, 150]]], np.float32)
        return boxes, kps


class FakeEmb:
    def embed(self, faces):
        return np.ones((len(faces), config.EMB_DIM), np.float32) / np.sqrt(config.EMB_DIM)


class EmptyDB:
    def search_cosine(self, q, top_k):
        return [[] for _ in range(len(q))]


class LegacyEmptyDB:
    """Pre-PR store behaviour: [] for an empty index."""

    def search_cosine(self, q, top_k):
        return []


def make_pipeline(db):
    p = Pipeline.__new__(Pipeline)
    p.det, p.emb, p.db = FakeDet(), FakeEmb(), db
    return p


@pytest.mark.parametrize("db", [EmptyDB(), LegacyEmptyDB()])
def test_recognize_with_empty_database_returns_unknown(db):
    img = np.full((240, 240, 3), 128, np.uint8)
    out = make_pipeline(db).recognize_image(img)
    assert len(out) == 1
    assert out[0]["prediction"] == {"person_id": "unknown", "similarity": 0.0}
    assert out[0]["top_k"] == []


def test_enroll_image_passes_enroll_threshold_per_call():
    p = make_pipeline(EmptyDB())
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    p.db.add_embeddings = lambda pid, embs, src: len(embs)
    p.enroll_image(img, person_id="70", source="x.jpg")
    assert p.det.calls == [config.ENROLL_DET_SCORE_THRESH]
