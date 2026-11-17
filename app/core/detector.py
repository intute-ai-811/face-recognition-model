#app/core/detector.py
import logging, numpy as np
from pathlib import Path
from insightface.model_zoo.scrfd import SCRFD
from app import config
from app.logging_config import timed, get_logger

log = get_logger(__name__)

class FaceDetector:
    def __init__(self):
        model_path = Path(config.SCRFD_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"SCRFD model not found at {model_path.resolve()}")
        self.detector = SCRFD(str(model_path))
        self.detector.prepare(ctx_id=-1, input_size=(640, 640))
        # The native threshold is a fixed floor and is never changed after startup.
        # Callers choose their own threshold via detect(min_score=...), applied as a
        # post-filter. Because NMS only lets higher-scoring boxes suppress lower ones,
        # "detect at the floor, then keep score >= t" returns exactly the boxes that
        # detecting at t would, so recognition at DET_SCORE_THRESH is unchanged.
        self.native_thresh = min(config.DET_SCORE_THRESH, config.ENROLL_DET_SCORE_THRESH)
        if hasattr(self.detector, "det_thresh"): self.detector.det_thresh = self.native_thresh
        if hasattr(self.detector, "nms_thresh"): self.detector.nms_thresh = config.DET_NMS_THRESH
        log.info("detector_ready", extra={"model": str(model_path.resolve())})

    def detect(self, img_bgr: np.ndarray, min_score: float | None = None):
        thresh = config.DET_SCORE_THRESH if min_score is None else float(min_score)
        if thresh < self.native_thresh:
            raise ValueError(
                f"min_score={thresh} is below the detector floor {self.native_thresh}"
            )
        with timed(log, "detect_infer"):
            bboxes, kps = self.detector.detect(img_bgr)
        if bboxes is None or len(bboxes) == 0:
            log.info("detect_out", extra={"faces": 0})
            return (np.zeros((0, 5), np.float32),
                    np.zeros((0, 5, 2), np.float32))
        keep = bboxes[:, 4] >= thresh
        bboxes = bboxes[keep]
        kps = kps[keep] if kps is not None else None
        n = len(bboxes)
        log.info("detect_out", extra={"faces": n})
        if n == 0 or kps is None:
            return (np.zeros((0, 5), np.float32),
                    np.zeros((0, 5, 2), np.float32))
        return bboxes.astype(np.float32), kps.astype(np.float32)
