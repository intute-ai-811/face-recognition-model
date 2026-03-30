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
        if hasattr(self.detector, "det_thresh"): self.detector.det_thresh = config.DET_SCORE_THRESH
        if hasattr(self.detector, "nms_thresh"): self.detector.nms_thresh = config.DET_NMS_THRESH
        log.info("detector_ready", extra={"model": str(model_path.resolve())})

    def detect(self, img_bgr: np.ndarray):
        with timed(log, "detect_infer"):
            bboxes, kps = self.detector.detect(img_bgr)
        n = 0 if bboxes is None else len(bboxes)
        log.info("detect_out", extra={"faces": n})
        if bboxes is None or n == 0:
            return (np.zeros((0, 5), np.float32),
                    np.zeros((0, 5, 2), np.float32))
        return bboxes.astype(np.float32), kps.astype(np.float32)
