# app/core/detector.py
"""
SCRFD face detector on ONNX Runtime.

ScrfdOnnx reproduces insightface 0.7.3's model_zoo.scrfd.SCRFD.detect() exactly
(same letterbox resize, blob normalisation, anchor decoding, ordering and NMS),
so detections are identical to the previous implementation, without the
insightface package and its ~15 unused dependencies. The equivalence is
checked by tests/test_detector_equivalence.py and scripts/pipeline_snapshot.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app import config
from app.core.ort_session import create_session
from app.logging_config import get_logger, timed
from app.observability.metrics import PIPELINE_STAGE_SECONDS

log = get_logger(__name__)


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


class ScrfdOnnx:
    """SCRFD with keypoints (the *_bnkps models): 3 strides x (score, bbox, kps) outputs."""

    INPUT_MEAN = 127.5
    INPUT_STD = 128.0

    def __init__(self, model_path: Path, input_size: tuple[int, int] = (640, 640)):
        self.session = create_session(model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3

        n_out = len(self.output_names)
        if n_out == 9:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, True
        elif n_out == 15:
            self.fmc, self.strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, True
        else:
            raise ValueError(
                f"Unsupported SCRFD model with {n_out} outputs; a keypoint model (9 or 15 outputs) is required"
            )

        # Fixed-size models declare their input size; dynamic ones use the configured size.
        shape = inp.shape
        self.input_size = input_size if isinstance(shape[2], str) or shape[2] is None else (shape[3], shape[2])
        self.det_thresh = 0.5
        self.nms_thresh = 0.4
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        centers = self._center_cache.get(key)
        if centers is None:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
            if len(self._center_cache) < 100:
                self._center_cache[key] = centers
        return centers

    def _forward(self, img: np.ndarray, threshold: float):
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.INPUT_STD, input_size, (self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height, input_width = blob.shape[2], blob.shape[3]

        scores_list, bboxes_list, kpss_list = [], [], []
        fmc = self.fmc
        for idx, stride in enumerate(self.strides):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                kps_preds = net_outs[idx + fmc * 2] * stride

            centers = self._anchor_centers(input_height // stride, input_width // stride, stride)
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss = distance2kps(centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def nms(self, dets: np.ndarray) -> list[int]:
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img: np.ndarray):
        """Returns (det[N, 5] = x1, y1, x2, y2, score, kpss[N, 5, 2]), sorted by score, after NMS."""
        input_size = self.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward(det_img, self.det_thresh)
        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order, :]
        keep = self.nms(pre_det)
        return pre_det[keep, :], kpss[order, :, :][keep, :, :]


class FaceDetector:
    def __init__(self):
        model_path = Path(config.SCRFD_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"SCRFD model not found at {model_path.resolve()}")
        self.detector = ScrfdOnnx(model_path, input_size=(640, 640))
        # The native threshold is a fixed floor and is never changed after startup.
        # Callers choose their own threshold via detect(min_score=...), applied as a
        # post-filter. Because NMS only lets higher-scoring boxes suppress lower ones,
        # "detect at the floor, then keep score >= t" returns exactly the boxes that
        # detecting at t would, so recognition at DET_SCORE_THRESH is unchanged.
        self.native_thresh = min(config.DET_SCORE_THRESH, config.ENROLL_DET_SCORE_THRESH)
        self.detector.det_thresh = self.native_thresh
        self.detector.nms_thresh = config.DET_NMS_THRESH
        log.info("detector_ready", extra={"model": str(model_path.resolve())})

    def detect(self, img_bgr: np.ndarray, min_score: float | None = None):
        thresh = config.DET_SCORE_THRESH if min_score is None else float(min_score)
        if thresh < self.native_thresh:
            raise ValueError(
                f"min_score={thresh} is below the detector floor {self.native_thresh}"
            )
        with timed(log, "detect_infer"), PIPELINE_STAGE_SECONDS.labels("detect").time():
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
