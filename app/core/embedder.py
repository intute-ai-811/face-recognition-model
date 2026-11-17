# app/core/embedder.py
"""
ArcFace-family embedder on ONNX Runtime.

ArcFaceOnnx reproduces insightface 0.7.3's model_zoo.arcface_onnx.ArcFaceONNX.get_feat()
(BGR->RGB, (x - mean) / std, NCHW blob). insightface decides mean/std by looking
for Sub/Mul nodes at the start of the graph; the deployed w600k_mbf model has none,
so mean = std = 127.5. That is pinned here instead of parsing the graph (which
needed the onnx package), and the golden-embedding test guards it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app import config
from app.core.ort_session import create_session
from app.logging_config import get_logger, timed

log = get_logger(__name__)


class ArcFaceOnnx:
    def __init__(self, model_path: Path, input_mean: float = 127.5, input_std: float = 127.5):
        self.session = create_session(model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = tuple(inp.shape[2:4][::-1])
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(f"expected one embedding output, got {len(outputs)}")
        self.output_names = [outputs[0].name]
        self.input_mean = input_mean
        self.input_std = input_std

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True,
        )
        return self.session.run(self.output_names, {self.input_name: blob})[0]


class FaceEmbedder:
    def __init__(self):
        model_path = Path(config.MOBILEFACENET_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"MobileFaceNet model not found at {model_path.resolve()}")
        self.arc = ArcFaceOnnx(model_path, input_mean=config.EMBED_INPUT_MEAN, input_std=config.EMBED_INPUT_STD)
        log.info("embedder_ready", extra={"model": str(model_path.resolve())})

    def embed(self, aligned_faces):
        n = len(aligned_faces)
        if n == 0:
            return np.zeros((0, config.EMB_DIM), np.float32)
        with timed(log, "embed_infer", faces=n):
            feats = []
            for img in aligned_faces:
                if img.shape[:2] != (config.ALIGNED_SIZE, config.ALIGNED_SIZE):
                    img = cv2.resize(img, (config.ALIGNED_SIZE, config.ALIGNED_SIZE))
                f = self.arc.get_feat(img)
                f = f / (np.linalg.norm(f) + 1e-12)
                feats.append(f.astype(np.float32))
        log.info("embed_out", extra={"faces": n})
        return np.vstack(feats)
