#app/core/embedder.py
import logging, numpy as np, cv2
from pathlib import Path
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from app import config
from app.logging_config import timed, get_logger

log = get_logger(__name__)

class FaceEmbedder:
    def __init__(self):
        model_path = Path(config.MOBILEFACENET_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"MobileFaceNet model not found at {model_path.resolve()}")
        self.arc = ArcFaceONNX(str(model_path))
        self.arc.prepare(ctx_id=-1)
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
