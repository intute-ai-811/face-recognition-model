#app/core/video.py
from __future__ import annotations
import cv2
import tempfile
import os
import numpy as np
from typing import List
from app.logging_config import get_logger, timed
from app import config

log = get_logger(__name__)


def iter_video_frames_from_bytes(
    video_bytes: bytes,
    sample_every: int = 2,
    max_frames: int = 180,
) -> List[np.ndarray]:
    log.info("video_read", extra={"bytes": len(video_bytes)})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise ValueError("Could not open video (unsupported codec or corrupt file)")

    frames, idx, kept = [], 0, 0
    with timed(log, "video_decode"):
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                frames.append(frame)
                kept += 1
            idx += 1

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    log.info("video_frames", extra={"decoded": idx, "kept": len(frames), "sample_every": sample_every})
    return frames


def pick_diverse_indices(
    embeddings: np.ndarray,
    max_keep: int = 20,
    dedupe_cosine: float = config.DIVERSITY_COSINE_TOL,
) -> list[int]:
    n = embeddings.shape[0]
    if n <= max_keep:
        log.debug("pick_diverse_indices: small set", extra={"n": n})
        return list(range(n))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embs = embeddings / norms

    keep = [0]
    used = np.zeros(n, dtype=bool)
    used[0] = True

    for _ in range(1, max_keep):
        sims = embs @ embs[keep[-1]].T
        sims = np.clip(sims, -1, 1)
        sims[used] = 1.0
        idx = np.argmin(sims)
        if 1 - sims[idx] < dedupe_cosine:
            break
        keep.append(int(idx))
        used[idx] = True

    log.info(
        "pick_diverse_indices_done",
        extra={"total": n, "kept": len(keep), "dedupe_cosine": dedupe_cosine},
    )
    return keep
