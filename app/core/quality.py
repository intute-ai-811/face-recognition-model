# app/core/quality.py
from __future__ import annotations
import numpy as np
import cv2


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality_scores(face_bgr: np.ndarray) -> dict:
    """Return simple quality metrics: sharpness and brightness."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharp = variance_of_laplacian(gray)
    bright = float(gray.mean())
    return {"sharpness": sharp, "brightness": bright}


def passes_quality(
    q: dict,
    min_sharp: float = 35.0,
    min_brightness: float = 40.0,
    max_brightness: float = 230.0,
) -> bool:
    return (q["sharpness"] >= min_sharp) and (min_brightness <= q["brightness"] <= max_brightness)
