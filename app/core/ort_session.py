# app/core/ort_session.py
"""Shared ONNX Runtime session factory with optional, env-driven thread settings."""
from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort

from app import config
from app.logging_config import get_logger

log = get_logger(__name__)


def _env_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def create_session(model_path: Path) -> ort.InferenceSession:
    """
    CPU inference session.

    Thread counts default to ONNX Runtime's own choice (identical to the previous
    insightface-based setup). They can be tuned per deployment without a code change:
      ORT_INTRA_OP_THREADS  threads used inside one operator (e.g. 2-4 on a small VPS)
      ORT_INTER_OP_THREADS  threads used across independent operators
    Tune with benchmarks/locustfile.py on the target machine before changing them.
    """
    intra = _env_int("ORT_INTRA_OP_THREADS")
    inter = _env_int("ORT_INTER_OP_THREADS")
    opts = None
    if intra is not None or inter is not None:
        opts = ort.SessionOptions()
        if intra is not None:
            opts.intra_op_num_threads = intra
        if inter is not None:
            opts.inter_op_num_threads = inter
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=list(config.ORT_PROVIDERS))
    log.info(
        "ort_session_ready",
        extra={"model": Path(model_path).name, "intra_op_threads": intra, "inter_op_threads": inter},
    )
    return session
