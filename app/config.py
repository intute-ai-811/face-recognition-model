# app/config.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Make BASE_DIR = .../AttendanceSystem/app
BASE_DIR: Path = Path(__file__).resolve().parent
# Models live in project_root/models
MODELS_DIR: Path = BASE_DIR.parent / "models"
# Data lives in project_root/data
DATA_DIR: Path = BASE_DIR.parent / "data"
FACES_DIR: Path = DATA_DIR / "faces"
EMBED_DIR: Path = DATA_DIR / "embeddings"
EMBED_INDEX_FILE: Path = EMBED_DIR / "index.pkl"

# Ensure data directories exist (harmless if already present)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# ── Models (allow env overrides) ──────────────────────────────────────────────
# If you want to point to absolute files, set:
#   export SCRFD_MODEL=/abs/path/to/scrfd_2.5g_bnkps.onnx
#   export MOBILEFACENET_MODEL=/abs/path/to/w600k_mbf.onnx
SCRFD_MODEL: Path = Path(
    os.getenv("SCRFD_MODEL", str(MODELS_DIR / "scrfd_2.5g_bnkps.onnx"))
)
MOBILEFACENET_MODEL: Path = Path(
    os.getenv("MOBILEFACENET_MODEL", str(MODELS_DIR / "w600k_mbf.onnx"))
)

# ONNXRuntime providers (put "CUDAExecutionProvider" first if you add CUDA)
ORT_PROVIDERS: list[str] = ["CPUExecutionProvider"]

# ── Detection ────────────────────────────────────────────────────────────────
DET_SCORE_THRESH: float = 0.45
DET_NMS_THRESH: float = 0.45
MAX_FACES: int = 10

# ── Alignment / Embedding ────────────────────────────────────────────────────
ALIGNED_SIZE: int = 112             # MobileFaceNet expects 112×112
EMB_DIM: Literal[128, 512] = 512    # MobileFaceNet ONNX is typically 128-D

# ── Recognition ──────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.55  # cosine similarity cutoff
TOP_K: int = 3

# ── Misc ─────────────────────────────────────────────────────────────────────
SAVE_RAW_UPLOADS: bool = False

# ── Video enrollment defaults ────────────────────────────────────────────────
VIDEO_SAMPLE_EVERY: int = 2         # keep every 2nd frame
VIDEO_MAX_FRAMES: int = 180         # cap before sampling
ENROLL_MIN_FACE_PX: int = 140       # min face size for enrollment
ENROLL_MAX_TEMPLATES: int = 12      # keep up to 12 diverse embeddings

#Pick Diverse Indice function parameter in core/video.py
DIVERSITY_COSINE_TOL = 0.05

#ERP URL
ERP_ATTENDANCE_URL = "https://j7lo9j074j.execute-api.ap-south-1.amazonaws.com/attendanceConfirmation"

#Name and Id mapping for ERP
ERP_USER_MAP = {
    "rhythm": 72
}

#Confirmation
ATTEND_AUTO_THRESHOLD = 0.75  # >= this → auto mark
ATTEND_MAYBE_THRESHOLD = 0.60 # between this and AUTO → ask confirmation
TOP_K = 3                     # already in your config, used for candidates