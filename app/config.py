from __future__ import annotations
from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR.parent / "data"
MODELS_DIR: Path = BASE_DIR.parent / "models"

FACES_DIR: Path = DATA_DIR / "faces"
EMBED_DIR: Path = DATA_DIR / "embeddings"
EMBED_INDEX_FILE: Path = EMBED_DIR / "index.pkl"
PEOPLE_FILE: Path = DATA_DIR / "people.pkl"

# ── Models (Environment Overrides) ──────────────────────────────────────────
SCRFD_MODEL: Path = Path(os.getenv("SCRFD_MODEL", str(MODELS_DIR / "scrfd_2.5g_bnkps.onnx")))
MOBILEFACENET_MODEL: Path = Path(os.getenv("MOBILEFACENET_MODEL", str(MODELS_DIR / "w600k_mbf.onnx")))
ORT_PROVIDERS: list[str] = ["CPUExecutionProvider"]

# ── Detection & Alignment ───────────────────────────────────────────────────
DET_SCORE_THRESH: float = 0.45
DET_NMS_THRESH: float = 0.45
ALIGNED_SIZE: int = 112 
EMB_DIM: int = 512  # Matching your ArcFaceONNX output

# ── Recognition Logic ───────────────────────────────────────────────────────
# We use a tiered threshold system for the /logout and /attendance flow
ATTEND_AUTO_THRESHOLD: float = 0.75   # Confidence >= 75% -> Success
ATTEND_MAYBE_THRESHOLD: float = 0.60  # 60% - 75% -> Ask for confirmation
SIMILARITY_THRESHOLD: float = 0.60    # Global floor for "unknown" vs "known"
TOP_K: int = 3

# ── Enrollment Defaults ─────────────────────────────────────────────────────
VIDEO_SAMPLE_EVERY: int = 2
VIDEO_MAX_FRAMES: int = 180
ENROLL_MIN_FACE_PX: int = 140
ENROLL_MAX_TEMPLATES: int = 12
DIVERSITY_COSINE_TOL: float = 0.05

# ── ERP Integration ─────────────────────────────────────────────────────────
ERP_ATTENDANCE_URL: str = os.getenv("ERP_ATTENDANCE_URL", "https://j7lo9j074j.execute-api.ap-south-1.amazonaws.com/attendanceConfirmation")