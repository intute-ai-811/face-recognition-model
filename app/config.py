#app/config.py
from __future__ import annotations
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR.parent / "data"
MODELS_DIR: Path = BASE_DIR.parent / "models"
FACES_DIR: Path = DATA_DIR / "faces"
EMBED_DIR: Path = DATA_DIR / "embeddings"
EMBED_INDEX_FILE: Path = EMBED_DIR / "index.pkl"
PEOPLE_FILE: Path = DATA_DIR / "people.pkl"

# ── Models ───────────────────────────────────────────────────────────────────
SCRFD_MODEL: Path = Path(os.getenv("SCRFD_MODEL", str(MODELS_DIR / "scrfd_2.5g_bnkps.onnx")))
MOBILEFACENET_MODEL: Path = Path(os.getenv("MOBILEFACENET_MODEL", str(MODELS_DIR / "w600k_mbf.onnx")))
ORT_PROVIDERS: list[str] = ["CPUExecutionProvider"]

# ── Detection & Alignment ────────────────────────────────────────────────────
DET_SCORE_THRESH: float = 0.45
DET_NMS_THRESH: float = 0.45
ALIGNED_SIZE: int = 112
EMB_DIM: int = 512

# ── Recognition Logic ────────────────────────────────────────────────────────
# Start conservative for attendance, then tune after real data.
SIMILARITY_THRESHOLD: float = 0.50
TOP_K: int = 3
RECOG_MIN_FACE_PX: int = 80

# ── Enrollment ───────────────────────────────────────────────────────────────
VIDEO_SAMPLE_EVERY: int = 2
VIDEO_MAX_FRAMES: int = 180
ENROLL_MIN_FACE_PX: int = 140
ENROLL_MAX_TEMPLATES: int = 35
DIVERSITY_COSINE_TOL: float = 0.05

# Request/session caps
MAX_IMAGES: int = 35
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024
ENROLL_MIN_GOOD_EMBEDDINGS: int = 5

# ── ERP Integration ──────────────────────────────────────────────────────────
ERP_ATTENDANCE_URL: str = os.getenv(
    "ERP_ATTENDANCE_URL",
    "https://j7lo9j074j.execute-api.ap-south-1.amazonaws.com/attendanceConfirmation",
)