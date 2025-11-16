import logging
import time
import cv2
import numpy as np
import os
import json
import io
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.logging_config import setup_logging
from app.core.recognize import Pipeline

# ─────────────── Setup ────────────────
setup_logging()
app = FastAPI(title="Attendance Face API")
log = logging.getLogger(__name__)
PIPE = Pipeline()

# ───────────── Firebase Setup ─────────────
firebase_initialized = False
firebase_bucket = None
firebase_error: str | None = None  # capture init error (if any)


def _load_bucket_from_google_services(gs_path: str = "google-services.json") -> Optional[str]:
    """Read bucket name from google-services.json if available."""
    if not os.path.exists(gs_path):
        return None
    with open(gs_path, "r") as f:
        cfg = json.load(f)
    try:
        return cfg["project_info"]["storage_bucket"]
    except Exception:
        return None


def init_firebase_once():
    """Initialize Firebase Admin using credentials + bucket."""
    global firebase_initialized, firebase_bucket
    if firebase_initialized:
        return

    cred_path = os.getenv("FIREBASE_CREDENTIALS_FILE")  # path to firebase_service.json
    if not cred_path or not os.path.exists(cred_path):
        raise RuntimeError("FIREBASE_CREDENTIALS_FILE not set or file not found.")

    bucket_name = os.getenv("FIREBASE_BUCKET") or _load_bucket_from_google_services()
    if not bucket_name:
        raise RuntimeError("No Firebase bucket configured. Set FIREBASE_BUCKET or include google-services.json.")

    import firebase_admin
    from firebase_admin import credentials, storage

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    firebase_bucket = storage.bucket()
    firebase_initialized = True
    logging.getLogger(__name__).info(f"Firebase initialized with bucket={bucket_name}")


def firebase_upload_bytes(
    folder_path: str,
    filename: str,
    data: bytes,
    content_type: str = "image/jpeg",
) -> Tuple[str, str]:
    """Upload bytes to Firebase Storage."""
    init_firebase_once()
    blob_path = f"{folder_path.rstrip('/')}/{filename}"
    blob = firebase_bucket.blob(blob_path)
    blob.upload_from_file(io.BytesIO(data), content_type=content_type)
    try:
        blob.make_public()  # may fail if Uniform Bucket-Level Access is on
        public_url = blob.public_url
    except Exception:
        public_url = ""
    gs_url = f"gs://{firebase_bucket.name}/{blob_path}"
    return gs_url, public_url


# ───────────── Init on startup ─────────────
@app.on_event("startup")
def _startup_init_firebase():
    global firebase_error
    try:
        init_firebase_once()
    except Exception as e:
        firebase_error = f"{type(e).__name__}: {e}"
        logging.getLogger(__name__).warning(f"Firebase init failed on startup: {firebase_error}")


# ───────────── Middleware ─────────────
@app.middleware("http")
async def http_logger(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    try:
        response = await call_next(request)
        dt = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.http").info(
            "http",
            extra={
                "method": method,
                "path": path,
                "status": response.status_code,
                "client": client,
                "ms": dt,
            },
        )
        return response
    except Exception as e:
        dt = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.http").exception(
            "http_error",
            extra={"method": method, "path": path, "client": client, "ms": dt, "error": str(e)},
        )
        raise


# ───────────── Errors ─────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error", "details": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.getLogger("app.api.server").exception("unhandled_exception")
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ───────────── Health ─────────────
@app.get("/healthz")
def healthz():
    # Lazy init if reload happened after startup
    global firebase_error
    if not firebase_initialized and firebase_error is None:
        try:
            init_firebase_once()
        except Exception as e:
            firebase_error = f"{type(e).__name__}: {e}"

    bucket_name = getattr(firebase_bucket, "name", None)
    return {
        "status": "ok",
        "firebase_ready": firebase_initialized,
        "firebase_bucket": bucket_name,
        "firebase_error": firebase_error,
        "env_seen": {
            "FIREBASE_CREDENTIALS_FILE": os.environ.get("FIREBASE_CREDENTIALS_FILE"),
            "FIREBASE_BUCKET": os.environ.get("FIREBASE_BUCKET"),
        },
        "people": PIPE.db.people(),
    }


# ───────────── Register Endpoint ─────────────
@app.post("/register")
async def register(
    name: str = Form(""),
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    max_images: int = Query(20, ge=1, le=200),
):
    from app import config

    person_id = str(user_id).strip()
    if not person_id:
        return JSONResponse(status_code=400, content={"error": "user_id required"})
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files[] received"})

    # Initialize Firebase once (no-op if already initialized)
    try:
        init_firebase_once()
    except Exception as e:
        # still allow local enroll, but surface error
        logging.getLogger(__name__).warning(f"Firebase init on /register failed: {e}")

    cap = min(max_images, getattr(config, "MAX_IMAGES", max_images))
    total_faces = total_emb = processed = errors = 0
    uploaded = []

    log.info(
        "api_register_in",
        extra={"person_id": person_id, "person_name": name, "files_count": len(files), "cap": cap},
    )

    for i, f in enumerate(files[:cap], start=1):
        try:
            raw = f.file.read()
            if not raw:
                errors += 1
                log.warning(
                    "register_empty_file",
                    extra={"idx": i, "upload_filename": getattr(f, "filename", None)},
                )
                continue

            safe_name = getattr(f, "filename", f"frame_{i:04d}.jpg")
            folder = f"users/{person_id}/raw"

            # Upload to Firebase (if initialized)
            if firebase_initialized:
                try:
                    gs_url, public_url = firebase_upload_bytes(
                        folder,
                        safe_name,
                        raw,
                        content_type=getattr(f, "content_type", "image/jpeg"),
                    )
                    uploaded.append(
                        {"file": safe_name, "gs_url": gs_url, "public_url": public_url}
                    )
                except Exception as up_e:
                    errors += 1
                    log.exception(
                        "firebase_upload_failed",
                        extra={"idx": i, "file": safe_name, "error": str(up_e)},
                    )

            # Enroll locally
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                errors += 1
                log.warning(
                    "register_decode_fail",
                    extra={"idx": i, "upload_filename": getattr(f, "filename", None)},
                )
                continue

            res = PIPE.enroll_image(img, person_id=person_id, source=safe_name)
            total_faces += int(res.get("faces", 0))
            total_emb += int(res.get("embeddings_added", 0))
            processed += 1

        except Exception as e:
            errors += 1
            log.exception("register_frame_error", extra={"idx": i, "error": str(e)})

    summary = {
        "ok": True,
        "person_id": person_id,
        "person_name": name,
        "received_files": len(files),
        "processed_files": processed,
        "errors": errors,
        "faces_detected": total_faces,
        "embeddings_added": total_emb,
        "uploaded": uploaded,
    }
    log.info("api_register_out", extra=summary)
    return summary


# ───────────── Mark Attendance (ERP ENABLED on confirm) ─────────────

@app.post("/mark_attendance")
async def mark_attendance(
    file: UploadFile = File(...),
):
    """
    Single-shot attendance:

    - Mobile app sends an image frame.
    - Server runs face recognition.
    - If confidence is high enough, it immediately sends the mapped user_id
      to the ERP/Lambda URL to write into the attendance DB.
    - Mobile app gets a simple confirmation JSON, no follow-up / confirm_user_id.
    """
    from app.clients.http import post_json
    from app import config

    try:
        # ─────────────────────────────────────────────
        # 0) Read & decode image
        # ─────────────────────────────────────────────
        raw = file.file.read()
        if not raw:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Empty image upload"},
            )

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid image"},
            )

        # ─────────────────────────────────────────────
        # 1) Run recognition
        # ─────────────────────────────────────────────
        preds = PIPE.recognize_image(img)

        if preds is None:
            return {"status": "no_face", "message": "No face detected."}

        if isinstance(preds, np.ndarray):
            if preds.size == 0:
                return {"status": "no_face", "message": "No face detected."}
            preds = list(preds)

        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if len(preds) == 0:
            return {"status": "no_face", "message": "No face detected."}

        best = preds[0]
        best_pid = str(best["prediction"]["person_id"])
        best_sim = float(best["prediction"]["similarity"])
        top_k = best.get("top_k", [])

        auto_thr = getattr(config, "ATTEND_AUTO_THRESHOLD", 0.75)
        maybe_thr = getattr(config, "ATTEND_MAYBE_THRESHOLD", 0.60)

        # ─────────────────────────────────────────────
        # 2) Check confidence
        # ─────────────────────────────────────────────
        if best_pid == "unknown" or best_sim < maybe_thr:
            # Not confident at all → do NOT mark
            return {
                "status": "no_match",
                "best": {"person_id": best_pid, "similarity": best_sim},
                "message": "Face not confidently recognized. Attendance not marked.",
            }

        # For anything below auto threshold, you can either:
        # - also auto-mark, OR
        # - be strict and refuse.
        #
        # Here we'll be strict: only auto-mark if >= auto_thr.
        if best_sim < auto_thr:
            return {
                "status": "low_confidence",
                "best": {"person_id": best_pid, "similarity": best_sim},
                "candidates": top_k[:3],
                "message": "Face recognized with low confidence. Attendance not auto-marked.",
            }

        # ─────────────────────────────────────────────
        # 3) Map recognized person_id → ERP user_id
        # ─────────────────────────────────────────────
        # Support two cases:
        #  - best_pid is already a numeric user_id ("72")
        #  - best_pid is a label ("rhythm") that must be mapped via ERP_USER_MAP
        erp_user_id = None

        if best_pid.isdigit():
            erp_user_id = int(best_pid)
        else:
            erp_user_id = config.ERP_USER_MAP.get(best_pid)

        if erp_user_id is None:
            # Cannot map to ERP id → log & bail
            log.warning(
                "erp_user_map_missing",
                extra={"best_pid": best_pid, "similarity": best_sim},
            )
            return {
                "status": "mapping_error",
                "best": {"person_id": best_pid, "similarity": best_sim},
                "message": f"Recognized {best_pid} but no ERP user mapping found. Attendance not marked.",
            }

        payload = {"user_id": erp_user_id}

        # ─────────────────────────────────────────────
        # 4) Fire request to ERP/Lambda to upsert attendance
        # ─────────────────────────────────────────────
        # We still await the HTTP call (so we can log issues),
        # but the mobile app just gets a simple success/fail message
        # and never needs to call back with confirm_user_id.
        try:
            status, resp = await post_json(config.ERP_ATTENDANCE_URL, payload)

            log.info(
                "erp_attendance_sent",
                extra={
                    "person_id": best_pid,
                    "erp_user_id": erp_user_id,
                    "similarity": best_sim,
                    "payload": payload,
                    "status": status,
                    "resp": resp,
                },
            )
        except Exception as e:
            log.exception(
                "erp_attendance_http_failed",
                extra={"error": str(e), "payload": payload},
            )
            # Tell app: face recognized, but ERP write failed
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "best": {"person_id": best_pid, "similarity": best_sim},
                    "message": "Face recognized but failed to mark attendance in ERP.",
                },
            )

        # Optional: you *can* look at status to decide what to tell app.
        # If you truly don't care and just want "marked", you can always return "marked".
        if status not in (200, 201):
            return JSONResponse(
                status_code=502,
                content={
                    "status": "erp_error",
                    "best": {"person_id": best_pid, "similarity": best_sim},
                    "message": "Face recognized but ERP/Lambda returned an error.",
                },
            )

        # ─────────────────────────────────────────────
        # 5) Success response to mobile app
        # ─────────────────────────────────────────────
        # Simple confirmation: attendance was (attempted to be) marked.
        return {
            "status": "marked",
            "user_id": erp_user_id,
            "person_id": best_pid,
            "similarity": best_sim,
            "message": "Attendance marked successfully.",
        }

    except Exception as e:
        log.exception("mark_attendance_failed", extra={"error": str(e)})
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)},
        )



# ───────────── Debug Firebase Test ─────────────
@app.post("/debug/firebase_test")
async def firebase_test(user_id: str = Form(...)):
    try:
        init_firebase_once()
        payload = f"hello from server at {int(time.time())}\n".encode()
        gs_url, public_url = firebase_upload_bytes(
            f"users/{user_id}/debug",
            "ping.txt",
            payload,
            content_type="text/plain",
        )
        return {
            "ok": True,
            "gs_url": gs_url,
            "public_url": public_url,
            "bucket": firebase_bucket.name,
        }
    except Exception as e:
        log.exception("firebase_test_failed", extra={"error": str(e)})
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


# ───────────── Root ─────────────
@app.get("/")
def root():
    return {
        "message": "Attendance Face API",
        "try": ["/healthz", "POST /register", "POST /mark_attendance"],
    }
