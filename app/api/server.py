import logging
import time
import cv2
import numpy as np
import os
import json
import io
import uuid
from typing import Optional, Tuple
import threading
import pickle
from fastapi import Header, HTTPException

from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool

from app.logging_config import setup_logging
from app.core.recognize import Pipeline

# ✅ Add this near your Firebase globals (top-level in server.py)
admin_lock = threading.Lock()

def _require_admin(x_admin_token: Optional[str]) -> None:
    expected = os.getenv("ADMIN_TOKEN")  # set this in docker env
    if not expected:
        # safer default: if you forgot to set ADMIN_TOKEN, don't expose admin endpoints
        raise HTTPException(status_code=503, detail="Admin endpoints disabled (ADMIN_TOKEN not set).")
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _people_dump() -> dict:
    """
    Best-effort: prefer PIPE.db.people() if available, else read people.pkl directly.
    """
    # Prefer your DB API if present
    try:
        if hasattr(PIPE, "db") and hasattr(PIPE.db, "people"):
            data = PIPE.db.people()
            # Ensure JSON-safe
            return dict(data) if isinstance(data, dict) else {"people": data}
    except Exception:
        pass

    # Fallback: read people.pkl
    from app import config
    if not config.PEOPLE_FILE.exists():
        return {}
    with open(config.PEOPLE_FILE, "rb") as f:
        obj = pickle.load(f)
    return dict(obj) if isinstance(obj, dict) else {"people": obj}


def _remove_person_from_pickles(person_id: str) -> dict:
    """
    Fallback deletion logic if PIPE.db does not provide deletion APIs.
    Removes:
      - person_id from people.pkl
      - all embeddings with label==person_id from index.pkl
    Returns counts.
    """
    from app import config

    removed_people = 0
    removed_embeddings = 0

    # --- people.pkl ---
    if config.PEOPLE_FILE.exists():
        with open(config.PEOPLE_FILE, "rb") as f:
            people = pickle.load(f)
        if isinstance(people, dict) and person_id in people:
            del people[person_id]
            removed_people = 1
            with open(config.PEOPLE_FILE, "wb") as f:
                pickle.dump(people, f)

    # --- index.pkl ---
    if config.EMBED_INDEX_FILE.exists():
        with open(config.EMBED_INDEX_FILE, "rb") as f:
            idx = pickle.load(f)

        # Common representation: {"embeddings": np.ndarray, "labels": list[str]}
        if isinstance(idx, dict) and "labels" in idx and "embeddings" in idx:
            labels = list(idx["labels"])
            embs = idx["embeddings"]

            keep_mask = [str(l) != str(person_id) for l in labels]
            removed_embeddings = int(len(labels) - sum(keep_mask))

            if removed_embeddings > 0:
                idx["labels"] = [l for l, k in zip(labels, keep_mask) if k]
                # embeddings: np.ndarray shape [N, D]
                idx["embeddings"] = embs[np.array(keep_mask, dtype=bool)]

                with open(config.EMBED_INDEX_FILE, "wb") as f:
                    pickle.dump(idx, f)

        # If your index.pkl is a custom class, we can’t safely mutate it here.
        # In that case, you should implement PIPE.db.remove_person(person_id) and persist.
        else:
            raise HTTPException(
                status_code=500,
                detail="Unknown index.pkl format. Implement deletion in storage layer (PIPE.db.remove_person)."
            )

    return {
        "removed_people_entry": removed_people,
        "removed_embeddings": removed_embeddings,
    }


def _try_remove_person(person_id: str) -> dict:
    """
    Preferred: call storage/db API if you have it.
    Fallback: delete directly from pickle files.
    """
    # 1) Try DB API(s) if your Storage layer has them
    if hasattr(PIPE, "db"):
        db = PIPE.db
        # best guesses for method names; use what exists
        for meth in ("remove_person", "delete_person", "drop_person"):
            if hasattr(db, meth):
                fn = getattr(db, meth)
                out = fn(person_id)
                # if it returns nothing, still ok
                return {"removed_via": f"PIPE.db.{meth}", "result": out}

        # If you have low-level accessors, you can still remove name entry
        # (embeddings removal still needs a db method)
        if hasattr(db, "set_person"):
            # optional: remove name mapping only (won't remove embeddings)
            pass

    # 2) Fallback: edit pickles directly
    out = _remove_person_from_pickles(str(person_id).strip())

    # Optional: if your DB/cache is in memory, you may need to reload
    if hasattr(PIPE, "db") and hasattr(PIPE.db, "reload"):
        try:
            PIPE.db.reload()
            out["db_reload"] = True
        except Exception:
            out["db_reload"] = False

    return {"removed_via": "pickle_fallback", **out}


# ─────────────── Setup ────────────────
setup_logging()
app = FastAPI(title="Attendance Face API")
log = logging.getLogger(__name__)
PIPE = Pipeline()

# ───────────── Firebase Setup ─────────────
firebase_initialized = False
firebase_bucket = None
firebase_error: str | None = None  # capture init error (if any)
firebase_lock = threading.Lock()


def _request_id(request: Optional[Request]) -> str:
    if request is None:
        return str(uuid.uuid4())
    rid = request.headers.get("x-request-id")
    return rid or str(uuid.uuid4())


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

    # prevent concurrent init races
    with firebase_lock:
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
        logging.getLogger(__name__).info("Firebase initialized", extra={"bucket": bucket_name})


def firebase_upload_bytes(
    folder_path: str,
    filename: str,
    data: bytes,
    content_type: str = "image/jpeg",
) -> Tuple[str, str]:
    """Upload bytes to Firebase Storage."""
    init_firebase_once()
    blob_path = f"{folder_path.rstrip('/')}/{os.path.basename(filename)}"
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
        logging.getLogger(__name__).warning("Firebase init failed on startup", extra={"error": firebase_error})


# ───────────── Middleware ─────────────
@app.middleware("http")
async def http_logger(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    rid = _request_id(request)

    try:
        response = await call_next(request)
        dt = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.http").info(
            "http",
            extra={
                "request_id": rid,
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
            extra={"request_id": rid, "method": method, "path": path, "client": client, "ms": dt, "error": str(e)},
        )
        raise


# ───────────── Errors ─────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = _request_id(request)
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "request_id": rid, "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = _request_id(request)
    logging.getLogger("app.api.server").exception("unhandled_exception", extra={"request_id": rid})
    # do not leak internals
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "request_id": rid})


# ───────────── Health ─────────────
@app.get("/healthz")
def healthz(request: Request):
    global firebase_error
    rid = _request_id(request)

    if not firebase_initialized and firebase_error is None:
        try:
            init_firebase_once()
        except Exception as e:
            firebase_error = f"{type(e).__name__}: {e}"

    bucket_name = getattr(firebase_bucket, "name", None)

    # Gate details for production safety
    show_details = os.getenv("HEALTHZ_DETAILS", "0") == "1"

    resp = {
        "status": "ok",
        "request_id": rid,
        "firebase_ready": firebase_initialized,
        "firebase_bucket": bucket_name,
    }
    if show_details:
        resp["firebase_error"] = firebase_error
        resp["env_seen"] = {
            "FIREBASE_CREDENTIALS_FILE": os.environ.get("FIREBASE_CREDENTIALS_FILE"),
            "FIREBASE_BUCKET": os.environ.get("FIREBASE_BUCKET"),
        }
        # expensive + potentially sensitive
        resp["people"] = PIPE.db.people()

    return resp


# ───────────── Register Endpoint ─────────────
@app.post("/register")
async def register(
    request: Request,
    name: str = Form(""),
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    max_images: int = Query(20, ge=1, le=200),
):
    from app import config

    rid = _request_id(request)
    person_id = str(user_id).strip()
    if not person_id:
        return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "user_id required"})
    if not files:
        return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "No files[] received"})

    # Save name locally (NO ERP mapping / auto-update). Best effort.
    try:
        if hasattr(PIPE, "db") and hasattr(PIPE.db, "set_person"):
            PIPE.db.set_person(person_id, name)
    except Exception as e:
        log.warning("set_person_name_failed", extra={"request_id": rid, "person_id": person_id, "error": str(e)})

    # Best-effort Firebase init
    try:
        init_firebase_once()
    except Exception as e:
        logging.getLogger(__name__).warning("Firebase init on /register failed", extra={"request_id": rid, "error": str(e)})

    cap = min(max_images, getattr(config, "MAX_IMAGES", max_images))
    max_bytes = getattr(config, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

    total_faces = total_emb = processed = errors = 0
    uploaded = []

    log.info(
        "api_register_in",
        extra={"request_id": rid, "person_id": person_id, "person_name": name, "files_count": len(files), "cap": cap},
    )

    for i, f in enumerate(files[:cap], start=1):
        try:
            try:
                raw = await f.read()
            finally:
                await f.close()

            if not raw:
                errors += 1
                log.warning("register_empty_file", extra={"request_id": rid, "idx": i, "upload_filename": getattr(f, "filename", None)})
                continue

            if len(raw) > max_bytes:
                errors += 1
                log.warning("register_file_too_large", extra={"request_id": rid, "idx": i, "bytes": len(raw), "max_bytes": max_bytes})
                continue

            safe_name = os.path.basename(getattr(f, "filename", f"frame_{i:04d}.jpg")) or f"frame_{i:04d}.jpg"
            folder = f"users/{person_id}/raw"

            # Upload to Firebase (if initialized). Offload blocking upload.
            if firebase_initialized:
                try:
                    gs_url, public_url = await run_in_threadpool(
                        firebase_upload_bytes,
                        folder,
                        safe_name,
                        raw,
                        getattr(f, "content_type", "image/jpeg"),
                    )
                    uploaded.append({"file": safe_name, "gs_url": gs_url, "public_url": public_url})
                except Exception as up_e:
                    errors += 1
                    log.exception("firebase_upload_failed", extra={"request_id": rid, "idx": i, "file": safe_name, "error": str(up_e)})

            # Decode (offload)
            arr = np.frombuffer(raw, np.uint8)
            img = await run_in_threadpool(cv2.imdecode, arr, cv2.IMREAD_COLOR)
            if img is None:
                errors += 1
                log.warning("register_decode_fail", extra={"request_id": rid, "idx": i, "upload_filename": getattr(f, "filename", None)})
                continue

            # Enroll (offload)
            res = await run_in_threadpool(PIPE.enroll_image, img, person_id=person_id, source=safe_name)
            total_faces += int(res.get("faces", 0))
            total_emb += int(res.get("embeddings_added", 0))
            processed += 1

        except Exception as e:
            errors += 1
            log.exception("register_frame_error", extra={"request_id": rid, "idx": i, "error": str(e)})

    ok = processed > 0 and total_emb > 0

    summary = {
        "ok": ok,
        "request_id": rid,
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

    # IMPORTANT: if nothing got enrolled, return non-200
    if not ok:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "request_id": rid,
                "message": "No valid images were enrolled.",
                "details": summary,
            },
        )

    return summary


# ───────────── Mark Attendance ─────────────
@app.post("/mark_attendance")
async def mark_attendance(request: Request, file: UploadFile = File(...)):
    from app.clients.http import post_json
    from app import config

    rid = _request_id(request)
    max_bytes = getattr(config, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

    try:
        try:
            raw = await file.read()
        finally:
            await file.close()

        if not raw:
            return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "Empty image upload"})

        if len(raw) > max_bytes:
            return JSONResponse(status_code=413, content={"status": "error", "request_id": rid, "message": "File too large"})

        arr = np.frombuffer(raw, np.uint8)
        img = await run_in_threadpool(cv2.imdecode, arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "Invalid image"})

        preds = await run_in_threadpool(PIPE.recognize_image, img)

        if preds is None:
            return JSONResponse(status_code=404, content={"status": "no_face", "request_id": rid, "message": "No face detected."})

        if isinstance(preds, np.ndarray):
            if preds.size == 0:
                return JSONResponse(status_code=404, content={"status": "no_face", "request_id": rid, "message": "No face detected."})
            preds = list(preds)

        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if len(preds) == 0:
            return JSONResponse(status_code=404, content={"status": "no_face", "request_id": rid, "message": "No face detected."})

        best = preds[0]
        best_pid = str(best["prediction"]["person_id"])
        best_sim = float(best["prediction"]["similarity"])
        top_k = best.get("top_k", [])

        # Lookup name from local people store (best effort; no mapping updates).
        person_name = None
        try:
            if hasattr(PIPE, "db") and hasattr(PIPE.db, "get_person_name"):
                person_name = PIPE.db.get_person_name(best_pid)
        except Exception:
            person_name = None

        auto_thr = getattr(config, "ATTEND_AUTO_THRESHOLD", 0.75)
        maybe_thr = getattr(config, "ATTEND_MAYBE_THRESHOLD", 0.60)

        if best_pid == "unknown" or best_sim < maybe_thr:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_match",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Face not confidently recognized. Attendance not marked.",
                },
            )

        if best_sim < auto_thr:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "low_confidence",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "candidates": top_k[:3],
                    "message": "Face recognized with low confidence. Attendance not auto-marked.",
                },
            )

        if best_pid.isdigit():
            erp_user_id = int(best_pid)
        else:
            erp_user_id = config.ERP_USER_MAP.get(best_pid)

        if erp_user_id is None:
            log.warning("erp_user_map_missing", extra={"request_id": rid, "best_pid": best_pid, "similarity": best_sim})
            return JSONResponse(
                status_code=409,
                content={
                    "status": "mapping_error",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Recognized user but no ERP user mapping found. Attendance not marked.",
                },
            )

        payload = {"user_id": erp_user_id}

        try:
            status, resp = await post_json(config.ERP_ATTENDANCE_URL, payload)
            log.info(
                "erp_attendance_sent",
                extra={"request_id": rid, "person_id": best_pid, "erp_user_id": erp_user_id, "similarity": best_sim, "status": status},
            )
            log.debug("erp_attendance_resp_debug", extra={"request_id": rid, "payload": payload, "resp": resp})
        except Exception as e:
            log.exception("erp_attendance_http_failed", extra={"request_id": rid, "error": str(e)})
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Face recognized but failed to mark attendance in ERP.",
                },
            )

        if status not in (200, 201):
            return JSONResponse(
                status_code=502,
                content={
                    "status": "erp_error",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Face recognized but ERP/Lambda returned an error.",
                },
            )

        return {
            "status": "marked",
            "request_id": rid,
            "user_id": erp_user_id,
            "person_id": best_pid,
            "person_name": person_name,
            "similarity": best_sim,
            "message": "Attendance marked successfully.",
        }

    except Exception:
        log.exception("mark_attendance_failed", extra={"request_id": rid})
        return JSONResponse(status_code=500, content={"status": "error", "request_id": rid, "message": "Internal server error"})


# ───────────── Debug Firebase Test ─────────────
@app.post("/debug/firebase_test")
async def firebase_test(request: Request, user_id: str = Form(...)):
    rid = _request_id(request)
    try:
        init_firebase_once()
        payload = f"hello from server at {int(time.time())}\n".encode()
        gs_url, public_url = await run_in_threadpool(
            firebase_upload_bytes,
            f"users/{str(user_id).strip()}/debug",
            "ping.txt",
            payload,
            "text/plain",
        )
        return {"ok": True, "request_id": rid, "gs_url": gs_url, "public_url": public_url, "bucket": firebase_bucket.name}
    except Exception:
        log.exception("firebase_test_failed", extra={"request_id": rid})
        return JSONResponse(status_code=500, content={"ok": False, "request_id": rid, "error": "firebase_test_failed"})


# ───────────── Logout ─────────────
@app.post("/logout")
async def logout(request: Request, file: UploadFile = File(...)):
    from app.clients.http import post_json
    from app import config

    rid = _request_id(request)
    max_bytes = getattr(config, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

    try:
        try:
            raw = await file.read()
        finally:
            await file.close()

        if not raw:
            return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "Empty image upload"})

        if len(raw) > max_bytes:
            return JSONResponse(status_code=413, content={"status": "error", "request_id": rid, "message": "File too large"})

        arr = np.frombuffer(raw, np.uint8)
        img = await run_in_threadpool(cv2.imdecode, arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"status": "error", "request_id": rid, "message": "Invalid image"})

        preds = await run_in_threadpool(PIPE.recognize_image, img)

        if preds is None:
            return JSONResponse(status_code=404, content={"status": "no_face", "request_id": rid, "message": "No face detected."})

        if isinstance(preds, np.ndarray):
            preds = list(preds)

        if not preds:
            return JSONResponse(status_code=404, content={"status": "no_face", "request_id": rid, "message": "No face detected."})

        best = preds[0]
        best_pid = str(best["prediction"]["person_id"])
        best_sim = float(best["prediction"]["similarity"])
        top_k = best.get("top_k", [])

        # Lookup name from local people store (best effort; no mapping updates).
        person_name = None
        try:
            if hasattr(PIPE, "db") and hasattr(PIPE.db, "get_person_name"):
                person_name = PIPE.db.get_person_name(best_pid)
        except Exception:
            person_name = None

        auto_thr = getattr(config, "ATTEND_AUTO_THRESHOLD", 0.75)
        maybe_thr = getattr(config, "ATTEND_MAYBE_THRESHOLD", 0.60)

        if best_pid == "unknown" or best_sim < maybe_thr:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_match",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Face not confidently recognized. Logout not marked.",
                },
            )

        if best_sim < auto_thr:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "low_confidence",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "candidates": top_k[:3],
                    "message": "Face recognized with low confidence. Logout not auto-marked.",
                },
            )

        if best_pid.isdigit():
            erp_user_id = int(best_pid)
        else:
            erp_user_id = config.ERP_USER_MAP.get(best_pid)

        if erp_user_id is None:
            log.warning("erp_user_map_missing_logout", extra={"request_id": rid, "best_pid": best_pid, "similarity": best_sim})
            return JSONResponse(
                status_code=409,
                content={
                    "status": "mapping_error",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Recognized user but no ERP user mapping found.",
                },
            )

        payload = {"user_id": erp_user_id, "type": "OUT"}

        try:
            status, resp = await post_json(config.ERP_LOGOUT_URL, payload)
            log.info(
                "erp_logout_sent",
                extra={"request_id": rid, "person_id": best_pid, "erp_user_id": erp_user_id, "similarity": best_sim, "status": status},
            )
            log.debug("erp_logout_resp_debug", extra={"request_id": rid, "payload": payload, "resp": resp})
        except Exception as e:
            log.exception(
                "erp_logout_http_failed",
                extra={
                    "request_id": rid,
                    "erp_url": config.ERP_LOGOUT_URL,
                    "payload": payload,
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                },
            )
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "request_id": rid,
                    "best": {
                        "person_id": best_pid,
                        "person_name": person_name,
                        "similarity": best_sim,
                    },
                    "message": "Face recognized but failed to mark logout in ERP.",
                },
            )


        if status not in (200, 201):
            return JSONResponse(
                status_code=502,
                content={
                    "status": "erp_error",
                    "request_id": rid,
                    "best": {"person_id": best_pid, "person_name": person_name, "similarity": best_sim},
                    "message": "Face recognized but ERP returned an error.",
                },
            )

        return {
            "status": "logged_out",
            "request_id": rid,
            "user_id": erp_user_id,
            "person_id": best_pid,
            "person_name": person_name,
            "similarity": best_sim,
            "message": "Logout marked successfully.",
        }

    except Exception:
        log.exception("logout_failed", extra={"request_id": rid})
        return JSONResponse(status_code=500, content={"status": "error", "request_id": rid, "message": "Internal server error"})


# ✅ Add these endpoints anywhere below your existing routes (server.py)

@app.get("/admin/people")
def admin_people(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
):
    rid = _request_id(request)
    _require_admin(x_admin_token)

    with admin_lock:
        people = _people_dump()

    return {
        "ok": True,
        "request_id": rid,
        "count": len(people) if isinstance(people, dict) else None,
        "people": people,
    }


@app.delete("/admin/person/{person_id}")
def admin_delete_person(
    person_id: str,
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
):
    rid = _request_id(request)
    _require_admin(x_admin_token)

    pid = str(person_id).strip()
    if not pid:
        raise HTTPException(status_code=400, detail="person_id required")

    with admin_lock:
        result = _try_remove_person(pid)

    log.warning("admin_person_deleted", extra={"request_id": rid, "person_id": pid, "result": result})
    return {"ok": True, "request_id": rid, "person_id": pid, "result": result}

# ───────────── Root ─────────────
@app.get("/")
def root(request: Request):
    return {
        "message": "Attendance Face API",
        "request_id": _request_id(request),
        "try": ["/healthz", "POST /register", "POST /mark_attendance"],
    }
