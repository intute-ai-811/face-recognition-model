# app/api/server.py
"""
FastAPI application: setup, middleware, error handlers and thin route handlers.

Business logic lives in app/services (attendance, enrollment, admin), request
helpers in app/api/uploads.py, auth in app/api/security.py and Firebase in
app/infrastructure/firebase.py. The public API is frozen by tests/test_api_contract.py.

Entry point (unchanged): uvicorn app.api.server:app
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2  # noqa: F401  (re-exported: tests patch server.cv2.imdecode)
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from app import config as app_config
from app.api.security import require_admin, require_enroll_key, token_matches
from app.api.uploads import request_id
from app.core.recognize import Pipeline
from app.infrastructure.firebase import FirebaseClient
from app.logging_config import setup_logging
from app.observability import metrics
from app.services import admin as admin_service
from app.services.attendance import CHECK_IN, CHECK_OUT, recognize_and_report
from app.services.enrollment import enroll_person

setup_logging()
log = logging.getLogger(__name__)

# Module-level singletons, looked up at request time (tests replace PIPE / firebase).
PIPE = Pipeline()
firebase = FirebaseClient()
admin_lock = threading.Lock()


def _store_stat(key: str) -> float:
    stats = getattr(getattr(PIPE, "db", None), "stats", None)
    try:
        return float(stats()[key]) if stats else float("nan")
    except Exception:
        return float("nan")


metrics.STORE_EMBEDDINGS.set_function(lambda: _store_stat("embeddings"))
metrics.STORE_PEOPLE.set_function(lambda: _store_stat("people"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    firebase.try_init("startup")
    yield


app = FastAPI(title="Attendance Face API", lifespan=lifespan)


# ───────────── Middleware ─────────────
@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    limit = getattr(app_config, "MAX_REQUEST_BYTES", None)
    length = request.headers.get("content-length")
    if limit and length and length.isdigit() and int(length) > limit:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "request_id": request_id(request), "message": "Request too large"},
        )
    return await call_next(request)


@app.middleware("http")
async def http_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Route template (e.g. /admin/person/{person_id}), never the raw path, so no IDs end up in labels.
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"
        metrics.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
        metrics.HTTP_LATENCY.labels(route).observe(time.perf_counter() - start)


@app.middleware("http")
async def http_logger(request: Request, call_next):
    start = time.perf_counter()
    fields = {
        "request_id": request_id(request),
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else "unknown",
    }
    try:
        response = await call_next(request)
        logging.getLogger("app.http").info(
            "http", extra={**fields, "status": response.status_code, "ms": int((time.perf_counter() - start) * 1000)}
        )
        return response
    except Exception as e:
        logging.getLogger("app.http").exception(
            "http_error", extra={**fields, "ms": int((time.perf_counter() - start) * 1000), "error": str(e)}
        )
        raise


# ───────────── Errors ─────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "request_id": request_id(request), "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = request_id(request)
    logging.getLogger("app.api.server").exception("unhandled_exception", extra={"request_id": rid})
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "request_id": rid})


# ───────────── Health ─────────────
@app.get("/healthz")
def healthz(request: Request):
    if firebase.enabled and not firebase.initialized and firebase.error is None:
        firebase.try_init("/healthz")

    resp = {
        "status": "ok",
        "request_id": request_id(request),
        "firebase_ready": firebase.initialized,
        "firebase_bucket": getattr(firebase.bucket, "name", None),
    }
    if os.getenv("HEALTHZ_DETAILS", "0") == "1":
        resp["firebase_error"] = firebase.error
        resp["env_seen"] = {
            "FIREBASE_CREDENTIALS_FILE": os.environ.get("FIREBASE_CREDENTIALS_FILE"),
            "FIREBASE_BUCKET": os.environ.get("FIREBASE_BUCKET"),
        }
        resp["people"] = PIPE.db.people()
    return resp


@app.get("/readyz")
def readyz(request: Request):
    """
    Ready = the enrollment store answers. (Models load at import; the process cannot
    start without them.) /healthz stays a plain liveness check with an unchanged response.
    """
    checks = {"store": False, "embeddings": None}
    try:
        db = PIPE.db
        stats = getattr(db, "stats", None)
        checks["embeddings"] = stats()["embeddings"] if stats else sum(dict(db.people()).values())
        checks["store"] = True
    except Exception as e:
        log.warning("readyz_failed", extra={"error": repr(e)})
    ready = checks["store"]
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "request_id": request_id(request), "checks": checks},
    )


@app.get("/metrics")
def metrics_endpoint(authorization: Optional[str] = Header(default=None)):
    """Prometheus metrics. If METRICS_TOKEN is set, requires 'Authorization: Bearer <token>'."""
    expected = os.getenv("METRICS_TOKEN")
    if expected:
        given = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
        if not token_matches(given, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


# ───────────── Enrollment ─────────────
@app.post("/register")
async def register(
    request: Request,
    name: str = Form(""),
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    max_images: int = Query(35, ge=1, le=200),
    x_enroll_token: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
):
    require_enroll_key(x_enroll_token, x_admin_token)
    return await enroll_person(PIPE, firebase, request_id(request), user_id, name, files, max_images)


# ───────────── Attendance ─────────────
@app.post("/mark_attendance")
async def mark_attendance(request: Request, file: UploadFile = File(...)):
    return await recognize_and_report(PIPE, request_id(request), file, CHECK_IN)


@app.post("/logout")
async def logout(request: Request, file: UploadFile = File(...)):
    return await recognize_and_report(PIPE, request_id(request), file, CHECK_OUT)


# ───────────── Admin ─────────────
@app.get("/admin/people")
def admin_people(request: Request, x_admin_token: Optional[str] = Header(default=None)):
    rid = request_id(request)
    require_admin(x_admin_token)
    with admin_lock:
        people = admin_service.people_dump(PIPE)
    return {
        "ok": True,
        "request_id": rid,
        "count": len(people) if isinstance(people, dict) else None,
        "people": people,
    }


@app.delete("/admin/person/{person_id}")
def admin_delete_person(person_id: str, request: Request, x_admin_token: Optional[str] = Header(default=None)):
    rid = request_id(request)
    require_admin(x_admin_token)
    pid = str(person_id).strip()
    if not pid:
        raise HTTPException(status_code=400, detail="person_id required")
    with admin_lock:
        result = admin_service.remove_person(PIPE, pid)
    log.warning("admin_person_deleted", extra={"request_id": rid, "person_id": pid, "result": result})
    return {"ok": True, "request_id": rid, "person_id": pid, "result": result}


# ───────────── Firebase debug (admin only) ─────────────
@app.post("/debug/firebase_test")
async def firebase_test(
    request: Request,
    user_id: str = Form(...),
    x_admin_token: Optional[str] = Header(default=None),
):
    rid = request_id(request)
    require_admin(x_admin_token)
    try:
        firebase.init_once()
        payload = f"hello from server at {int(time.time())}\n".encode()
        gs_url, public_url = await run_in_threadpool(
            firebase.upload_bytes, f"users/{str(user_id).strip()}/debug", "ping.txt", payload, "text/plain"
        )
        return {"ok": True, "request_id": rid, "gs_url": gs_url, "public_url": public_url, "bucket": firebase.bucket.name}
    except Exception:
        log.exception("firebase_test_failed", extra={"request_id": rid})
        return JSONResponse(status_code=500, content={"ok": False, "request_id": rid, "error": "firebase_test_failed"})


# ───────────── Root ─────────────
@app.get("/")
def root(request: Request):
    return {
        "message": "Attendance Face API",
        "request_id": request_id(request),
        "try": ["/healthz", "POST /register", "POST /mark_attendance"],
    }
