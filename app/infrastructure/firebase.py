# app/infrastructure/firebase.py
"""Firebase Storage client for optional debug uploads (UPLOAD_TO_FIREBASE=1). Objects stay private."""
from __future__ import annotations

import io
import json
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _load_bucket_from_google_services(gs_path: str = "google-services.json") -> str | None:
    if not os.path.exists(gs_path):
        return None
    with open(gs_path, "r") as f:
        cfg = json.load(f)
    try:
        return cfg["project_info"]["storage_bucket"]
    except Exception:
        return None


def encode_debug_jpg(img: np.ndarray, max_w: int = 640, quality: int = 80) -> bytes:
    h, w = img.shape[:2]
    if w > max_w:
        new_h = int(h * (max_w / w))
        img = cv2.resize(img, (max_w, new_h))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return enc.tobytes() if ok else b""


class FirebaseClient:
    def __init__(self, enabled: bool | None = None):
        self.enabled = (os.getenv("UPLOAD_TO_FIREBASE", "0") == "1") if enabled is None else enabled
        self.initialized = False
        self.bucket = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def init_once(self) -> None:
        if self.initialized:
            return
        with self._lock:
            if self.initialized:
                return

            cred_path = os.getenv("FIREBASE_CREDENTIALS_FILE")
            if not cred_path or not os.path.exists(cred_path):
                raise RuntimeError("FIREBASE_CREDENTIALS_FILE not set or file not found.")

            bucket_name = os.getenv("FIREBASE_BUCKET") or _load_bucket_from_google_services()
            if not bucket_name:
                raise RuntimeError("No Firebase bucket configured.")

            import firebase_admin
            from firebase_admin import credentials, storage

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
            self.bucket = storage.bucket()
            self.initialized = True
            log.info("Firebase initialized", extra={"bucket": bucket_name})

    def try_init(self, context: str, request_id: str | None = None, record_error: bool = True) -> None:
        """Initialise if enabled; never raise. record_error=True stores the failure for /healthz."""
        if not self.enabled:
            return
        try:
            self.init_once()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if record_error:
                self.error = err
            log.warning(f"Firebase init on {context} failed", extra={"request_id": request_id, "error": err})

    def upload_bytes(self, folder_path: str, filename: str, data: bytes, content_type: str = "image/jpeg") -> tuple[str, str]:
        self.init_once()
        blob_path = f"{folder_path.rstrip('/')}/{os.path.basename(filename)}"
        blob = self.bucket.blob(blob_path)
        blob.upload_from_file(io.BytesIO(data), content_type=content_type)
        # Objects stay private: these are employee face images. public_url is kept
        # (empty) for response compatibility.
        public_url = ""
        gs_url = f"gs://{self.bucket.name}/{blob_path}"
        return gs_url, public_url

    def upload_in_background(self, rid: str, folder: str, filename: str, data: bytes, content_type: str) -> None:
        def run():
            try:
                gs_url, public_url = self.upload_bytes(folder, filename, data, content_type)
                log.info("firebase_upload_ok", extra={"request_id": rid, "file": filename, "gs_url": gs_url})
            except Exception as e:
                log.exception("firebase_upload_failed", extra={"request_id": rid, "file": filename, "error": str(e)})

        threading.Thread(target=run, daemon=True).start()
