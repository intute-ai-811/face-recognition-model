# app/services/enrollment.py
"""Enrollment flow for POST /register. Responses unchanged (see tests/test_api_contract.py)."""
from __future__ import annotations

import logging
import os

from fastapi import UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import config
from app.api.uploads import decode_image, read_upload_capped
from app.infrastructure.firebase import FirebaseClient, encode_debug_jpg

log = logging.getLogger("app.api.server")


async def enroll_person(
    pipe,
    firebase: FirebaseClient,
    rid: str,
    user_id: str,
    name: str,
    files: list[UploadFile],
    max_images: int,
):
    person_id = str(user_id).strip()

    if not person_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "request_id": rid, "message": "user_id required"},
        )
    if not files:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "request_id": rid, "message": "No files[] received"},
        )

    try:
        if hasattr(pipe, "db") and hasattr(pipe.db, "set_person"):
            pipe.db.set_person(person_id, name)
    except Exception as e:
        log.warning("set_person_name_failed", extra={"request_id": rid, "person_id": person_id, "error": str(e)})

    firebase.try_init("/register", request_id=rid, record_error=False)

    cap = min(max_images, getattr(config, "MAX_IMAGES", max_images))
    max_bytes = getattr(config, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

    total_faces = total_emb = processed = errors = 0
    uploaded = []

    log.info("api_register_in", extra={"request_id": rid, "person_id": person_id, "files_count": len(files), "cap": cap})

    for i, f in enumerate(files[:cap], start=1):
        try:
            raw = await read_upload_capped(f, max_bytes)
            if not raw:
                errors += 1
                log.warning("register_empty_file", extra={"request_id": rid, "idx": i})
                continue
            if len(raw) > max_bytes:
                errors += 1
                log.warning("register_file_too_large", extra={"request_id": rid, "idx": i, "bytes": len(raw)})
                continue

            safe_name = os.path.basename(getattr(f, "filename", f"frame_{i:04d}.jpg")) or f"frame_{i:04d}.jpg"

            img = await run_in_threadpool(decode_image, raw)
            if img is None:
                errors += 1
                log.warning("register_decode_fail", extra={"request_id": rid, "idx": i})
                continue

            res = await run_in_threadpool(pipe.enroll_image, img, person_id=person_id, source=safe_name)
            faces = int(res.get("faces", 0))
            embs = int(res.get("embeddings_added", 0))
            total_faces += faces
            total_emb += embs
            processed += 1

            if firebase.enabled and firebase.initialized and ((i <= 5) or faces == 0 or embs == 0):
                debug_bytes = encode_debug_jpg(img, max_w=640, quality=80)
                if debug_bytes:
                    reason = "first5" if i <= 5 else ("noface" if faces == 0 else "noemb")
                    debug_name = f"{reason}_{i:04d}_{safe_name}"
                    debug_folder = f"users/{person_id}/debug_failed"
                    firebase.upload_in_background(rid, debug_folder, debug_name, debug_bytes, "image/jpeg")
                    uploaded.append({"file": debug_name, "folder": debug_folder, "reason": reason})

        except Exception as e:
            errors += 1
            log.exception("register_frame_error", extra={"request_id": rid, "idx": i, "error": str(e)})

    summary = {
        "ok": total_emb >= config.ENROLL_MIN_GOOD_EMBEDDINGS,
        "request_id": rid,
        "person_id": person_id,
        "person_name": name,
        "received_files": len(files),
        "processed_files": processed,
        "errors": errors,
        "faces_detected": total_faces,
        "embeddings_added": total_emb,
        "uploaded_debug": uploaded,
    }
    log.info("api_register_out", extra={k: v for k, v in summary.items() if k != "person_name"})

    if total_emb == 0:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "request_id": rid, "message": "No valid images were enrolled.", "details": summary},
        )

    if total_emb < config.ENROLL_MIN_GOOD_EMBEDDINGS:
        return JSONResponse(
            status_code=400,
            content={
                "status": "insufficient_enrollment",
                "request_id": rid,
                "embeddings_added": total_emb,
                "message": (
                    f"Only {total_emb} face embeddings captured. Need at least "
                    f"{config.ENROLL_MIN_GOOD_EMBEDDINGS}. Ensure good lighting and hold face steady."
                ),
                "details": summary,
            },
        )

    return summary
