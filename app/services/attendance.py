# app/services/attendance.py
"""
One recognise-then-report flow shared by /mark_attendance (check-in) and /logout
(check-out). The two endpoints used to be ~170 near-identical lines each; the only
differences are captured in AttendanceAction. Responses are unchanged (see
tests/test_api_contract.py).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from fastapi import UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import config
from app.api.uploads import decode_image, read_upload_capped
from app.clients import http as erp_http
from app.observability import metrics

log = logging.getLogger("app.api.server")


@dataclass(frozen=True)
class AttendanceAction:
    metric: str               # "check_in" / "check_out" - metrics label
    verb: str                 # "Attendance" / "Logout" - used in messages
    erp_extra: dict           # merged into the ERP payload next to user_id
    sent_event: str           # log event after the ERP call
    failed_event: str         # log event when the ERP call raises
    crash_event: str          # log event for unexpected errors
    ok_status: str            # "status" of the success response
    ok_message: str
    already_status: str       # ERP "status" meaning nothing to do
    already_time_key: str     # ERP attendance field holding the earlier time
    already_message: str      # "... at {time}"
    already_message_no_time: str


CHECK_IN = AttendanceAction(
    metric="check_in",
    verb="Attendance",
    erp_extra={},
    sent_event="erp_attendance_sent",
    failed_event="erp_attendance_http_failed",
    crash_event="mark_attendance_failed",
    ok_status="marked",
    ok_message="Attendance marked successfully.",
    already_status="already_checked_in",
    already_time_key="check_in_time_ist_text",
    already_message="Already checked in at {time}",
    already_message_no_time="Already checked in.",
)

CHECK_OUT = AttendanceAction(
    metric="check_out",
    verb="Logout",
    erp_extra={"type": "OUT"},
    sent_event="erp_logout_sent",
    failed_event="erp_logout_http_failed",
    crash_event="logout_failed",
    ok_status="logged_out",
    ok_message="Logout marked successfully.",
    already_status="already_checked_out",
    already_time_key="check_out_time_ist_text",
    already_message="Already logged out at {time}",
    already_message_no_time="Already logged out.",
)


def _error(code: int, rid: str, status: str, message: str, best: dict | None = None) -> JSONResponse:
    content = {"status": status, "request_id": rid}
    if best is not None:
        content["best"] = best
    content["message"] = message
    return JSONResponse(status_code=code, content=content)


def _retry_reason(status, resp) -> str | None:
    if status is None and isinstance(resp, dict) and resp.get("error_kind") == "connect":
        return "connect"
    if status in getattr(config, "ERP_RETRY_STATUSES", ()):
        return "status_5xx"
    return None


async def post_erp_with_retry(payload: dict, action: AttendanceAction, rid: str):
    """
    POST to the ERP, retrying up to ERP_RETRIES times on connection errors and 502/503/504.
    Not retried: timeouts, other statuses, success. Duplicates are safe: the ERP answers
    a repeated check-in/out with already_checked_in / already_checked_out.
    """
    retries = max(0, int(getattr(config, "ERP_RETRIES", 0)))
    backoff = float(getattr(config, "ERP_RETRY_BACKOFF_S", 0.5))
    attempt = 0
    while True:
        status, resp = await erp_http.post_json(config.ERP_ATTENDANCE_URL, payload)
        reason = _retry_reason(status, resp)
        if reason is None or attempt >= retries:
            return status, resp
        attempt += 1
        metrics.ERP_RETRIES.labels(action.metric, reason).inc()
        log.warning("erp_retry", extra={"request_id": rid, "attempt": attempt, "reason": reason, "status": status})
        await asyncio.sleep(backoff * attempt)


async def recognize_and_report(pipe, rid: str, file: UploadFile, action: AttendanceAction):
    max_bytes = getattr(config, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
    verb_lower = action.verb.lower()

    def outcome(name: str):
        metrics.ATTENDANCE_OUTCOMES.labels(action.metric, name).inc()

    try:
        raw = await read_upload_capped(file, max_bytes)
        if not raw:
            outcome("invalid_upload")
            return _error(400, rid, "error", "Empty image upload")
        if len(raw) > max_bytes:
            outcome("invalid_upload")
            return _error(413, rid, "error", "File too large")

        img = await run_in_threadpool(decode_image, raw)
        if img is None:
            outcome("invalid_upload")
            return _error(400, rid, "error", "Invalid image")

        with metrics.PIPELINE_STAGE_SECONDS.labels("recognize").time():
            preds = await run_in_threadpool(pipe.recognize_image, img)
        if preds is None or len(preds) == 0:
            outcome("no_face")
            return _error(404, rid, "no_face", "No face detected.")

        best = preds[0]
        best_pid = str(best["prediction"]["person_id"])
        best_sim = float(best["prediction"]["similarity"])

        person_name = None
        try:
            if hasattr(pipe, "db") and hasattr(pipe.db, "get_person_name"):
                person_name = pipe.db.get_person_name(best_pid)
        except Exception:
            pass
        best_info = {"person_id": best_pid, "person_name": person_name, "similarity": best_sim}

        def confidence_log(outcome: str) -> None:
            log.info(
                "recognition_confidence",
                extra={
                    "request_id": rid, "person_id": best_pid, "similarity": best_sim,
                    "threshold": config.SIMILARITY_THRESHOLD, "outcome": outcome,
                },
            )

        if best_pid == "unknown":
            confidence_log("no_match")
            outcome("no_match")
            return _error(404, rid, "no_match", f"Face not recognized. {action.verb} not marked.", best_info)

        if not best_pid.isdigit():
            outcome("mapping_error")
            return _error(
                409, rid, "mapping_error",
                f"Recognized user id is not numeric. {action.verb} not marked.", best_info,
            )

        confidence_log("match")
        erp_user_id = int(best_pid)
        payload = {"user_id": erp_user_id, **action.erp_extra}

        erp_start = time.perf_counter()
        try:
            status, resp = await post_erp_with_retry(payload, action, rid)
            metrics.ERP_REQUEST_SECONDS.labels(action.metric).observe(time.perf_counter() - erp_start)
            log.info(
                action.sent_event,
                extra={
                    "request_id": rid, "person_id": best_pid, "erp_user_id": erp_user_id,
                    "similarity": best_sim, "status": status,
                },
            )
        except Exception as e:
            log.exception(action.failed_event, extra={"request_id": rid, "error": str(e)})
            metrics.ERP_FAILURES.labels(action.metric, "exception").inc()
            outcome("erp_exception")
            return _error(502, rid, "error", f"Face recognized but failed to mark {verb_lower} in ERP.", best_info)

        if status not in (200, 201):
            metrics.ERP_FAILURES.labels(action.metric, "status").inc()
            outcome("erp_error")
            return _error(502, rid, "erp_error", "Face recognized but ERP returned an error.", best_info)

        msg = action.ok_message
        if isinstance(resp, dict) and resp.get("status") == action.already_status:
            earlier = resp.get("attendance", {}).get(action.already_time_key)
            msg = action.already_message.format(time=earlier) if earlier else action.already_message_no_time

        outcome("already" if msg != action.ok_message else action.ok_status)
        return {
            "status": action.ok_status,
            "request_id": rid,
            "user_id": erp_user_id,
            "person_id": best_pid,
            "person_name": person_name,
            "similarity": best_sim,
            "message": msg,
        }

    except Exception:
        outcome("error")
        log.exception(action.crash_event, extra={"request_id": rid})
        return JSONResponse(
            status_code=500,
            content={"status": "error", "request_id": rid, "message": "Internal server error"},
        )
