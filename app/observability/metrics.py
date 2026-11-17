# app/observability/metrics.py
"""
Prometheus metrics. Labels never carry person IDs, names, images or embeddings:
only route templates, HTTP status codes, fixed outcome names and stage names.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Request-level
HTTP_REQUESTS = Counter(
    "face_api_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "face_api_http_request_seconds", "HTTP request latency", ["route"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# Recognition / attendance
ATTENDANCE_OUTCOMES = Counter(
    "face_api_attendance_outcomes_total",
    "Result of /mark_attendance and /logout",
    ["action", "outcome"],  # action: check_in|check_out; outcome: fixed set, see services/attendance.py
)
PIPELINE_STAGE_SECONDS = Histogram(
    "face_api_pipeline_stage_seconds", "Model inference time per stage", ["stage"],  # detect|embed|recognize
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# ERP
ERP_REQUEST_SECONDS = Histogram(
    "face_api_erp_request_seconds", "ERP call latency", ["action"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8),
)
ERP_FAILURES = Counter(
    "face_api_erp_failures_total", "ERP calls that raised or returned non-2xx", ["action", "kind"]  # kind: exception|status
)

# Enrollment
ENROLLMENT_OUTCOMES = Counter(
    "face_api_enrollment_outcomes_total", "Result of /register", ["outcome"]  # enrolled|insufficient|no_valid_images|rejected
)
ENROLLMENT_IMAGES = Counter(
    "face_api_enrollment_images_total", "Enrollment images by result", ["result"]  # embedded|no_embedding|empty|too_large|undecodable|error
)

# Store (set via callbacks at scrape time)
STORE_EMBEDDINGS = Gauge("face_api_store_embeddings", "Enrolled embeddings for the active model")
STORE_PEOPLE = Gauge("face_api_store_people", "Enrolled people for the active model")


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
