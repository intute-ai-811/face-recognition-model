# app/api/security.py
"""Token checks for admin and enrollment endpoints."""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException


def token_matches(given: str | None, expected: str | None) -> bool:
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode(), expected.encode())


def require_admin(x_admin_token: str | None) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled (ADMIN_TOKEN not set).")
    if not token_matches(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_enroll_key(x_enroll_token: str | None, x_admin_token: str | None) -> None:
    """
    Enrollment authentication, off by default (ENROLL_REQUIRE_KEY=0) so existing
    kiosk clients keep working. When enabled, /register needs X-Enroll-Token
    matching ENROLL_TOKEN, or X-Admin-Token matching ADMIN_TOKEN.
    """
    if os.getenv("ENROLL_REQUIRE_KEY", "0") != "1":
        return
    enroll_token = os.getenv("ENROLL_TOKEN")
    admin_token = os.getenv("ADMIN_TOKEN")
    if not enroll_token and not admin_token:
        raise HTTPException(
            status_code=503,
            detail="Enrollment disabled (ENROLL_REQUIRE_KEY=1 but no ENROLL_TOKEN/ADMIN_TOKEN set).",
        )
    if token_matches(x_enroll_token, enroll_token) or token_matches(x_admin_token, admin_token):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")
