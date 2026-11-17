# app/api/uploads.py
"""Request helpers: request IDs, capped upload reads, safe image decoding."""
from __future__ import annotations

import io
import logging
import uuid

import cv2
import numpy as np
from fastapi import UploadFile
from starlette.requests import Request

from app import config

log = logging.getLogger(__name__)


def request_id(request: Request | None) -> str:
    if request is None:
        return str(uuid.uuid4())
    return request.headers.get("x-request-id") or str(uuid.uuid4())


async def read_upload_capped(f: UploadFile, max_bytes: int) -> bytes:
    """Read at most max_bytes + 1 bytes, so an oversized file is detected without loading it whole."""
    try:
        return await f.read(max_bytes + 1)
    finally:
        await f.close()


def decode_image(raw: bytes):
    """
    Decode with a pixel cap. The header is read first (cheap) so a small file that
    expands to a huge image is rejected before allocation. Returns None on failure,
    which callers map to their existing "invalid image" responses.
    """
    max_pixels = getattr(config, "MAX_IMAGE_PIXELS", 50_000_000)
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
        if w * h > max_pixels:
            log.warning("image_too_many_pixels", extra={"w": w, "h": h})
            return None
    except Exception:
        pass  # Not a format Pillow knows; OpenCV decides, then the post-decode check applies.
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is not None and img.shape[0] * img.shape[1] > max_pixels:
        log.warning("image_too_many_pixels", extra={"w": img.shape[1], "h": img.shape[0]})
        return None
    return img
