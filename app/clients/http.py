# app/clients/http.py
import asyncio

import aiohttp
from app.logging_config import get_logger

log = get_logger(__name__)

async def post_json(url: str, payload: dict, timeout: int = 8):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    data = {"raw": text}
                # Log status and response keys only; ERP bodies can contain personal data.
                log.info(
                    "http_post",
                    extra={
                        "url": url,
                        "status": resp.status,
                        "response_keys": sorted(data) if isinstance(data, dict) else type(data).__name__,
                    },
                )
                return resp.status, data
    except Exception as e:
        log.exception("http_post_failed", extra={"url": url, "error": str(e)})
        return None, {"error": str(e), "error_kind": error_kind(e)}


def error_kind(e: BaseException) -> str:
    """'timeout' (ERP may still be processing), 'connect' (request likely never handled) or 'other'."""
    if isinstance(e, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
        return "timeout"
    if isinstance(e, aiohttp.ClientConnectionError):
        return "connect"
    return "other"
