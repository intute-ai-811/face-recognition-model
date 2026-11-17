"""ERP retry on transient failures (PR 11). Kiosk responses are unchanged; only fewer 502s."""
import asyncio
import socket

import aiohttp
import httpx
import numpy as np
import pytest

from app import config

IMG = ("x.jpg", b"abc", "image/jpeg")
MATCH = [{"prediction": {"person_id": "72", "similarity": 0.8}, "top_k": []}]
CONNECT_FAIL = (None, {"error": "Cannot connect", "error_kind": "connect"})
TIMEOUT_FAIL = (None, {"error": "", "error_kind": "timeout"})


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test")


@pytest.fixture
def erp(server_module, monkeypatch):
    """erp(*responses): the fake ERP returns these in order (last one repeats). Returns the call log."""
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: np.zeros((16, 16, 3), np.uint8))
    monkeypatch.setattr(config, "ERP_RETRY_BACKOFF_S", 0.0)

    class Pipe:
        db = server_module.PIPE.db

        def recognize_image(self, img):
            return MATCH

    server_module.PIPE = Pipe()
    calls = []

    def use(*responses):
        import app.clients.http as http_client

        async def fake(url, payload):
            calls.append(payload)
            return responses[min(len(calls), len(responses)) - 1]

        monkeypatch.setattr(http_client, "post_json", fake)
        return calls

    return use


async def post(server_module, path="/mark_attendance"):
    async with client(server_module.app) as ac:
        return await ac.post(path, files={"file": IMG})


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [CONNECT_FAIL, (503, {}), (502, {"raw": "Bad Gateway"}), (504, {})])
async def test_transient_failure_is_retried_once_and_succeeds(server_module, erp, first):
    calls = erp(first, (200, {"ok": True}))
    r = await post(server_module)
    assert r.status_code == 200 and r.json()["message"] == "Attendance marked successfully."
    assert calls == [{"user_id": 72}, {"user_id": 72}]


@pytest.mark.asyncio
async def test_retry_reaching_already_recorded_reports_already(server_module, erp):
    # First attempt was recorded by the ERP but the reply was lost: the retry is harmless.
    calls = erp((503, {}), (200, {"status": "already_checked_out", "attendance": {"check_out_time_ist_text": "18:02"}}))
    r = await post(server_module, "/logout")
    assert r.status_code == 200 and r.json()["message"] == "Already logged out at 18:02"
    assert calls == [{"user_id": 72, "type": "OUT"}] * 2


@pytest.mark.asyncio
async def test_still_failing_after_retry_gives_same_502_as_before(server_module, erp):
    calls = erp((503, {}))
    r = await post(server_module)
    assert len(calls) == 2
    assert r.status_code == 502
    assert r.json() == {
        "status": "erp_error", "request_id": r.json()["request_id"],
        "best": {"person_id": "72", "person_name": None, "similarity": 0.8},
        "message": "Face recognized but ERP returned an error.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [TIMEOUT_FAIL, (None, {"error": "x", "error_kind": "other"}), (None, {"error": "x"}),
                                  (500, {}), (400, {}), (404, {}), (200, {"ok": True}), (201, {})])
async def test_not_retried(server_module, erp, resp):
    calls = erp(resp)
    await post(server_module)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retries_disabled(server_module, erp, monkeypatch):
    monkeypatch.setattr(config, "ERP_RETRIES", 0)
    calls = erp(CONNECT_FAIL, (200, {"ok": True}))
    r = await post(server_module)
    assert len(calls) == 1 and r.status_code == 502


@pytest.mark.asyncio
async def test_retry_is_counted_in_metrics(server_module, erp):
    erp(CONNECT_FAIL, (200, {"ok": True}))
    await post(server_module)
    async with client(server_module.app) as ac:
        text = (await ac.get("/metrics")).text
    assert 'face_api_erp_retries_total{action="check_in",reason="connect"}' in text


def test_error_kind_classification():
    from app.clients.http import error_kind
    assert error_kind(asyncio.TimeoutError()) == "timeout"
    assert error_kind(aiohttp.ServerTimeoutError()) == "timeout"
    assert error_kind(aiohttp.ServerDisconnectedError()) == "connect"
    assert error_kind(aiohttp.ClientConnectionError()) == "connect"
    assert error_kind(ValueError()) == "other"


@pytest.mark.asyncio
async def test_real_client_reports_connect_for_refused_port():
    from app.clients import http as http_client
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    status, data = await http_client.post_json(f"http://127.0.0.1:{port}/x", {"user_id": 1}, timeout=2)
    assert status is None and data["error_kind"] == "connect"
