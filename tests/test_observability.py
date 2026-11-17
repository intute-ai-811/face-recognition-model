"""Metrics, readiness and JSON logging (PR 10)."""
import json
import logging
import re

import httpx
import numpy as np
import pytest

IMG = ("x.jpg", b"abc", "image/jpeg")


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test")


def sample(text: str, name: str, **labels) -> float:
    """Value of one Prometheus sample, 0 if absent."""
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    for line in text.splitlines():
        if line.startswith(f"{name}{{{want}}} ") or (not labels and line.startswith(f"{name} ")):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


@pytest.fixture
def match(server_module, monkeypatch):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: np.zeros((16, 16, 3), np.uint8))

    class Pipe:
        db = server_module.PIPE.db

        def recognize_image(self, img):
            return [{"prediction": {"person_id": "4242", "similarity": 0.8}, "top_k": []}]

    server_module.PIPE = Pipe()
    import app.clients.http as http_client

    async def ok(url, payload):
        return 200, {"ok": True}

    monkeypatch.setattr(http_client, "post_json", ok)


@pytest.mark.asyncio
async def test_metrics_count_outcomes_and_routes(server_module, match):
    async with client(server_module.app) as ac:
        before = (await ac.get("/metrics")).text
        await ac.post("/mark_attendance", files={"file": IMG})
        await ac.post("/logout", files={"file": IMG})
        await ac.delete("/admin/person/4242")  # 503/401, but must be labelled by template
        after = (await ac.get("/metrics")).text

    def delta(name, **labels):
        return sample(after, name, **labels) - sample(before, name, **labels)

    assert delta("face_api_attendance_outcomes_total", action="check_in", outcome="marked") == 1
    assert delta("face_api_attendance_outcomes_total", action="check_out", outcome="logged_out") == 1
    assert delta("face_api_http_requests_total", method="POST", route="/mark_attendance", status="200") == 1
    assert "face_api_erp_request_seconds_count" in after
    assert 'route="/admin/person/{person_id}"' in after


@pytest.mark.asyncio
async def test_metrics_never_contain_person_ids(server_module, match):
    async with client(server_module.app) as ac:
        await ac.post("/mark_attendance", files={"file": IMG})
        await ac.delete("/admin/person/4242")
        text = (await ac.get("/metrics")).text
    label_values = re.findall(r'="([^"]*)"', text)
    assert not any("4242" in v for v in label_values)


@pytest.mark.asyncio
async def test_unknown_paths_are_one_label(server_module):
    async with client(server_module.app) as ac:
        await ac.get("/does-not-exist/123")
        text = (await ac.get("/metrics")).text
    assert 'route="unmatched"' in text and "does-not-exist" not in text


@pytest.mark.asyncio
async def test_metrics_token(server_module, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "scrape-secret")
    async with client(server_module.app) as ac:
        assert (await ac.get("/metrics")).status_code == 401
        assert (await ac.get("/metrics", headers={"Authorization": "Bearer nope"})).status_code == 401
        ok = await ac.get("/metrics", headers={"Authorization": "Bearer scrape-secret"})
    assert ok.status_code == 200 and "face_api_http_requests_total" in ok.text


@pytest.mark.asyncio
async def test_readyz_ready(server_module):
    server_module.PIPE.db.stats = lambda: {"embeddings": 7, "people": 2}
    async with client(server_module.app) as ac:
        r = await ac.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready" and r.json()["checks"] == {"store": True, "embeddings": 7}


@pytest.mark.asyncio
async def test_readyz_not_ready_when_store_fails(server_module):
    def broken():
        raise OSError("db locked")

    server_module.PIPE.db.stats = broken
    async with client(server_module.app) as ac:
        r = await ac.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


def test_store_stats_gauges_read_without_logging(server_module, caplog):
    server_module.PIPE.db.stats = lambda: {"embeddings": 11, "people": 3}
    from app.observability import metrics
    body, _ = metrics.render()
    text = body.decode()
    assert sample(text, "face_api_store_embeddings") == 11
    assert sample(text, "face_api_store_people") == 3


def test_json_log_format(monkeypatch, capsys):
    from app.logging_config import JSONFormatter

    rec = logging.LogRecord("app.test", logging.INFO, __file__, 1, "erp_attendance_sent", None, None)
    rec.request_id, rec.status = "rid-1", 200
    line = JSONFormatter().format(rec)
    obj = json.loads(line)
    assert obj["msg"] == "erp_attendance_sent" and obj["request_id"] == "rid-1" and obj["status"] == 200
    assert obj["level"] == "INFO" and obj["logger"] == "app.test" and "ts" in obj

    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec2 = logging.LogRecord("app.test", logging.ERROR, __file__, 1, "x", None, sys.exc_info())
    assert "ValueError: boom" in json.loads(JSONFormatter().format(rec2))["exc"]


def test_setup_logging_selects_format(monkeypatch):
    from app import logging_config

    monkeypatch.setenv("LOG_FORMAT", "json")
    logging_config.setup_logging()
    assert isinstance(logging_config._handler.formatter, logging_config.JSONFormatter)
    monkeypatch.setenv("LOG_FORMAT", "kv")
    logging_config.setup_logging()
    assert isinstance(logging_config._handler.formatter, logging_config.KVFormatter)
