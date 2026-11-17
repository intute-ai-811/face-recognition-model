"""Security quick wins (PR 3)."""
import logging

import cv2
import httpx
import numpy as np
import pytest

IMG = ("x.jpg", b"abc", "image/jpeg")


def client(app):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def img(server_module, monkeypatch):
    dummy = np.zeros((16, 16, 3), np.uint8)
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy)
    return dummy


def register_files(n):
    return [("files", (f"{i}.jpg", b"abc", "image/jpeg")) for i in range(n)]


# ── enrollment key ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_open_by_default(server_module, img, monkeypatch):
    monkeypatch.delenv("ENROLL_REQUIRE_KEY", raising=False)
    from app import config
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70"}, files=register_files(config.ENROLL_MIN_GOOD_EMBEDDINGS))
    assert r.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,code",
    [
        ({}, 401),
        ({"X-Enroll-Token": "wrong"}, 401),
        ({"X-Admin-Token": "wrong"}, 401),
        ({"X-Enroll-Token": "kiosk-secret"}, 200),
        ({"X-Admin-Token": "admin-secret"}, 200),
    ],
)
async def test_register_requires_key_when_enabled(server_module, img, monkeypatch, headers, code):
    from app import config
    monkeypatch.setenv("ENROLL_REQUIRE_KEY", "1")
    monkeypatch.setenv("ENROLL_TOKEN", "kiosk-secret")
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    async with client(server_module.app) as ac:
        r = await ac.post(
            "/register", headers=headers, data={"user_id": "70"},
            files=register_files(config.ENROLL_MIN_GOOD_EMBEDDINGS),
        )
    assert r.status_code == code


@pytest.mark.asyncio
async def test_register_rejected_before_any_enrollment(server_module, img, monkeypatch):
    monkeypatch.setenv("ENROLL_REQUIRE_KEY", "1")
    monkeypatch.setenv("ENROLL_TOKEN", "kiosk-secret")
    calls = []
    server_module.PIPE.enroll_image = lambda *a, **k: calls.append(1) or {"faces": 1, "embeddings_added": 1}
    server_module.PIPE.db.set_person = lambda *a: calls.append("name")
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70", "name": "X"}, files=register_files(5))
    assert r.status_code == 401
    assert calls == []


@pytest.mark.asyncio
async def test_register_key_enabled_without_tokens_is_503(server_module, img, monkeypatch):
    monkeypatch.setenv("ENROLL_REQUIRE_KEY", "1")
    monkeypatch.delenv("ENROLL_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70"}, files=register_files(1))
    assert r.status_code == 503


def test_token_compare_is_constant_time_helper(server_module):
    assert server_module._token_matches("abc", "abc")
    assert not server_module._token_matches("abc", "abd")
    assert not server_module._token_matches(None, "abc")
    assert not server_module._token_matches("abc", None)
    assert not server_module._token_matches("", "")


# ── debug endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_debug_firebase_requires_admin(server_module, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    async with client(server_module.app) as ac:
        r = await ac.post("/debug/firebase_test", data={"user_id": "70"})
        r2 = await ac.post("/debug/firebase_test", data={"user_id": "70"}, headers={"X-Admin-Token": "bad"})
    assert r.status_code == 401 and r2.status_code == 401


@pytest.mark.asyncio
async def test_debug_firebase_disabled_without_admin_token(server_module, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    async with client(server_module.app) as ac:
        r = await ac.post("/debug/firebase_test", data={"user_id": "70"})
    assert r.status_code == 503


# ── firebase objects stay private ────────────────────────────────────────────

def test_firebase_upload_never_makes_objects_public(server_module, monkeypatch):
    class Blob:
        public_calls = 0

        def upload_from_file(self, f, content_type):
            self.data = f.read()

        def make_public(self):
            Blob.public_calls += 1

    class Bucket:
        name = "bucket"

        def blob(self, path):
            return Blob()

    monkeypatch.setattr(server_module, "init_firebase_once", lambda: None)
    monkeypatch.setattr(server_module, "firebase_bucket", Bucket())
    gs_url, public_url = server_module.firebase_upload_bytes("users/70/debug", "a.jpg", b"x")
    assert gs_url == "gs://bucket/users/70/debug/a.jpg"
    assert public_url == ""
    assert Blob.public_calls == 0


# ── upload limits ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_over_content_length_cap_is_413(server_module, monkeypatch, config_module):
    monkeypatch.setattr(config_module, "MAX_REQUEST_BYTES", 100, raising=False)
    async with client(server_module.app) as ac:
        r = await ac.post("/mark_attendance", files={"file": ("x.jpg", b"0" * 500, "image/jpeg")})
    assert r.status_code == 413
    assert set(r.json()) == {"status", "request_id", "message"}
    assert r.json()["message"] == "Request too large"


@pytest.mark.asyncio
async def test_oversized_file_reads_only_cap_plus_one(server_module, monkeypatch):
    seen = []

    class FakeUpload:
        async def read(self, size=-1):
            seen.append(size)
            return b"0" * (10 if size < 0 else min(size, 10))

        async def close(self):
            pass

    raw = await server_module._read_upload_capped(FakeUpload(), 4)
    assert seen == [5]
    assert len(raw) == 5  # > cap, so callers return 413 as before


def _png(h, w):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_decode_rejects_images_over_pixel_cap(server_module, monkeypatch, config_module):
    monkeypatch.setattr(config_module, "MAX_IMAGE_PIXELS", 10_000, raising=False)
    assert server_module._decode_image(_png(200, 200)) is None   # 40k px
    img = server_module._decode_image(_png(50, 50))              # 2.5k px
    assert img is not None and img.shape == (50, 50, 3)


def test_decode_invalid_bytes_returns_none(server_module):
    assert server_module._decode_image(b"not an image") is None


@pytest.mark.asyncio
async def test_huge_image_gets_existing_invalid_image_response(server_module, monkeypatch, config_module):
    monkeypatch.setattr(config_module, "MAX_IMAGE_PIXELS", 10_000, raising=False)
    async with client(server_module.app) as ac:
        r = await ac.post("/mark_attendance", files={"file": ("x.png", _png(200, 200), "image/png")})
    assert r.status_code == 400
    assert r.json()["message"] == "Invalid image"


# ── log redaction ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_logs_do_not_contain_person_name(server_module, img, caplog):
    caplog.set_level(logging.INFO)
    async with client(server_module.app) as ac:
        await ac.post("/register", data={"user_id": "70", "name": "Zebulon Quarry"}, files=register_files(5))
    for rec in caplog.records:
        assert "Zebulon Quarry" not in str(rec.__dict__)


@pytest.mark.asyncio
async def test_erp_client_logs_only_status_and_keys(monkeypatch, caplog):
    import app.clients.http as http_client

    class Resp:
        status = 200

        async def text(self):
            return '{"name": "Zebulon Quarry"}'

        async def json(self):
            return {"name": "Zebulon Quarry"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Session:
        def post(self, url, json, timeout):
            return Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(http_client.aiohttp, "ClientSession", Session)
    caplog.set_level(logging.INFO)
    status, data = await http_client.post_json("http://erp", {"user_id": 1})
    assert status == 200 and data == {"name": "Zebulon Quarry"}
    for rec in caplog.records:
        assert "Zebulon Quarry" not in str(rec.__dict__)


# ── docker build context ─────────────────────────────────────────────────────

def test_dockerignore_is_an_allow_list():
    from pathlib import Path
    lines = [
        ln.strip() for ln in (Path(__file__).resolve().parents[1] / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines[0] == "*"
    assert set(ln for ln in lines if ln.startswith("!")) == {"!app", "!pyproject.toml", "!uv.lock", "!.python-version", "!scripts/migrate_pickle_to_sqlite.py"}
