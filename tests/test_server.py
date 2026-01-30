import numpy as np
import pytest
import httpx


def make_async_client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def dummy_img():
    return np.zeros((16, 16, 3), dtype=np.uint8)


@pytest.mark.asyncio
async def test_root_includes_request_id(server_module):
    rid = "rid-123"
    async with make_async_client(server_module.app) as ac:
        r = await ac.get("/", headers={"X-Request-ID": rid})
        assert r.status_code == 200
        body = r.json()
        assert body["request_id"] == rid
        assert "try" in body


@pytest.mark.asyncio
async def test_validation_422_includes_request_id(server_module):
    rid = "rid-422"
    async with make_async_client(server_module.app) as ac:
        # /register requires user_id
        r = await ac.post(
            "/register",
            headers={"X-Request-ID": rid},
            data={"name": "x"},
            files=[("files", ("x.jpg", b"abc", "image/jpeg"))],
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "validation_error"
        assert body["request_id"] == rid
        assert "details" in body


@pytest.mark.asyncio
async def test_healthz_gated_by_default(server_module, monkeypatch):
    # default is HEALTHZ_DETAILS=0
    monkeypatch.delenv("HEALTHZ_DETAILS", raising=False)

    async with make_async_client(server_module.app) as ac:
        r = await ac.get("/healthz", headers={"X-Request-ID": "rid-h"})
        assert r.status_code == 200
        body = r.json()

        assert body["status"] == "ok"
        assert body["request_id"] == "rid-h"

        # These should be gated OFF by default
        assert "firebase_error" not in body
        assert "env_seen" not in body
        assert "people" not in body


@pytest.mark.asyncio
async def test_healthz_details_enabled(server_module, monkeypatch):
    monkeypatch.setenv("HEALTHZ_DETAILS", "1")
    server_module.firebase_error = "RuntimeError: boom"

    async with make_async_client(server_module.app) as ac:
        r = await ac.get("/healthz", headers={"X-Request-ID": "rid-hd"})
        assert r.status_code == 200
        body = r.json()

        assert body["status"] == "ok"
        assert body["request_id"] == "rid-hd"
        assert body["firebase_error"] == "RuntimeError: boom"
        assert "env_seen" in body
        assert "people" in body


@pytest.mark.asyncio
async def test_register_returns_400_when_no_valid_images_enrolled(server_module, monkeypatch, dummy_img):
    """
    Force decode failure => processed=0 => should return 400 with status:error.
    """
    rid = "rid-reg-400"

    # Make cv2.imdecode return None to simulate bad image
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: None)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/register",
            headers={"X-Request-ID": rid},
            data={"user_id": "u1", "name": "Test"},
            files=[("files", ("img.jpg", b"not-a-real-image", "image/jpeg"))],
        )
        assert r.status_code == 400
        body = r.json()
        assert body["status"] == "error"
        assert body["request_id"] == rid
        assert body["message"] == "No valid images were enrolled."
        assert "details" in body


@pytest.mark.asyncio
async def test_register_success_200(server_module, monkeypatch, dummy_img):
    """
    Decode ok + default DummyPipeline enroll_image adds embeddings => 200.
    """
    rid = "rid-reg-200"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/register",
            headers={"X-Request-ID": rid},
            data={"user_id": "u1", "name": "Test"},
            files=[("files", ("img.jpg", b"fakebytes", "image/jpeg"))],
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_size_limit_413(server_module, monkeypatch, config_module, dummy_img):
    rid = "rid-att-413"

    monkeypatch.setattr(config_module, "MAX_UPLOAD_BYTES", 3, raising=False)
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"1234", "image/jpeg")},
        )
        assert r.status_code == 413
        body = r.json()
        assert body["status"] == "error"
        assert body["request_id"] == rid
        assert body["message"] == "File too large"


@pytest.mark.asyncio
async def test_mark_attendance_no_face_404(server_module, monkeypatch, dummy_img):
    rid = "rid-att-404"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    # Default DummyPipeline.recognize_image returns None => no_face
    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )
        assert r.status_code == 404
        body = r.json()
        assert body["status"] == "no_face"
        assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_logout_no_face_404(server_module, monkeypatch, dummy_img):
    rid = "rid-out-404"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/logout",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )
        assert r.status_code == 404
        body = r.json()
        assert body["status"] == "no_face"
        assert body["request_id"] == rid
