import pytest
import httpx
import numpy as np


def make_async_client(app):
    # Do not raise exceptions into pytest; assert on HTTP response instead.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def dummy_img():
    return np.zeros((16, 16, 3), dtype=np.uint8)


# ──────────────────────────────
# /mark_attendance tests
# ──────────────────────────────

@pytest.mark.asyncio
async def test_mark_attendance_empty_upload_400(server_module):
    rid = "rid-empty"
    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"", "image/jpeg")},
        )
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert body["message"] == "Empty image upload"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_size_limit_413(server_module, monkeypatch, config_module, dummy_img):
    rid = "rid-413"
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
    assert body["message"] == "File too large"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_invalid_image_returns_400(server_module, monkeypatch):
    rid = "rid-badimg"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: None)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert body["message"] == "Invalid image"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_no_face_404_when_preds_none(server_module, monkeypatch, dummy_img):
    rid = "rid-noface"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    # conftest DummyPipeline.recognize_image returns None => no_face
    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "no_face"
    assert body["message"] == "No face detected."
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_numpy_empty_preds_404(server_module, monkeypatch, dummy_img):
    rid = "rid-np-empty"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)

    class PipeNPEmpty:
        def recognize_image(self, img):
            return np.array([])

    server_module.PIPE = PipeNPEmpty()

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
async def test_mark_attendance_preds_dict_is_wrapped_and_succeeds(server_module, monkeypatch, config_module, dummy_img):
    """
    Server wraps non-list preds into list; should still work.
    """
    rid = "rid-dict"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ERP_ATTENDANCE_URL", "http://erp/attendance", raising=False)

    class Pipe:
        def recognize_image(self, img):
            return {"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}

    server_module.PIPE = Pipe()

    import app.clients.http as http_client

    async def ok_post_json(url, payload):
        return 200, {"ok": True}

    monkeypatch.setattr(http_client, "post_json", ok_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 200
    assert r.json()["status"] == "marked"
    assert r.json()["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_no_match_below_maybe_threshold(server_module, monkeypatch, config_module, dummy_img):
    rid = "rid-nm"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.50}, "top_k": []}]

    server_module.PIPE = Pipe()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "no_match"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_low_confidence_between_thresholds(server_module, monkeypatch, config_module, dummy_img):
    rid = "rid-lc"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{
                "prediction": {"person_id": "72", "similarity": 0.70},
                "top_k": [{"person_id": "72", "similarity": 0.70}],
            }]

    server_module.PIPE = Pipe()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "low_confidence"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_mapping_error_when_label_not_in_map(server_module, monkeypatch, config_module, dummy_img):
    """
    High confidence, non-digit person_id => requires ERP_USER_MAP mapping.
    """
    rid = "rid-map"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_USER_MAP", {}, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "someone", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = Pipe()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "mapping_error"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_erp_http_exception_returns_502(server_module, monkeypatch, config_module, dummy_img):
    """
    ERP call raises exception => 502.
    """
    rid = "rid-erp-boom"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_ATTENDANCE_URL", "http://erp/attendance", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = Pipe()

    import app.clients.http as http_client

    async def boom_post_json(url, payload):
        raise RuntimeError("network down")

    monkeypatch.setattr(http_client, "post_json", boom_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "error"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_erp_non_200_returns_502(server_module, monkeypatch, config_module, dummy_img):
    rid = "rid-erp-500"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_ATTENDANCE_URL", "http://erp/attendance", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = Pipe()

    import app.clients.http as http_client

    async def fake_post_json(url, payload):
        return 500, {"ok": False}

    monkeypatch.setattr(http_client, "post_json", fake_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "erp_error"
    assert body["request_id"] == rid


@pytest.mark.asyncio
async def test_mark_attendance_success_200_includes_person_name(server_module, monkeypatch, config_module, dummy_img):
    """
    Success path: high confidence, ERP returns 200, and response includes person_name.
    We stub PIPE.db.get_person_name to return a name for the predicted person_id.
    """
    rid = "rid-ok"
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_ATTENDANCE_URL", "http://erp/attendance", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class PipeHigh:
        def recognize_image(self, img):
            return [{
                "prediction": {"person_id": "72", "similarity": 0.99},
                "top_k": [{"person_id": "72", "similarity": 0.99}],
            }]

    class DBWithName:
        def get_person_name(self, person_id: str):
            return "Rhythm"

    server_module.PIPE = PipeHigh()
    server_module.PIPE.db = DBWithName()

    import app.clients.http as http_client

    async def ok_post_json(url, payload):
        return 200, {"ok": True}

    monkeypatch.setattr(http_client, "post_json", ok_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post(
            "/mark_attendance",
            headers={"X-Request-ID": rid},
            files={"file": ("x.jpg", b"abc", "image/jpeg")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "marked"
    assert body["request_id"] == rid
    assert body["person_id"] == "72"
    assert body["person_name"] == "Rhythm"
    assert "user_id" in body


# ──────────────────────────────
# /logout tests
# ──────────────────────────────

@pytest.mark.asyncio
async def test_logout_empty_upload_400(server_module):
    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert body["message"] == "Empty image upload"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_size_limit_413(server_module, monkeypatch, config_module):
    monkeypatch.setattr(config_module, "MAX_UPLOAD_BYTES", 3, raising=False)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"1234", "image/jpeg")})

    assert r.status_code == 413
    body = r.json()
    assert body["status"] == "error"
    assert body["message"] == "File too large"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_invalid_image_returns_400(server_module, monkeypatch):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: None)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 400
    body = r.json()
    assert body["status"] == "error"
    assert body["message"] == "Invalid image"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_no_match_404(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)

    class PipeNoMatch:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "unknown", "similarity": 0.10}, "top_k": []}]

    server_module.PIPE = PipeNoMatch()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 404
    body = r.json()
    assert body["status"] == "no_match"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_low_confidence_409(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)

    class PipeLowConf:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.70}, "top_k": [{"person_id": "72", "similarity": 0.70}]}]

    server_module.PIPE = PipeLowConf()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "low_confidence"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_mapping_error_409(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_USER_MAP", {}, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)

    class PipeHighConfNonDigit:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "rhythm", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = PipeHighConfNonDigit()

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "mapping_error"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_erp_http_exception_returns_502(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_LOGOUT_URL", "http://erp/logout", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)

    class PipeHighConf:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = PipeHighConf()

    import app.clients.http as http_client

    async def boom_post_json(url, payload):
        raise RuntimeError("network down")

    monkeypatch.setattr(http_client, "post_json", boom_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "error"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_erp_non_200_returns_502(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_LOGOUT_URL", "http://erp/logout", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class Pipe:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = Pipe()

    import app.clients.http as http_client

    async def fake_post_json(url, payload):
        return 500, {"ok": False}

    monkeypatch.setattr(http_client, "post_json", fake_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "erp_error"
    assert "request_id" in body


@pytest.mark.asyncio
async def test_logout_success_logged_out_200(server_module, monkeypatch, config_module, dummy_img):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy_img)
    monkeypatch.setattr(config_module, "ERP_LOGOUT_URL", "http://erp/logout", raising=False)
    monkeypatch.setattr(config_module, "ATTEND_AUTO_THRESHOLD", 0.75, raising=False)
    monkeypatch.setattr(config_module, "ATTEND_MAYBE_THRESHOLD", 0.60, raising=False)

    class PipeHighConf:
        def recognize_image(self, img):
            return [{"prediction": {"person_id": "72", "similarity": 0.99}, "top_k": []}]

    server_module.PIPE = PipeHighConf()

    import app.clients.http as http_client

    async def ok_post_json(url, payload):
        return 200, {"ok": True}

    monkeypatch.setattr(http_client, "post_json", ok_post_json)

    async with make_async_client(server_module.app) as ac:
        r = await ac.post("/logout", files={"file": ("x.jpg", b"abc", "image/jpeg")})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "logged_out"
    assert body["user_id"] == 72
    assert "request_id" in body
