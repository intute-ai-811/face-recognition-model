"""
API contract snapshots (PR 2).

Freezes the status code and the exact set of top-level response keys for every
endpoint outcome, so later refactors (PR 8+) cannot change the API silently.
If one of these tests fails, the API changed: that needs explicit approval
before the snapshot is updated.
"""
import httpx
import numpy as np
import pytest

IMG = ("x.jpg", b"abc", "image/jpeg")
BEST = {"person_id", "person_name", "similarity"}


def client(app):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def img(server_module, monkeypatch):
    dummy = np.zeros((16, 16, 3), np.uint8)
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: dummy)
    return dummy


def pipe_returning(server_module, preds):
    class Pipe:
        db = server_module.PIPE.db

        def recognize_image(self, image):
            return preds

    server_module.PIPE = Pipe()


def erp_returning(monkeypatch, status, resp=None, exc=None):
    import app.clients.http as http_client

    async def fake_post_json(url, payload):
        if exc:
            raise exc
        return status, resp if resp is not None else {"ok": True}

    monkeypatch.setattr(http_client, "post_json", fake_post_json)


MATCH = [{"prediction": {"person_id": "72", "similarity": 0.8}, "top_k": []}]
UNKNOWN = [{"prediction": {"person_id": "unknown", "similarity": 0.3}, "top_k": []}]
NON_NUMERIC = [{"prediction": {"person_id": "abc", "similarity": 0.8}, "top_k": []}]


# ── GET / and /healthz ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_contract(server_module):
    async with client(server_module.app) as ac:
        r = await ac.get("/")
    assert r.status_code == 200
    assert set(r.json()) == {"message", "request_id", "try"}


@pytest.mark.asyncio
async def test_healthz_contract_default(server_module, monkeypatch):
    monkeypatch.delenv("HEALTHZ_DETAILS", raising=False)
    async with client(server_module.app) as ac:
        r = await ac.get("/healthz")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "request_id", "firebase_ready", "firebase_bucket"}


# ── POST /register ───────────────────────────────────────────────────────────

REGISTER_SUMMARY = {
    "ok", "request_id", "person_id", "person_name", "received_files", "processed_files",
    "errors", "faces_detected", "embeddings_added", "uploaded_debug",
}


@pytest.mark.asyncio
async def test_register_success_contract(server_module, img):
    from app import config
    files = [("files", (f"{i}.jpg", b"abc", "image/jpeg")) for i in range(config.ENROLL_MIN_GOOD_EMBEDDINGS)]
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70", "name": "A"}, files=files)
    assert r.status_code == 200
    assert set(r.json()) == REGISTER_SUMMARY


@pytest.mark.asyncio
async def test_register_no_valid_images_contract(server_module, monkeypatch):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: None)
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70"}, files=[("files", IMG)])
    assert r.status_code == 400
    body = r.json()
    assert set(body) == {"status", "request_id", "message", "details"}
    assert body["status"] == "error"
    assert set(body["details"]) == REGISTER_SUMMARY


@pytest.mark.asyncio
async def test_register_insufficient_contract(server_module, img):
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "70"}, files=[("files", IMG)])
    assert r.status_code == 400
    body = r.json()
    assert set(body) == {"status", "request_id", "embeddings_added", "message", "details"}
    assert body["status"] == "insufficient_enrollment"


@pytest.mark.asyncio
async def test_register_blank_user_id_contract(server_module, img):
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"user_id": "  "}, files=[("files", IMG)])
    assert r.status_code == 400
    assert set(r.json()) == {"status", "request_id", "message"}


@pytest.mark.asyncio
async def test_validation_error_contract(server_module):
    async with client(server_module.app) as ac:
        r = await ac.post("/register", data={"name": "x"}, files=[("files", IMG)])
    assert r.status_code == 422
    assert set(r.json()) == {"error", "request_id", "details"}


# ── POST /mark_attendance and /logout ────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,ok_status,ok_code",
    [("/mark_attendance", "marked", 200), ("/logout", "logged_out", 200)],
)
async def test_recognition_success_contract(server_module, img, monkeypatch, path, ok_status, ok_code):
    pipe_returning(server_module, MATCH)
    erp_returning(monkeypatch, 200)
    async with client(server_module.app) as ac:
        r = await ac.post(path, files={"file": IMG})
    assert r.status_code == ok_code
    body = r.json()
    assert set(body) == {"status", "request_id", "user_id", "person_id", "person_name", "similarity", "message"}
    assert body["status"] == ok_status


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/mark_attendance", "/logout"])
@pytest.mark.parametrize(
    "case,preds,erp,code,status,keys",
    [
        ("no_face", [], None, 404, "no_face", {"status", "request_id", "message"}),
        ("no_match", UNKNOWN, None, 404, "no_match", {"status", "request_id", "best", "message"}),
        ("mapping", NON_NUMERIC, None, 409, "mapping_error", {"status", "request_id", "best", "message"}),
        ("erp_non_2xx", MATCH, (500, None), 502, "erp_error", {"status", "request_id", "best", "message"}),
        ("erp_raises", MATCH, (None, RuntimeError("down")), 502, "error", {"status", "request_id", "best", "message"}),
    ],
)
async def test_recognition_failure_contracts(server_module, img, monkeypatch, path, case, preds, erp, code, status, keys):
    pipe_returning(server_module, preds)
    if erp:
        erp_status, exc = erp
        erp_returning(monkeypatch, erp_status, exc=exc if isinstance(exc, Exception) else None)
    async with client(server_module.app) as ac:
        r = await ac.post(path, files={"file": IMG})
    assert r.status_code == code
    body = r.json()
    assert set(body) == keys
    assert body["status"] == status
    if "best" in body:
        assert set(body["best"]) == BEST


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/mark_attendance", "/logout"])
async def test_recognition_upload_error_contracts(server_module, monkeypatch, config_module, path):
    async with client(server_module.app) as ac:
        empty = await ac.post(path, files={"file": ("x.jpg", b"", "image/jpeg")})
        monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: None)
        invalid = await ac.post(path, files={"file": IMG})
        monkeypatch.setattr(config_module, "MAX_UPLOAD_BYTES", 2, raising=False)
        too_big = await ac.post(path, files={"file": IMG})
    for r, code in ((empty, 400), (invalid, 400), (too_big, 413)):
        assert r.status_code == code
        assert set(r.json()) == {"status", "request_id", "message"}


# ── Admin ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_disabled_without_token_contract(server_module, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    async with client(server_module.app) as ac:
        r = await ac.get("/admin/people")
    assert r.status_code == 503
    assert set(r.json()) == {"detail"}


@pytest.mark.asyncio
async def test_admin_wrong_token_contract(server_module, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    async with client(server_module.app) as ac:
        r = await ac.get("/admin/people", headers={"X-Admin-Token": "nope"})
        d = await ac.delete("/admin/person/70", headers={"X-Admin-Token": "nope"})
    assert r.status_code == 401 and d.status_code == 401


@pytest.mark.asyncio
async def test_admin_people_contract(server_module, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    async with client(server_module.app) as ac:
        r = await ac.get("/admin/people", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200
    assert set(r.json()) == {"ok", "request_id", "count", "people"}


@pytest.mark.asyncio
async def test_admin_delete_contract(server_module, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    server_module.PIPE.db.delete_person = lambda pid: 3
    async with client(server_module.app) as ac:
        r = await ac.delete("/admin/person/70", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "request_id", "person_id", "result"}
    assert body["result"] == {"removed_via": "PIPE.db.delete_person", "result": 3}
