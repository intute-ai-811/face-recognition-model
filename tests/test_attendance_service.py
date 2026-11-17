"""Pins the exact check-in vs check-out differences now expressed as data in app/services/attendance.py."""
import httpx
import numpy as np
import pytest

IMG = ("x.jpg", b"abc", "image/jpeg")


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test")


@pytest.fixture
def setup(server_module, monkeypatch):
    monkeypatch.setattr(server_module.cv2, "imdecode", lambda arr, mode: np.zeros((16, 16, 3), np.uint8))
    sent = []

    def use(preds, status=200, resp=None, exc=None):
        class Pipe:
            db = server_module.PIPE.db

            def recognize_image(self, img):
                return preds

        server_module.PIPE = Pipe()
        import app.clients.http as http_client

        async def fake(url, payload):
            sent.append(payload)
            if exc:
                raise exc
            return status, resp if resp is not None else {"ok": True}

        monkeypatch.setattr(http_client, "post_json", fake)

    return use, sent


MATCH = [{"prediction": {"person_id": "72", "similarity": 0.8}, "top_k": []}]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,payload", [
    ("/mark_attendance", {"user_id": 72}),
    ("/logout", {"user_id": 72, "type": "OUT"}),
])
async def test_erp_payload(server_module, setup, path, payload):
    use, sent = setup
    use(MATCH)
    async with client(server_module.app) as ac:
        await ac.post(path, files={"file": IMG})
    assert sent == [payload]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,erp_resp,message", [
    ("/mark_attendance", {"ok": True}, "Attendance marked successfully."),
    ("/mark_attendance", {"status": "already_checked_in", "attendance": {"check_in_time_ist_text": "09:01"}}, "Already checked in at 09:01"),
    ("/mark_attendance", {"status": "already_checked_in"}, "Already checked in."),
    ("/logout", {"ok": True}, "Logout marked successfully."),
    ("/logout", {"status": "already_checked_out", "attendance": {"check_out_time_ist_text": "18:02"}}, "Already logged out at 18:02"),
    ("/logout", {"status": "already_checked_out"}, "Already logged out."),
    # The other action's "already" status must not be interpreted.
    ("/logout", {"status": "already_checked_in", "attendance": {"check_in_time_ist_text": "09:01"}}, "Logout marked successfully."),
])
async def test_success_messages(server_module, setup, path, erp_resp, message):
    use, _ = setup
    use(MATCH, resp=erp_resp)
    async with client(server_module.app) as ac:
        r = await ac.post(path, files={"file": IMG})
    assert r.status_code == 200 and r.json()["message"] == message


@pytest.mark.asyncio
@pytest.mark.parametrize("path,verb", [("/mark_attendance", "Attendance"), ("/logout", "Logout")])
async def test_failure_messages(server_module, setup, path, verb):
    use, _ = setup
    async with client(server_module.app) as ac:
        use([{"prediction": {"person_id": "unknown", "similarity": 0.2}, "top_k": []}])
        assert (await ac.post(path, files={"file": IMG})).json()["message"] == f"Face not recognized. {verb} not marked."
        use([{"prediction": {"person_id": "abc", "similarity": 0.9}, "top_k": []}])
        assert (await ac.post(path, files={"file": IMG})).json()["message"] == f"Recognized user id is not numeric. {verb} not marked."
        use(MATCH, exc=RuntimeError("down"))
        assert (await ac.post(path, files={"file": IMG})).json()["message"] == f"Face recognized but failed to mark {verb.lower()} in ERP."
        use(MATCH, status=500)
        assert (await ac.post(path, files={"file": IMG})).json()["message"] == "Face recognized but ERP returned an error."
