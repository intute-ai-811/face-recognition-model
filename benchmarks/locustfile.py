"""
Shift-change load test for /mark_attendance.

NEVER point this at production: every recognised face triggers an ERP call.
Run the API locally (or on a staging copy) with ERP_ATTENDANCE_URL aimed at
benchmarks/mock_erp.py.

    python benchmarks/mock_erp.py --port 9999 &
    ERP_ATTENDANCE_URL=http://127.0.0.1:9999/attendance \
        uvicorn app.api.server:app --host 127.0.0.1 --port 6900 &
    LOAD_TEST_IMAGES=test.jpg:rhythm1.jpeg \
        uvx locust -f benchmarks/locustfile.py --host http://127.0.0.1:6900 \
        --headless -u 50 -r 50 -t 60s --csv results/shift_change

-u 50 -r 50 starts 50 kiosk users at once (a shift-change burst); each user
posts an image, then waits 1-3 s, like people queueing at a kiosk.
"""
import itertools
import os
from pathlib import Path

from locust import HttpUser, between, task

_paths = [p for p in os.environ.get("LOAD_TEST_IMAGES", "").split(":") if p]
if not _paths:
    raise SystemExit("set LOAD_TEST_IMAGES=img1.jpg:img2.jpg")
_IMAGES = [(Path(p).name, Path(p).read_bytes()) for p in _paths]
_cycle = itertools.cycle(_IMAGES)


class Kiosk(HttpUser):
    wait_time = between(1, 3)

    @task
    def mark_attendance(self):
        name, data = next(_cycle)
        with self.client.post(
            "/mark_attendance",
            files={"file": (name, data, "image/jpeg")},
            name="/mark_attendance",
            catch_response=True,
        ) as r:
            # 200 marked, 404 no_face/no_match are valid outcomes; anything else is an error.
            if r.status_code in (200, 404):
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")
