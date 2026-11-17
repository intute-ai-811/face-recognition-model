"""
Stand-in ERP endpoint for load tests, so a benchmark can never mark real attendance.

    python benchmarks/mock_erp.py --port 9999
    ERP_ATTENDANCE_URL=http://127.0.0.1:9999/attendance uvicorn app.api.server:app --port 6900
"""
import argparse

from aiohttp import web


async def handle(request: web.Request) -> web.Response:
    await request.read()
    return web.json_response({"status": "ok", "mock": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999)
    args = ap.parse_args()
    app = web.Application()
    app.router.add_post("/{tail:.*}", handle)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
