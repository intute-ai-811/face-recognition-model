# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   builder - compiles/installs locked dependencies into /opt/venv (needs a C++ toolchain for insightface)
#   runtime - slim image with only the venv, the app code and two shared libraries
#
# Models, data and credentials are NOT in the image; they are mounted at runtime
# (see .github/workflows/attendance.yml).

# python:3.13.18-slim-bookworm, pinned by digest for reproducible builds.
ARG PYTHON_IMAGE=python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72

# ─────────────────────────────── builder ───────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential g++ \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN python -m pip install uv==0.13.8

WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
# --frozen: install exactly what uv.lock pins; fail if the lock is out of date.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────── runtime ───────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=6900

# libglib2.0-0: OpenCV; libgomp1: ONNX Runtime / OpenMP
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ARG APP_USER=app
ARG APP_GROUP=app
ARG APP_UID=10001
RUN addgroup --system ${APP_GROUP} \
 && adduser --system --uid ${APP_UID} --ingroup ${APP_GROUP} --home /home/${APP_USER} ${APP_USER}
ENV HOME=/home/${APP_USER} \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache

COPY --from=builder /opt/venv /opt/venv

WORKDIR /code
COPY --chown=${APP_USER}:${APP_GROUP} app/ /code/app/

# Mount points; real content comes from host volumes.
RUN mkdir -p /code/data/faces /code/models "$XDG_CACHE_HOME" \
 && chown -R ${APP_USER}:${APP_GROUP} /code/data /code/models /home/${APP_USER}

USER ${APP_USER}

EXPOSE 6900

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import os,urllib.request as u,sys; \
    sys.exit(0 if u.urlopen('http://127.0.0.1:%s/healthz' % os.getenv('UVICORN_PORT','6900'), timeout=3).status == 200 else 1)" || exit 1

# exec so uvicorn is PID 1 and receives SIGTERM for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn app.api.server:app --host \"$UVICORN_HOST\" --port \"$UVICORN_PORT\""]
