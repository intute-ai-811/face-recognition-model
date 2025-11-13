# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

# ---- Environment & runtime tuning ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Non-root user
ARG APP_USER=app
ARG APP_GROUP=app
ARG APP_UID=10001
RUN addgroup --system ${APP_GROUP} \
 && adduser  --system --uid ${APP_UID} --ingroup ${APP_GROUP} --home /home/${APP_USER} ${APP_USER}

# Caches
ENV HOME=/home/${APP_USER} \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache \
    TORCH_HOME=/home/${APP_USER}/.cache/torch
RUN mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

WORKDIR /code

# ---- Copy only requirements first (better caching) ----
COPY requirements.txt .

# ---- Build & runtime system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Optional: slim the image
RUN apt-get purge -y --auto-remove build-essential g++ cmake || true && \
    rm -rf /var/lib/apt/lists/*

# ---- App code ----
# Copy source code and models, making sure permissions belong to the app user
COPY --chown=${APP_USER}:${APP_GROUP} . /code
# If you want to be explicit (no-op if already present in repo):
# COPY --chown=${APP_USER}:${APP_GROUP} models/ /code/models/

# ---- Ensure runtime dirs exist & are writable by the non-root user ----
# Create /code/data and a default faces subdir so imports that mkdir won't fail
RUN mkdir -p /code/data/faces /code/models \
 && chown -R ${APP_USER}:${APP_GROUP} /code /home/${APP_USER}

USER ${APP_USER}

EXPOSE 8000

# Basic liveness check (adjust path if your API differs)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request as u,sys; \
    sys.exit(0 if u.urlopen('http://127.0.0.1:8000/docs', timeout=3).status < 500 else 1)" || exit 1

# Start FastAPI
CMD uvicorn app.api.server:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}