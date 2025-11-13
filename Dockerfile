# syntax=docker/dockerfile:1.7

# Base image with Python
FROM python:3.11-slim-bookworm

# ---- Environment & runtime tuning ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Create a non-root user WITH a real home so libs can cache safely
ARG APP_USER=app
ARG APP_GROUP=app
ARG APP_UID=10001
RUN addgroup --system ${APP_GROUP} \
 && adduser --system --uid ${APP_UID} --ingroup ${APP_GROUP} --home /home/${APP_USER} ${APP_USER}

# Set cache locations to writable paths under the user's home
ENV HOME=/home/${APP_USER} \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache \
    TORCH_HOME=/home/${APP_USER}/.cache/torch

# Ensure cache dirs exist
RUN mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

# Set a dedicated workdir
WORKDIR /code

# ---- System deps (opencv/mediapipe/torch helpers) ----
# libgl1: OpenCV GUI-less requirement
# libglib2.0-0: OpenCV/mediapipe runtime
# libsm6, libxext6: often required by cv2 for image ops
# ffmpeg: for video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps (cache-friendly) ----
# Copy only requirements first to leverage layer caching
COPY requirements.txt .

# If you pin heavy libs (torch/torchvision/mediapipe/opencv-python),
# wheels will be pulled; using BuildKit pip cache speeds up rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# ---- App code ----
# Copy the rest of your project into /code
COPY . .

# Make sure both /code and the user's home (cache dirs) are owned by the non-root user
RUN chown -R ${APP_USER}:${APP_GROUP} /code /home/${APP_USER}

# Switch to non-root for security
USER ${APP_USER}

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with uvicorn
# Shell form so env vars expand correctly
# If your app entry changes, update "app.main:app"
CMD uvicorn app.main:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}