# VoxScribe local engine. Builds natively for arm64 on Apple Silicon.
# This container runs the local voice engine. Mic capture is in the Electron client.
FROM python:3.12-slim

# ffmpeg is here for audio decoding when a real ASR backend is enabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for reproducible dependency resolution.
RUN pip install --no-cache-dir uv

WORKDIR /srv

# Copy dependency files and the pinned core wheel before the app source so
# Docker can cache the install layer when only source code changes.
COPY pyproject.toml uv.lock ./
COPY wheels/ ./wheels/

# Install the engine and its runtime dependencies (no dev extras).
RUN uv sync --frozen --no-dev

COPY app ./app

# Data directory for the database and recorded sessions (mount a volume here).
ENV DATA_DIR=/data
RUN mkdir -p /data

# The engine binds to 127.0.0.1 (loopback) by design. The HEALTHCHECK runs
# inside the container where loopback is reachable. External clients cannot
# reach the engine through this exposed port; authentication and a configurable
# bind address are owned by a later task (PL08, PL27).
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

# VOXSCRIBE_DATABASE_URL must be set (via env_file or -e) for the engine to start.
CMD ["python", "-m", "app.local.engine", "--port", "8000"]
