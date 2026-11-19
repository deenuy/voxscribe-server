# VoxScribe local engine. Builds natively for arm64 on Apple Silicon.
# This container runs the local engine. Mic capture is in the Electron client.
FROM python:3.12-slim

# ffmpeg is here for audio decoding when a real ASR backend is enabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image at a pinned version.
# Never use pip to install uv: pip is not used in this project.
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /usr/local/bin/uv

WORKDIR /srv

# Copy the dependency manifest, the lock file and the pinned core wheel first.
# The source package is NOT present yet; --no-install-project installs only the
# third-party dependencies so this layer is cached when only source code changes.
COPY pyproject.toml uv.lock README.md ./
COPY wheels/ ./wheels/
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the application source and install the project itself.
# engine_version() reads importlib.metadata, which is populated by this install.
COPY app ./app
RUN uv sync --frozen --no-dev

# Put the venv on PATH so the CMD python resolves to the venv interpreter.
ENV PATH="/srv/.venv/bin:$PATH"

# Data directory for the database and recorded sessions (mount a volume here).
ENV DATA_DIR=/data
RUN mkdir -p /data

# Note on PortAudio / sounddevice:
#   sounddevice is in the `wake` optional extra and is NOT installed here.
#   The CMD below does not pass --wake, so the PortAudio adapter is never
#   imported and libportaudio2 is not needed. If --wake is ever added to the
#   CMD, install libportaudio2 and add voxscribe-server[wake] to the sync.
#
# Note on the exposed port:
#   The engine binds to 127.0.0.1 (loopback) by design. The HEALTHCHECK runs
#   inside the container where loopback is reachable. The host-mapped port
#   8000 is NOT reachable from outside the container. A configurable bind
#   address is owned by task PL27.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

# VOXSCRIBE_DATABASE_URL must be set (via env_file or -e) for the engine to start.
CMD ["python", "-m", "app.local.engine", "--port", "8000"]
