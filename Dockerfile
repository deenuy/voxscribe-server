# VoxScribe processing backend. Builds natively for arm64 on Apple Silicon.
# This container does LLM processing only. Mic capture is in the Electron client.
FROM python:3.12-slim

# ffmpeg is here for audio decoding when a real ASR backend is enabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Recorded sessions and the SQLite db live on a mounted volume (see compose).
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker keeps the in-memory SSE hub correct. For multiple workers,
# move the progress hub to Redis pub/sub.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
